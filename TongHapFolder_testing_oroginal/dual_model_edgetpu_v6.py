import csv
import cv2
import numpy as np
import argparse
import time
import threading
import queue
from pathlib import Path

# ──────────────────────────────────────────────
# Runtime 로드
# ──────────────────────────────────────────────
try:
    from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus

    EDGETPU_AVAILABLE = True
except ImportError:
    EDGETPU_AVAILABLE = False

    def list_edge_tpus():
        return []


try:
    import tflite_runtime.interpreter as tflite

    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite

        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
LANE_CONF_THRESH = 0.45
LANE_NMS_IOU = 0.35

OBS_CONF_THRESH = 0.75
OBS_NMS_IOU = 0.35
DRAW_LANE_MASK = False  # ★ True면 세그 마스크 오버레이, False면 bbox만

LANE_NAMES = {0: "curve", 1: "eeu", 2: "line"}
LANE_COLOR = {0: (0, 255, 255), 1: (0, 255, 0), 2: (0, 0, 255)}
OBS_PALETTE = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 200, 80),
    (200, 80, 255),
    (80, 255, 200),
]

OBS_CLASS_NAMES = [
    "KNU",
    "SL",
    "box",
    "car",
    "parking",
    "person",
]

INTER_DESC = {
    "no_lane": "No Lane",
    "straight": "Straight",
    "curve_left": "Curve-L",
    "left_t": "T-Left",
    "down_t": "T-Down",
    "cross": "Cross(+)",
}

# ROI: 상단 몇 % 를 제거할지 (0.0 ~ 1.0)
LANE_ROI_TOP = 0.0


# ══════════════════════════════════════════════
# EdgeTPU 엔진
# ══════════════════════════════════════════════
class EdgeTPUEngine:
    def __init__(self, model_path: str, use_edgetpu=True, device=""):
        if use_edgetpu and EDGETPU_AVAILABLE:
            kw = {"device": device} if device else {}
            self.interp = make_interpreter(model_path, **kw)
            self.on_tpu = True
            tag = f"EdgeTPU[{device if device else 'auto'}]"
        elif TFLITE_AVAILABLE:
            self.interp = tflite.Interpreter(model_path=model_path, num_threads=3)
            self.on_tpu = False
            tag = "CPU-TFLite"
        else:
            raise RuntimeError("pycoral 또는 tflite_runtime 없음")

        self.interp.allocate_tensors()
        self.ind = self.interp.get_input_details()
        self.outd = self.interp.get_output_details()
        in0 = self.ind[0]
        self.in_scale, self.in_zp = in0["quantization"]
        self.in_dtype = in0["dtype"]
        sh = in0["shape"]
        self.img_h, self.img_w = int(sh[1]), int(sh[2])

        # preprocess 상수 미리 계산 (매 프레임 재계산 방지)
        if self.in_dtype == np.int8:
            if self.in_scale > 0:
                # LUT: uint8(0~255) → int8 변환 테이블 미리 생성
                # float 연산을 완전히 없애고 cv2.LUT로 대체
                lut = (
                    np.arange(256, dtype=np.float32) / 255.0 / self.in_scale
                    + self.in_zp
                )
                self._lut = np.clip(lut, -128, 127).astype(np.int8)
            else:
                lut = np.arange(256, dtype=np.float32) - 128.0
                self._lut = np.clip(lut, -128, 127).astype(np.int8)
            self._pre_scale = True  # LUT 사용 표시
        elif self.in_dtype == np.uint8:
            if self.in_scale > 0:
                lut = (
                    np.arange(256, dtype=np.float32) / 255.0 / self.in_scale
                    + self.in_zp
                )
                self._lut = np.clip(lut, 0, 255).astype(np.uint8)
                self._pre_scale = True
            else:
                self._lut = None
                self._pre_scale = None
        else:
            self._lut = None
            self._pre_scale = None
        self.pre_ms = 0.0
        self.invoke_ms = 0.0
        print(
            f"  [{tag}] {Path(model_path).name}  ({self.img_h}x{self.img_w})  {self.in_dtype.__name__}"
        )
        for i, od in enumerate(self.outd):
            print(f"    out[{i}] {od['shape']}  {od['dtype'].__name__}")

        # EdgeTPU ops 진단: 몇 개 레이어가 실제로 TPU에서 실행되는지 출력
        if self.on_tpu:
            try:
                details = self.interp.get_tensor_details()
                tpu_ops = sum(
                    1
                    for d in self.interp._get_ops_details()
                    if "edgetpu" in str(d).lower() or "delegate" in str(d).lower()
                )
                print(f"    [TPU diag] tensors={len(details)}")
            except Exception:
                pass
            # 첫 invoke로 실제 실행 시간 측정
            try:
                dummy = np.zeros((1, self.img_h, self.img_w, 3), dtype=self.in_dtype)
                self.interp.set_tensor(self.ind[0]["index"], dummy)
                _t = time.perf_counter()
                self.interp.invoke()
                _ms = (time.perf_counter() - _t) * 1000.0
                print(
                    f"    [TPU diag] 워밍업 invoke={_ms:.1f}ms  "
                    f"({'EdgeTPU 정상' if _ms < 50 else '⚠ CPU fallback 의심 >50ms'})"
                )
            except Exception as e:
                print(f"    [TPU diag] 워밍업 실패: {e}")

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        # ① BGR→RGB + 리사이즈를 한 번에 (cvtColor 후 resize)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(
            rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
        )

        if self._lut is not None:
            # ② LUT 적용: float 변환 없이 uint8→int8 직접 매핑 (cv2.LUT는 C++ 수준)
            # cv2.LUT는 uint8 출력만 지원하므로 view로 int8 재해석
            mapped = cv2.LUT(small, self._lut.view(np.uint8))
            return np.expand_dims(mapped.view(self.in_dtype), 0)

        # LUT 없는 경우 (float32 모델)
        return np.expand_dims(small.astype(np.float32) * (1.0 / 255.0), 0)

    def dequant(self, t, idx):
        sc, zp = self.outd[idx]["quantization"]
        if self.outd[idx]["dtype"] in (np.uint8, np.int8) and sc > 0:
            return (t.astype(np.float32) - zp) * sc
        return t.astype(np.float32)

    def infer(self, bgr: np.ndarray):
        t0 = time.perf_counter()
        inp = self.preprocess(bgr)
        t1 = time.perf_counter()
        self.interp.set_tensor(self.ind[0]["index"], inp)
        self.interp.invoke()
        t2 = time.perf_counter()
        self.pre_ms = (t1 - t0) * 1000.0
        self.invoke_ms = (t2 - t1) * 1000.0
        ms = (t2 - t0) * 1000.0
        outs = [
            self.dequant(self.interp.get_tensor(od["index"]), i)
            for i, od in enumerate(self.outd)
        ]
        return outs, ms


# ══════════════════════════════════════════════
# Lane 후처리
# ══════════════════════════════════════════════
def parse_lane_det(outputs, H, W):
    """
    세그멘테이션 없는 순수 detection lane 모델용
    출력: [1, 7, 2100] = 4(bbox) + 3(classes: curve/eeu/line)
    """
    shapes = []
    if not outputs:
        return shapes

    det_raw = outputs[0][0]  # [7, 2100]
    det = det_raw if det_raw.shape[0] < det_raw.shape[1] else det_raw.T
    # det shape: [7, 2100]

    nc = det.shape[0] - 4
    if nc <= 0:
        return shapes

    boxes = det[:4, :].T  # [2100, 4]
    scores = det[4 : 4 + nc, :].T  # [2100, 3]

    cids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)

    # print(f"[LANE_DEBUG] max_conf={confs.max():.3f}, above_thresh={np.sum(confs > LANE_CONF_THRESH)}")

    keep = confs > LANE_CONF_THRESH
    if not np.any(keep):
        return shapes

    boxes = boxes[keep]
    confs = confs[keep]
    cids = cids[keep]

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = ((cx - bw / 2) * W).astype(int)
    y1 = ((cy - bh / 2) * H).astype(int)
    x2 = ((cx + bw / 2) * W).astype(int)
    y2 = ((cy + bh / 2) * H).astype(int)

    idx = cv2.dnn.NMSBoxes(
        np.stack([x1, y1, x2 - x1, y2 - y1], 1).tolist(),
        confs.tolist(),
        LANE_CONF_THRESH,
        LANE_NMS_IOU,
    )
    if len(idx) == 0:
        return shapes

    for ii in idx:
        i = int(ii)
        cid = int(cids[i])
        bx1 = int(np.clip(x1[i], 0, W - 1))
        by1 = int(np.clip(y1[i], 0, H - 1))
        bx2 = int(np.clip(x2[i], 0, W - 1))
        by2 = int(np.clip(y2[i], 0, H - 1))
        cy_norm = float(((by1 + by2) / 2) / H)
        if cy_norm < LANE_ROI_TOP:
            continue
        shapes.append(
            {
                "class_id": cid,
                "class_name": LANE_NAMES.get(cid, "?"),
                "cx_norm": float(((bx1 + bx2) / 2) / W),
                "cy_norm": cy_norm,
                "bbox": (bx1, by1, bx2, by2),
                "conf": float(confs[i]),
                "mask": None,
            }
        )

    return _merge_lane_shapes(shapes, H, W)


def parse_lane(outputs, H, W):
    """
    새 lane 모델은 detection-only 후처리 사용.
    기존 seg용 parse_lane()은 사용하지 않음.
    """
    return parse_lane_det(outputs, H, W)


# ══════════════════════════════════════════════
# Lane 로직
# ══════════════════════════════════════════════
# =========================
# 3x3 zone helper
# =========================
_X1 = 1.0 / 3.0
_X2 = 2.0 / 3.0
_Y1 = 1.0 / 3.0
_Y2 = 2.0 / 3.0

_LANE_MID_X = 0.5

_ZONE_W_TOP = 0.20
_ZONE_W_MID = 0.30
_ZONE_W_BOTTOM = 0.50


def _x_zone(cx: float) -> str:
    if cx < _X1:
        return "left"
    if cx < _X2:
        return "center"
    return "right"


def _y_zone(cy: float) -> str:
    if cy < _Y1:
        return "top"
    if cy < _Y2:
        return "mid"
    return "bottom"


def _zone_weight(cy: float) -> float:
    z = _y_zone(cy)
    if z == "bottom":
        return _ZONE_W_BOTTOM
    if z == "mid":
        return _ZONE_W_MID
    return _ZONE_W_TOP


def _has_class(shapes, cls_name, x=None, y=None):
    for s in shapes:
        if s["class_name"] != cls_name:
            continue
        if x is not None and _x_zone(float(s["cx_norm"])) != x:
            continue
        if y is not None and _y_zone(float(s["cy_norm"])) != y:
            continue
        return True
    return False


def _count_class(shapes, cls_name, x=None, y=None):
    c = 0
    for s in shapes:
        if s["class_name"] != cls_name:
            continue
        if x is not None and _x_zone(float(s["cx_norm"])) != x:
            continue
        if y is not None and _y_zone(float(s["cy_norm"])) != y:
            continue
        c += 1
    return c


def compute_lane_error(shapes):
    """
    straight용 angle 계산
    - line만 사용
    - 상/중/하 전체 사용
    - center only는 제외
    - left only / right only 허용
    - 하단에 더 큰 가중치 부여
    """
    lines = [s for s in shapes if s["class_name"] == "line"]
    if not lines:
        return 0.0, "lost"

    left_pts = []
    right_pts = []

    for s in lines:
        cx = float(s["cx_norm"])
        cy = float(s["cy_norm"])
        w = _zone_weight(cy)
        xz = _x_zone(cx)

        if xz == "left":
            left_pts.append((cx, w))
        elif xz == "right":
            right_pts.append((cx, w))

    def weighted_mean(items):
        if not items:
            return None
        ws = sum(w for _, w in items)
        if ws <= 1e-6:
            return None
        return sum(x * w for x, w in items) / ws

    left_mean = weighted_mean(left_pts)
    right_mean = weighted_mean(right_pts)

    if left_mean is None and right_mean is None:
        return 0.0, "lost"

    if left_mean is not None and right_mean is not None:
        lane_center = (left_mean + right_mean) * 0.5
        err = ((_LANE_MID_X - lane_center) / 0.5) * 10.0
        err = float(np.clip(err, -10.0, 10.0))
        return err, "ok"

    if left_mean is not None:
        err = ((_LANE_MID_X - left_mean) / 0.5) * 10.0
        err = float(np.clip(err, -10.0, 10.0))
        return err, "left_only"

    err = ((_LANE_MID_X - right_mean) / 0.5) * 10.0
    err = float(np.clip(err, -10.0, 10.0))
    return err, "right_only"


def _raw_classify(shapes):
    left_bottom_curve = _has_class(shapes, "curve", x="left", y="bottom")

    mid_eeu = _has_class(shapes, "eeu", y="mid")
    bottom_eeu = _has_class(shapes, "eeu", y="bottom")
    right_line_any = _has_class(shapes, "line", x="right")
    has_line_left = _has_class(shapes, "line", x="left")

    eeu_bottom_left = _count_class(shapes, "eeu", x="left", y="bottom")
    eeu_bottom_center = _count_class(shapes, "eeu", x="center", y="bottom")
    eeu_bottom_right = _count_class(shapes, "eeu", x="right", y="bottom")
    eeu_bottom_wide = (
        sum(
            [
                eeu_bottom_left > 0,
                eeu_bottom_center > 0,
                eeu_bottom_right > 0,
            ]
        )
        >= 2
    )

    curve_mid_left = _has_class(shapes, "curve", x="left", y="mid")
    curve_mid_right = _has_class(shapes, "curve", x="right", y="mid")
    curve_bottom_left = _has_class(shapes, "curve", x="left", y="bottom")
    curve_bottom_right = _has_class(shapes, "curve", x="right", y="bottom")
    cross_cond = (curve_mid_left and curve_mid_right) or (
        curve_bottom_left and curve_bottom_right
    )

    if left_bottom_curve:
        return "left_t"
    if eeu_bottom_wide:
        return "down_t"
    if cross_cond and not eeu_bottom_wide:
        return "cross"
    if (mid_eeu or bottom_eeu) and right_line_any and not has_line_left:
        return "left_t"
    return None


def _resolve_cross_down(shapes):
    raw = _raw_classify(shapes)
    if raw in ("left_t", "down_t", "cross"):
        return raw
    return None


class IntersectionFSM:
    def __init__(self, *args, **kwargs):
        self.cur = None

    def update(self, shapes):
        raw = _raw_classify(shapes)
        resolved = _resolve_cross_down(shapes)
        self.cur = resolved
        if resolved is None:
            return (False, None), raw
        return (True, resolved), raw


OBS_OUTPUT_ID = {
    "KNU": 10,
    "box": 10,
    "SL": 2,
    "person": 3,
    "car": 4,
    "parking": 5,
}


def make_output(le, ls, is_inter, itype, obs_list):
    """
    ID 체계
      0  = 차선 없음
      1  = 일반 차선
      2  = SL 감지
      3  = person 감지
      4  = car 감지
      5  = parking 감지
      6  = left_t
      8  = down_t
      9  = cross
      10 = KNU / box / 물류 pass 계열
    """
    err = round(le, 2)
    result = []

    if is_inter:
        inter_id = {
            "left_t": 6,
            "down_t": 8,
            "cross": 9,
        }.get(itype)
        if inter_id is not None:
            result.append([inter_id, err])
        else:
            result.append([1, err])
    elif ls == "lost":
        result.append([0, 0.0])
    else:
        result.append([1, err])

    for o in obs_list:
        oid = OBS_OUTPUT_ID.get(o["class_name"])
        if oid is not None:
            result.append([oid, None])

    return result


# ══════════════════════════════════════════════
# 스레드 클래스들
# ══════════════════════════════════════════════

import subprocess


class FrameGrabber(threading.Thread):
    """
    is_file=True   → VideoCapture(파일) 큐 방식
    is_file=False, use_csi=True  → libcamera-vid 파이프 (라즈베리파이5 CSI)
    is_file=False, use_csi=False → VideoCapture(웹캠/V4L2) 최신 프레임 방식
    """

    def __init__(self, cap, is_file=False, use_csi=False, cam_w=320, cam_h=320, fps=30):
        super().__init__(daemon=True)
        self.cap = cap
        self.is_file = is_file
        self.use_csi = use_csi
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.fps = fps
        self._f = None
        self._lk = threading.Lock()
        self._q = queue.Queue(maxsize=8) if is_file else None
        self.alive = True
        self.eof = False
        self._proc = None

    def run(self):
        if self.is_file:
            self._run_file()
        elif self.use_csi:
            self._run_csi()
        else:
            self._run_v4l2()

    def _run_file(self):
        while self.alive:
            ok, f = self.cap.read()
            if ok:
                try:
                    self._q.put(f, timeout=2.0)
                except queue.Full:
                    pass
            else:
                self.eof = True
                break

    def _run_csi(self):
        """rpicam-vid stdout MJPEG BGR"""
        cmd = [
            "rpicam-vid",
            "--width",
            str(self.cam_w),
            "--height",
            str(self.cam_h),
            "--framerate",
            str(self.fps),
            "--codec",
            "mjpeg",
            "--nopreview",
            "--timeout",
            "0",
            "-o",
            "-",
        ]
        print(f"[INFO] rpicam-vid: {self.cam_w}x{self.cam_h} @ {self.fps}fps (mjpeg)")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1024 * 1024,
            )
            buf = b""
            while self.alive:
                chunk = self._proc.stdout.read(65536)
                if not chunk:
                    self.eof = True
                    break
                buf += chunk
                while True:
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2)
                    if start == -1 or end == -1:
                        break
                    jpg = buf[start : end + 2]
                    buf = buf[end + 2 :]
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        with self._lk:
                            self._f = frame
        except Exception as e:
            print(f"[ERR] rpicam-vid: {e}")
            self.eof = True

    def _run_v4l2(self):
        while self.alive:
            ok = self.cap.grab()
            if not ok:
                self.eof = True
                break
            for _ in range(2):
                self.cap.grab()
            ok, f = self.cap.retrieve()
            if ok:
                with self._lk:
                    self._f = f
            else:
                self.eof = True
                break

    def get(self):
        if self.is_file:
            try:
                return self._q.get(timeout=1.0)
            except queue.Empty:
                return None
        else:
            with self._lk:
                return None if self._f is None else self._f.copy()

    def stop(self):
        self.alive = False
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass


class InferWorker(threading.Thread):
    def __init__(self, engine, name):
        super().__init__(daemon=True, name=name)
        self.engine = engine
        self.alive = True
        self._in = None
        self._in_lk = threading.Lock()
        self._out = None
        self._out_lk = threading.Lock()
        self._in_ev = threading.Event()
        self._out_ev = threading.Event()

    def push(self, frame):
        self._out_ev.clear()
        with self._in_lk:
            self._in = frame
        self._in_ev.set()

    def result_wait(self, timeout=2.0):
        self._out_ev.wait(timeout)
        with self._out_lk:
            return self._out

    def run(self):
        while self.alive:
            self._in_ev.wait(1.0)
            self._in_ev.clear()
            with self._in_lk:
                f = self._in
            if f is None:
                continue
            r = self.engine.infer(f)
            with self._out_lk:
                self._out = r
            self._out_ev.set()

    def stop(self):
        self.alive = False


class SaveThread(threading.Thread):
    def __init__(self, writer):
        super().__init__(daemon=True)
        self.writer = writer
        self.q = queue.Queue(maxsize=4)
        self.alive = True

    def push(self, data):
        try:
            self.q.put_nowait(data)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(data)
            except:
                pass

    def run(self):
        while self.alive:
            try:
                d = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            self.writer.write(_draw_frame(d))

    def stop(self):
        self.alive = False
        time.sleep(0.3)
        while not self.q.empty():
            try:
                self.writer.write(_draw_frame(self.q.get_nowait()))
            except:
                break
        self.writer.release()
        print("[INFO] 영상 저장 완료")


class CsvLogger:
    CSV_HEADER = [
        "timestamp",
        "frame_idx",
        "lane_error",
        "lane_status",
        "inter_type",
        "lane_ms",
        "obs_ms",
        "fps",
        "lane_count",
        "lane_classes",
        "lane_confs",
        "obs_count",
        "obs_classes",
        "obs_conf",
        "output_packets",
    ]

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(self.CSV_HEADER)
        self._lk = threading.Lock()
        print(f"[INFO] CSV 로그 저장: {path}")

    def log(
        self,
        frame_idx,
        le,
        ls,
        is_inter,
        itype,
        lane_ms,
        obs_ms,
        fps_val,
        shapes,
        obs_list,
        out,
    ):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        inter_type = itype if is_inter and itype else ""

        lane_classes = "|".join(s["class_name"] for s in shapes)
        lane_confs = "|".join(f"{s['conf']:.3f}" for s in shapes)

        obs_classes = "|".join(o["class_name"] for o in obs_list)
        obs_conf = f"{obs_list[0]['conf']:.3f}" if obs_list else ""

        packets = " ".join(str(pkt) for pkt in out)
        row = [
            ts,
            frame_idx,
            f"{le:+.4f}",
            ls,
            inter_type,
            f"{lane_ms:.1f}",
            f"{obs_ms:.1f}",
            f"{fps_val:.1f}",
            len(shapes),
            lane_classes,
            lane_confs,
            len(obs_list),
            obs_classes,
            obs_conf,
            packets,
        ]
        with self._lk:
            self._w.writerow(row)

    def close(self):
        with self._lk:
            self._f.flush()
            self._f.close()
        print(f"[INFO] CSV 저장 완료: {self.path}")


# ══════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════
def main_loop(
    lane_eng,
    obs_eng,
    grabber,
    coral,
    fps,
    obs_skip,
    save_video,
    disp_scale,
    orig_h,
    orig_w,
    source,
    total_frames=0,
    save_fps=30,
):

    is_file = (
        (source != "csi")
        and (not isinstance(source, int))
        and Path(str(source)).exists()
    )
    mode_str = f"{coral}-Coral"

    src_stem = Path(str(source)).stem if Path(str(source)).exists() else "cam"
    csv_path = f"log_{src_stem}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_logger = CsvLogger(csv_path)

    save_th = None
    if save_video:
        src_name = Path(str(source)).stem if is_file else "cam"
        outp = f"result_{src_name}.mp4"
        dw = int(orig_w * disp_scale)
        dh = int(orig_h * disp_scale)
        writer = cv2.VideoWriter(
            outp, cv2.VideoWriter_fourcc(*"mp4v"), save_fps, (dw, dh)
        )
        save_th = SaveThread(writer)
        save_th.start()
        print(f"[INFO] 영상 저장: {outp}  ({dw}x{dh} @ {save_fps:.2f}fps)")

    lane_w = obs_w = None
    if coral == 2:
        lane_w = InferWorker(lane_eng, "lane-w")
        obs_w = InferWorker(obs_eng, "obs-w")
        lane_w.start()
        obs_w.start()

    fsm = IntersectionFSM()
    interval = 1.0 / fps
    fc = 0
    fps_t0 = time.time()
    fps_val = 0.0
    frame_idx = 0
    obs_list = []
    obs_ms = 0.0
    lane_ms = 0.0
    prev_sh = []
    p_le = 0.0
    p_ls = "lost"
    p_is = False
    p_it = None
    p_raw = None

    try:
        while True:
            if grabber.eof:
                print("[INFO] 영상 재생 완료. 종료합니다.")
                break

            t_loop = time.perf_counter()
            frame = grabber.get()
            t_get = time.perf_counter()

            if frame is None:
                time.sleep(0.003)
                continue

            H, W = frame.shape[:2]
            fc += 1
            frame_idx += 1

            if coral == 2:
                lane_w.push(frame)
                obs_w.push(frame)

                lr = lane_w.result_wait(timeout=2.0)
                or_ = obs_w.result_wait(timeout=2.0)

                if lr:
                    lo, lane_ms = lr
                    shapes = parse_lane(lo, H, W)
                    le, ls = compute_lane_error(shapes)
                    (is_i, it), raw = fsm.update(shapes)
                    prev_sh = shapes
                    p_le = le
                    p_ls = ls
                    p_is = is_i
                    p_it = it
                    p_raw = raw

                if or_:
                    oo, obs_ms = or_
                    obs_list = parse_obstacle(oo, H, W)
            else:
                lo, lane_ms = lane_eng.infer(frame)
                prev_sh = parse_lane(lo, H, W)
                p_le, p_ls = compute_lane_error(prev_sh)
                (p_is, p_it), p_raw = fsm.update(prev_sh)
                if fc % obs_skip == 0:
                    oo, obs_ms = obs_eng.infer(frame)
                    obs_list = parse_obstacle(oo, H, W)

            el = time.time() - fps_t0
            if el >= 1.0:
                fps_val = fc / el
                fc = 0
                fps_t0 = time.time()

            t_proc = time.perf_counter()

            draw_data = {
                "frame": frame,
                "shapes": prev_sh,
                "obs": obs_list,
                "le": p_le,
                "ls": p_ls,
                "is_inter": p_is,
                "itype": p_it,
                "raw": p_raw,
                "lane_ms": lane_ms,
                "obs_ms": obs_ms,
                "fps_val": fps_val,
                "mode_str": mode_str,
                "scale": disp_scale,
                "progress": (frame_idx, total_frames) if total_frames > 0 else None,
            }

            if save_th:
                save_th.push(draw_data)

            if not is_file:
                sl = interval - (time.time() - t_loop)
                if sl > 0:
                    time.sleep(sl)

            out = make_output(p_le, p_ls, p_is, p_it, obs_list)

            if total_frames > 0:
                pct = frame_idx / total_frames * 100
                remain = total_frames - frame_idx
                remain_sec = remain / fps if fps > 0 else 0
                progress_str = f"[{pct:5.1f}%] {frame_idx}/{total_frames}f 남은프레임:{remain} ({remain_sec:.1f}s)"
            else:
                progress_str = ""

            print(
                f"[{mode_str}] "
                f"grab={(t_get-t_loop)*1000:.1f}ms "
                f"proc={(t_proc-t_get)*1000:.1f}ms "
                f"lane:{p_le:+.2f}|{p_ls}|{p_it}|"
                f"lane={lane_ms:.0f}ms"
                f"(pre={lane_eng.pre_ms:.1f}/inv={lane_eng.invoke_ms:.1f}) "
                f"obs={obs_ms:.0f}ms "
                f"FPS={fps_val:.1f}  {progress_str}"
            )
            print(f">>> {' '.join(str(pkt) for pkt in out)}")

            csv_logger.log(
                frame_idx=frame_idx,
                le=p_le,
                ls=p_ls,
                is_inter=p_is,
                itype=p_it,
                lane_ms=lane_ms,
                obs_ms=obs_ms,
                fps_val=fps_val,
                shapes=prev_sh,
                obs_list=obs_list,
                out=out,
            )

    finally:
        csv_logger.close()
        if save_th:
            save_th.stop()
        if lane_w:
            lane_w.stop()
        if obs_w:
            obs_w.stop()


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
def run(
    lane_model,
    obs_model,
    source,
    fps,
    coral,
    use_edgetpu,
    obs_skip,
    save_video,
    disp_scale,
    cam_w,
    cam_h,
):

    print(f"\n=== Dual-Model EdgeTPU v7 (latency reduced) ===")

    if use_edgetpu and EDGETPU_AVAILABLE:
        tpus = list_edge_tpus()
        print(f"[INFO] 연결된 EdgeTPU 장치: {len(tpus)}개")
        for i, t in enumerate(tpus):
            print(f"  [{i}] type={t.get('type','?')}  path={t.get('path','?')}")
        if coral == 2 and len(tpus) < 2:
            print(f"[WARN] --coral 2 지정했지만 장치가 {len(tpus)}개 감지됨!")
            print("       USB 전력 또는 연결 상태를 확인하세요.")

    if coral == 2 and use_edgetpu:
        print("[INFO] lane 모델 -> usb:0 로드 중...")
        lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=True, device="usb:0")
        print("[INFO] usb:0 완료. 1.0초 대기...")
        time.sleep(1.0)
        print("[INFO] obs  모델 -> usb:1 로드 중...")
        obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=True, device="usb:1")
        print("[INFO] usb:1 완료. 두 Coral 모두 점등 확인!\n")
    else:
        lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=use_edgetpu)
        obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=use_edgetpu)

    # 'csi' 키워드 또는 정수 인덱스는 카메라, 실제 존재하는 경로만 파일
    is_file = (
        (source != "csi")
        and (not isinstance(source, int))
        and Path(str(source)).exists()
    )

    use_csi = source == "csi"
    cap = None

    if not is_file:
        if use_csi:
            # ── libcamera-vid CSI 카메라 (라즈베리파이5) ──────────────────
            print(
                f"[INFO] libcamera-vid CSI 카메라 사용 (rpicam-vid): {cam_w}x{cam_h} @ {fps}fps"
            )
            orig_w, orig_h = cam_w, cam_h
            src_fps = float(fps)
            save_fps = float(fps)
        else:
            # ── 일반 V4L2 웹캠 ────────────────────────────────────────────
            v4l2_src = 0 if source == "csi" else source
            cap = cv2.VideoCapture(v4l2_src, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, fps)
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            save_fps = float(fps)
    # ── 파일 소스 ──────────────────────────────────────────────────────
    else:
        cap = cv2.VideoCapture(str(source))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        save_fps = src_fps if src_fps > 0 else fps
    print(
        f"[INFO] 해상도: {orig_w}x{orig_h}  obs_skip={obs_skip}  disp_scale={disp_scale}"
    )
    print(f"[INFO] 저장 FPS: {save_fps:.2f}  (원본: {src_fps:.2f})\n")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if (is_file and cap) else 0

    grabber = FrameGrabber(
        cap, is_file=is_file, use_csi=use_csi, cam_w=cam_w, cam_h=cam_h, fps=fps
    )
    grabber.start()
    try:
        main_loop(
            lane_eng,
            obs_eng,
            grabber,
            coral,
            fps,
            obs_skip,
            save_video,
            disp_scale,
            orig_h,
            orig_w,
            source,
            total_frames,
            save_fps,
        )
    finally:
        grabber.stop()
        if cap is not None:
            cap.release()
        print("[INFO] 종료")


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
FPS/지연 줄이기 추천:
  --coral 2
  --cam-w 320 --cam-h 320   (CSI 기본값)
  DRAW_LANE_MASK = False

CSI 카메라 사용 (라즈베리파이5 + Picamera2):
  python test_model_edgeTPU.py          # --source 생략 시 CSI 자동 사용
  python test_model_edgeTPU.py --source csi

파일 재생:
  python test_model_edgeTPU.py --source video.mp4
        """,
    )
    ap.add_argument("--lane-model", default="best_int8_edgetpu.tflite")
    ap.add_argument("--obs-model", default="obs_int8_edgetpu.tflite")
    ap.add_argument(
        "--source",
        default="csi",
        help="'csi'(기본) → Picamera2, 숫자 → /dev/videoN, 파일경로 → 영상 재생",
    )
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--coral", type=int, default=1, choices=[1, 2])
    ap.add_argument("--obs-skip", type=int, default=2)
    ap.add_argument("--no-edgetpu", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--disp-scale", type=float, default=1.0)
    ap.add_argument("--cam-w", type=int, default=320)  # CSI 기본 320
    ap.add_argument("--cam-h", type=int, default=320)  # CSI 기본 320
    args = ap.parse_args()

    # source 파싱: 숫자 → int(웹캠 인덱스), 'csi' or 파일경로 → 그대로 str
    if str(args.source).isdigit():
        src = int(args.source)
    else:
        src = args.source  # 'csi' 또는 파일 경로
    run(
        lane_model=args.lane_model,
        obs_model=args.obs_model,
        source=src,
        fps=args.fps,
        coral=args.coral,
        use_edgetpu=not args.no_edgetpu,
        obs_skip=args.obs_skip,
        save_video=args.save,
        disp_scale=args.disp_scale,
        cam_w=args.cam_w,
        cam_h=args.cam_h,
    )
