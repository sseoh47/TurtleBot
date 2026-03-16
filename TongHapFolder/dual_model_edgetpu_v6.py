"""
dual_model_edgetpu_v6.py
══════════════════════════════════════════════════════════════════════
Lane Detection (YOLOv8-det) + Obstacle Detection (YOLOv8-det)
Raspberry Pi 5 + Google Coral EdgeTPU USB x2

★ 수정 이력
  v5: - OBS_CLASS_NAMES = 학습한 6개 클래스로 교체
      - Coral 2개: device='usb:0'/'usb:1' 공식 형식으로 확정
      - 두 interpreter 사이 sleep(1.0) 으로 초기화 완료 대기
      - list_edge_tpus() 로 연결된 장치 수 사전 확인 + 경고
      - SaveThread 분리로 영상 저장 충돌 제거
      - DisplayThread writer 분리 (충돌 방지)
      - CONF_THRESH=0.45 / MASK_THRESH=0.65 / NMS_IOU=0.35 유지

  v6: - lane 모델 segmentation → detection 으로 변경
        * parse_lane: proto/mask/mcoef 제거, det 전용으로 단순화
        * _draw_frame: overlay 블렌딩 제거, lane bbox 직접 그리기
        * --no-mask 옵션 제거 (더 이상 불필요)
        * MASK_THRESH 상수 제거

사용법:
  # Coral 2개 (권장)
  python dual_model_edgetpu_v6.py \
      --lane-model best_int8_edgetpu.tflite \
      --obs-model  obs_int8_edgetpu.tflite \
      --coral 2 --obs-classes 6 --source 0

  # 저장 포함
  python dual_model_edgetpu_v6.py \
      --lane-model best_int8_edgetpu.tflite \
      --obs-model  obs_int8_edgetpu.tflite \
      --coral 2 --obs-classes 6 --save --source 0

  # 최고 FPS (해상도 낮추기)
  python dual_model_edgetpu_v6.py \
      --lane-model best_int8_edgetpu.tflite \
      --obs-model  obs_int8_edgetpu.tflite \
      --coral 2 --obs-classes 6 \
      --cam-w 320 --cam-h 240 --source 0
"""

##=======================================================
## AI파트 코드 발췌
##=======================================================

import csv
import cv2
import numpy as np
import argparse
import time
import threading
import queue
import os
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
CONF_THRESH = 0.35  # det 모델 기준으로 낮춤 (필요시 조정)
NMS_IOU = 0.35  # 겹치는 박스 적극 제거
LANE_DEBUG_ONCE = True  # 첫 프레임 디버그 출력 (확인 후 False로 변경)

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

# ★ 학습한 6개 클래스 (data.yaml names 순서와 동일)
OBS_CLASS_NAMES = [
    "KNU",  # class 0
    "SL",  # class 1
    "box",  # class 2
    "car",  # class 3
    "parking",  # class 4
    "person",  # class 5
]

INTER_DESC = {
    "straight": "Straight",
    "right_t": "T-Right",
    "left_t": "T-Left",
    "down_t": "T-Down",
    "cross": "Cross(+)",
    "curve_right": "Curve-R",
    "curve_left": "Curve-L",
    "approaching": "Approaching",
}


# ══════════════════════════════════════════════
# EdgeTPU 엔진
# ══════════════════════════════════════════════
class EdgeTPUEngine:
    """
    device 파라미터 (공식 형식 coral.ai/docs/edgetpu/multiple-edgetpu):
      'usb:0'  -> 첫 번째 USB Coral
      'usb:1'  -> 두 번째 USB Coral
      ''       -> 자동 선택 (1개일 때)
    """

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
        print(
            f"  [{tag}] {Path(model_path).name}  "
            f"({self.img_h}x{self.img_w})  {self.in_dtype.__name__}"
        )
        for i, od in enumerate(self.outd):
            print(f"    out[{i}] {od['shape']}  {od['dtype'].__name__}")

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(
            rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
        )
        if self.in_dtype == np.int8:
            f = small.astype(np.float32) * (1.0 / 255.0)
            q = (
                np.round(f / self.in_scale + self.in_zp)
                if self.in_scale > 0
                else f * 255.0 - 128.0
            )
            return np.expand_dims(np.clip(q, -128, 127).astype(np.int8), 0)
        elif self.in_dtype == np.uint8:
            if self.in_scale > 0:
                f = small.astype(np.float32) * (1.0 / 255.0)
                q = np.round(f / self.in_scale + self.in_zp)
                return np.expand_dims(np.clip(q, 0, 255).astype(np.uint8), 0)
            return np.expand_dims(small, 0)
        return np.expand_dims(small.astype(np.float32) * (1.0 / 255.0), 0)

    def dequant(self, t, idx):
        sc, zp = self.outd[idx]["quantization"]
        if self.outd[idx]["dtype"] in (np.uint8, np.int8) and sc > 0:
            return (t.astype(np.float32) - zp) * sc
        return t.astype(np.float32)

    def infer(self, bgr: np.ndarray):
        inp = self.preprocess(bgr)
        self.interp.set_tensor(self.ind[0]["index"], inp)
        t0 = time.perf_counter()
        self.interp.invoke()
        ms = (time.perf_counter() - t0) * 1000.0
        outs = [
            self.dequant(self.interp.get_tensor(od["index"]), i)
            for i, od in enumerate(self.outd)
        ]
        return outs, ms


# ══════════════════════════════════════════════
# Lane 후처리
# ══════════════════════════════════════════════
def parse_lane(outputs, H, W):
    """
    YOLOv8-det (detection 전용) 출력 파싱.
    output[0]: [1, 4+nc, num_anchors] 또는 [1, num_anchors, 4+nc]
    반환: shapes 리스트 (overlay 없음)
    """
    global LANE_DEBUG_ONCE
    shapes = []
    if not outputs:
        return shapes

    det = outputs[0][0]

    # ── 디버그: 첫 프레임에서 텐서 shape·score 분포 출력 ──
    if LANE_DEBUG_ONCE:
        print(f"[LANE DEBUG] raw output shape: {outputs[0].shape}")
        print(f"[LANE DEBUG] det shape before T: {det.shape}")
        if det.shape[0] > det.shape[1]:
            _d = det.T
        else:
            _d = det
        _nc = _d.shape[0] - 4
        if _nc > 0:
            _scores = _d[4:, :].T
            _confs = np.max(_scores, axis=1)
            print(f"[LANE DEBUG] nc={_nc}  anchors={_d.shape[1]}")
            print(
                f"[LANE DEBUG] score max={_confs.max():.4f}  "
                f"mean={_confs.mean():.4f}  "
                f">0.35: {(_confs>0.35).sum()}  "
                f">0.50: {(_confs>0.50).sum()}  "
                f">0.70: {(_confs>0.70).sum()}"
            )
        LANE_DEBUG_ONCE = False
    # ────────────────────────────────────────────────────────

    if det.shape[0] > det.shape[1]:
        det = det.T  # → [num_anchors, 4+nc]
    nc = det.shape[0] - 4
    if nc <= 0:
        return shapes

    boxes = det[:4, :].T  # [num_anchors, 4]  cx,cy,w,h (정규화)
    scores = det[4:, :].T  # [num_anchors, nc]

    cids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)
    keep = confs > CONF_THRESH
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
        CONF_THRESH,
        NMS_IOU,
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
        shapes.append(
            {
                "class_id": cid,
                "class_name": LANE_NAMES.get(cid, "?"),
                "cx_norm": float(((bx1 + bx2) / 2) / W),
                "cy_norm": float(((by1 + by2) / 2) / H),
                "bbox": (bx1, by1, bx2, by2),
                "conf": float(confs[i]),
            }
        )
    return shapes


# ══════════════════════════════════════════════
# Obstacle 후처리
# ══════════════════════════════════════════════
def parse_obstacle(outputs, H, W):
    dets = []
    if not outputs:
        return dets
    det = outputs[0][0]
    if det.shape[0] > det.shape[1]:
        det = det.T
    if det.shape[0] - 4 <= 0:
        return dets
    boxes = det[:4, :].T
    scores = det[4:, :].T
    cids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)
    keep = confs > CONF_THRESH
    if not np.any(keep):
        return dets
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
        CONF_THRESH,
        NMS_IOU,
    )
    if len(idx) == 0:
        return dets
    for ii in idx:
        i = int(ii)
        cid = int(cids[i])
        name = OBS_CLASS_NAMES[cid] if cid < len(OBS_CLASS_NAMES) else str(cid)
        bx1 = int(np.clip(x1[i], 0, W - 1))
        by1 = int(np.clip(y1[i], 0, H - 1))
        bx2 = int(np.clip(x2[i], 0, W - 1))
        by2 = int(np.clip(y2[i], 0, H - 1))
        area = (bx2 - bx1) * (by2 - by1)
        dets.append(
            {
                "class_id": cid,
                "class_name": name,
                "bbox": (bx1, by1, bx2, by2),
                "conf": float(confs[i]),
                "area": area,
            }
        )

    # 우선순위: bbox 넓이 큰 것 → confidence 큰 것, 최종 1개만 반환
    dets.sort(key=lambda d: (d["area"], d["conf"]), reverse=True)
    return dets[:1]


# ══════════════════════════════════════════════
# Lane 로직
# ══════════════════════════════════════════════
def classify_intersection(shapes, history):
    lines = [s for s in shapes if s["class_name"] == "line"]
    curves = [s for s in shapes if s["class_name"] == "curve"]
    eeus = [s for s in shapes if s["class_name"] == "eeu"]
    has_eeu, has_curve, has_line = bool(eeus), bool(curves), bool(lines)
    if has_eeu:
        if has_curve:
            cx = float(np.mean([s["cx_norm"] for s in curves]))
            return "curve_right" if cx >= 0.5 else "curve_left"
        rec = history[-11:]
        pr = sum(
            1
            for f in rec
            if any(s["class_name"] == "curve" and s["cx_norm"] >= 0.5 for s in f)
        )
        pl = sum(
            1
            for f in rec
            if any(s["class_name"] == "curve" and s["cx_norm"] < 0.5 for s in f)
        )
        if pr > pl and pr >= 2:
            return "curve_right"
        if pl > pr and pl >= 2:
            return "curve_left"
        return "down_t"
    if has_curve:
        lc = [s for s in curves if s["cx_norm"] < 0.5]
        rc = [s for s in curves if s["cx_norm"] >= 0.5]
        ll = [s for s in lines if s["cx_norm"] < 0.5]
        rl = [s for s in lines if s["cx_norm"] >= 0.5]
        if lc and rc:
            return "cross"
        if rc and ll:
            return "right_t"
        if lc and rl:
            return "left_t"
        return "approaching"
    if has_line:
        return "straight"
    return None


def compute_lane_error(shapes):
    mid = 0.5
    ll = [s for s in shapes if s["class_name"] == "line" and s["cx_norm"] < 0.5]
    rl = [s for s in shapes if s["class_name"] == "line" and s["cx_norm"] >= 0.5]
    if ll and rl:
        return (
            float(
                np.clip(
                    (
                        (
                            np.mean([s["cx_norm"] for s in ll])
                            + np.mean([s["cx_norm"] for s in rl])
                        )
                        / 2
                        - mid
                    )
                    / mid,
                    -1,
                    1,
                )
            ),
            "ok",
        )
    if ll:
        return (
            float(np.clip((np.mean([s["cx_norm"] for s in ll]) - mid) / mid, -1, 1)),
            "left_only",
        )
    if rl:
        return (
            float(np.clip((np.mean([s["cx_norm"] for s in rl]) - mid) / mid, -1, 1)),
            "right_only",
        )
    return 0.0, "lost"


class IntersectionFSM:
    def __init__(
        self, confirm=2, hold=10, maxhist=33
    ):  # confirm 2: 2연속으로 교차로 판단
        self.confirm = confirm
        self.hold = hold
        self.consec = 0
        self.hold_until = 0.0
        self.cur = None
        self.hist = []
        self.maxhist = maxhist

    def update(self, shapes):
        self.hist.append(shapes)
        if len(self.hist) > self.maxhist:
            self.hist.pop(0)
        raw = classify_intersection(shapes, self.hist)
        return self._fsm(raw), raw

    def _fsm(self, raw):
        now = time.time()
        if now < self.hold_until:
            return True, self.cur
        if raw in ("curve_right", "curve_left"):
            self.cur = raw
            self.hold_until = now + self.hold
            self.consec = 0
            return True, self.cur
        if raw and raw not in ("approaching", "straight", None):
            self.consec = self.consec + 1 if raw == self.cur else 1
            self.cur = raw
            if self.consec >= self.confirm:
                self.hold_until = now + self.hold
                self.consec = 0
                return True, self.cur
        else:
            self.consec = 0
        return False, None


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════
def _draw_frame(d: dict) -> np.ndarray:
    frame = d["frame"]
    shapes = d["shapes"]
    obs_list = d["obs"]
    le = d["le"]
    ls = d["ls"]
    is_inter = d["is_inter"]
    itype = d["itype"]
    raw = d["raw"]
    lane_ms = d["lane_ms"]
    obs_ms = d["obs_ms"]
    fps_val = d["fps_val"]
    mode_str = d["mode_str"]
    scale = d.get("scale", 1.0)
    progress = d.get("progress", None)  # (current_frame, total_frames) or None

    vis = frame.copy()
    h, w = vis.shape[:2]

    # lane: detection bbox + 클래스명 표시
    for s in shapes:
        x1, y1, x2, y2 = s["bbox"]
        col = LANE_COLOR.get(s["class_id"], (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            vis,
            f"{s['class_name']} {s['conf']:.2f}",
            (x1, max(y1 - 5, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            col,
            1,
        )

    for o in obs_list:
        x1, y1, x2, y2 = o["bbox"]
        col = OBS_PALETTE[o["class_id"] % len(OBS_PALETTE)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            vis,
            f"{o['class_name']} {o['conf']:.2f}",
            (x1, max(y1 - 5, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            col,
            1,
        )

    by = h - 20
    bcx = w // 2
    bx = int(bcx + le * bcx)
    cv2.line(vis, (0, by), (w, by), (50, 50, 50), 1)
    cv2.line(vis, (bcx, by), (bx, by), (0, 255, 255), 5)
    cv2.circle(vis, (bx, by), 7, (0, 255, 255), -1)
    cv2.line(vis, (bcx, by - 8), (bcx, by + 8), (255, 255, 255), 1)

    cv2.putText(
        vis,
        f"err:{le:+.3f} [{ls}]",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"raw:{raw or 'none'}",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (160, 160, 160),
        1,
    )
    if is_inter:
        cv2.putText(
            vis,
            f">> {INTER_DESC.get(itype, itype or '')}",
            (8, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
        )
    # FPS/ms 화면 표시 제거 (터미널 출력으로 대체)

    # 영상 진행률 바 (파일 소스일 때만)
    if progress is not None:
        cur, total = progress
        pct = cur / total if total > 0 else 0
        bar_w = int(w * pct)
        cv2.rectangle(vis, (0, h - 6), (w, h), (50, 50, 50), -1)  # 배경
        cv2.rectangle(vis, (0, h - 6), (bar_w, h), (0, 200, 100), -1)  # 진행
        pct_txt = f"{pct*100:.1f}%  {cur}/{total}f"
        cv2.putText(
            vis,
            pct_txt,
            (w - 160, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
        )

    if obs_list:
        cv2.putText(
            vis,
            f"OBS:{len(obs_list)}",
            (w - 80, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )

    if scale != 1.0:
        vis = cv2.resize(
            vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST
        )
    return vis


# 객체 클래스명 -> 출력 ID 맵핑 (0~10 체계)
# KNU, 는 10으로 출력 box는 아직 미정
OBS_OUTPUT_ID = {
    "KNU": 10,  #
    "SL": 2,  # 2번
    "person": 3,  # 3번
    "car": 4,  # 4번
    "parking": 5,  # 5번
}


def make_output(le, ls, is_inter, itype, obs_list):
    """
    ID 체계: 0~9
      0 = 차선 없음
      1 = 차선 인식
      2 = SL 감지
      3 = person 감지
      4 = car 감지
      5 = parking 감지
      6 = ㅓ자 교차로
      7 = ㅏ자 교차로
      8 = ㅜ자 교차로
      9 = +자 교차로

    반환: 패킷 리스트
      예) [[1, 0.123], [4, None], [3, None]]
    """
    err = round(le, 4)
    result = []

    if is_inter:
        inter_id = {"left_t": 6, "right_t": 7, "down_t": 8, "cross": 9}.get(itype)
        if inter_id is not None:
            result.append([inter_id, err])
        else:
            result.append([1, err])
    elif ls == "lost":
        result.append([0, None])
    else:
        result.append([1, err])

    # 객체 패킷 (KNU, box 제외)
    for o in obs_list:
        oid = OBS_OUTPUT_ID.get(o["class_name"])
        if oid is not None:
            result.append([oid, None])

    return result


# ══════════════════════════════════════════════
# 스레드 클래스들
# ══════════════════════════════════════════════
class FrameGrabber(threading.Thread):
    """카메라 캡처 전담 - 항상 최신 프레임 유지. 영상 끝나면 eof=True"""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self._f = None
        self._lk = threading.Lock()
        self.alive = True
        self.eof = False

    def run(self):
        while self.alive:
            ok, f = self.cap.read()
            if ok:
                with self._lk:
                    self._f = f
            else:
                self.eof = True  # 영상 끝 또는 카메라 끊김
                break

    def get(self):
        with self._lk:
            return None if self._f is None else self._f.copy()

    def stop(self):
        self.alive = False


class InferWorker(threading.Thread):
    """Coral 2개 모드 전용 추론 워커"""

    def __init__(self, engine, name):
        super().__init__(daemon=True, name=name)
        self.engine = engine
        self.alive = True
        self._in = None
        self._in_lk = threading.Lock()
        self._out = None
        self._out_lk = threading.Lock()
        self._ev = threading.Event()

    def push(self, frame):
        with self._in_lk:
            self._in = frame
        self._ev.set()

    def result(self):
        with self._out_lk:
            return self._out

    def run(self):
        while self.alive:
            self._ev.wait(1.0)
            self._ev.clear()
            with self._in_lk:
                f = self._in
            if f is None:
                continue
            r = self.engine.infer(f)
            with self._out_lk:
                self._out = r

    def stop(self):
        self.alive = False


class DisplayThread(threading.Thread):
    """draw + imshow 전담 - 메인 추론루프 블로킹 제거"""

    def __init__(self, win_name):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=1)
        self.win_name = win_name
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
            try:
                cv2.imshow(self.win_name, _draw_frame(d))
                cv2.waitKey(1)
            except Exception:
                pass

    def stop(self):
        self.alive = False


class SaveThread(threading.Thread):
    """영상 저장 전담 - DisplayThread와 분리로 writer 충돌 방지"""

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
        # 큐에 남은 프레임 모두 저장
        while not self.q.empty():
            try:
                self.writer.write(_draw_frame(self.q.get_nowait()))
            except:
                break
        self.writer.release()
        print("[INFO] 영상 저장 완료")


class CsvLogger:
    """
    print 로그를 CSV 파일로 저장하는 클래스
    컬럼: timestamp, frame_idx, lane_error, lane_status, inter_type,
          lane_ms, obs_ms, fps, obs_count, obs_classes, output_packets
    """

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

        # lane: 감지된 각 shape의 클래스명 + confidence
        lane_classes = "|".join(s["class_name"] for s in shapes)
        lane_confs = "|".join(f"{s['conf']:.3f}" for s in shapes)

        # obs: 1개만 출력되므로 단일 값
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


def _check_display():
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        cv2.imshow("_t", np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("_t")
        return True
    except:
        return False


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
    no_display,
    disp_scale,
    orig_h,
    orig_w,
    source,
    total_frames=0,
    save_fps=30,
):

    is_file = Path(str(source)).exists()
    has_disp = not no_display and _check_display()
    mode_str = f"{coral}-Coral"
    win_name = f"DualTPU [{mode_str}]"

    # ── CSV 로거 ──────────────────────────────
    src_stem = Path(str(source)).stem if Path(str(source)).exists() else "cam"
    csv_path = f"log_{src_stem}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_logger = CsvLogger(csv_path)

    # ── VideoWriter + SaveThread ──────────────
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

    # ── DisplayThread ─────────────────────────
    disp_th = None
    if has_disp:
        disp_th = DisplayThread(win_name)
        disp_th.start()
        print("[INFO] DisplayThread 시작 (시각화 분리)")

    # ── Coral 2개 워커 ────────────────────────
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
    frame_idx = 0  # 전체 프레임 인덱스 (progress 표시용)
    obs_list = []
    obs_ms = 0.0
    lane_ms = 0.0
    prev_sh = []
    p_le = 0.0
    p_ls = "lost"
    p_is = False
    p_it = None
    p_raw = None
    quit_flag = False

    try:
        while not quit_flag:
            # 영상 끝나면 루프 종료
            if grabber.eof:
                print("[INFO] 영상 재생 완료. 종료합니다.")
                break
            t0 = time.time()
            frame = grabber.get()
            if frame is None:
                time.sleep(0.003)
                continue

            H, W = frame.shape[:2]
            fc += 1
            frame_idx += 1

            if coral == 2:
                lane_w.push(frame)
                obs_w.push(frame)
                lr = lane_w.result()
                or_ = obs_w.result()
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

            if has_disp:
                disp_th.push(draw_data)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    quit_flag = True

            if save_th:
                save_th.push(draw_data)

            sl = interval - (time.time() - t0)
            if sl > 0:
                time.sleep(sl)

            out = make_output(p_le, p_ls, p_is, p_it, obs_list)

            # 진행률 (파일 소스일 때만)
            if total_frames > 0:
                pct = frame_idx / total_frames * 100
                remain = total_frames - frame_idx
                remain_sec = remain / fps if fps > 0 else 0
                progress_str = (
                    f"[{pct:5.1f}%] {frame_idx}/{total_frames}f "
                    f"남은프레임:{remain} "
                    f"({remain_sec:.1f}s)"
                )
            else:
                progress_str = ""

            print(
                f"[{mode_str}] "
                f"lane:{p_le:+.4f}|{p_ls}|{p_it}|"
                f"lane={lane_ms:.0f}ms obs={obs_ms:.0f}ms "
                f"FPS={fps_val:.1f}  {progress_str}"
            )
            print(f">>> {' '.join(str(pkt) for pkt in out)}")

            # ── CSV 로그 저장 ──────────────────
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
        if disp_th:
            disp_th.stop()
            cv2.destroyAllWindows()
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
    no_display,
    disp_scale,
    cam_w,
    cam_h,
):

    print(f"\n=== Dual-Model EdgeTPU v6 (lane: det) ===")

    # ── Coral 연결 상태 사전 확인 ─────────────
    if use_edgetpu and EDGETPU_AVAILABLE:
        tpus = list_edge_tpus()
        print(f"[INFO] 연결된 EdgeTPU 장치: {len(tpus)}개")
        for i, t in enumerate(tpus):
            print(f"  [{i}] type={t.get('type','?')}  path={t.get('path','?')}")
        if coral == 2 and len(tpus) < 2:
            print(f"[WARN] --coral 2 지정했지만 장치가 {len(tpus)}개 감지됨!")
            print("       USB 전력 또는 연결 상태를 확인하세요.")

    # ── 모델 로드 ─────────────────────────────
    if coral == 2 and use_edgetpu:
        # ★ 공식 형식 usb:0 / usb:1 (pycoral 2.0.0 검증 완료)
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

    is_file = Path(str(source)).exists()
    cap = cv2.VideoCapture(source if not is_file else str(source))
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 파일 소스일 때 원본 fps 사용, 웹캠은 --fps 인자 사용
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    save_fps = src_fps if is_file and src_fps > 0 else fps
    print(
        f"[INFO] 해상도: {orig_w}x{orig_h}  obs_skip={obs_skip}  disp_scale={disp_scale}"
    )
    print(f"[INFO] 저장 FPS: {save_fps:.2f}  (원본: {src_fps:.2f})\n")

    # 파일 소스일 때 총 프레임 수 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0

    grabber = FrameGrabber(cap)
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
            no_display,
            disp_scale,
            orig_h,
            orig_w,
            source,
            total_frames,
            save_fps,
        )
    finally:
        grabber.stop()
        cap.release()
        print("[INFO] 종료")


###===============================================================
## 팀장코드~ 추가내용 시작
###===============================================================
from dataclasses import dataclass
from typing import Optional


@dataclass
class DualInferenceResult:
    line_id: int
    angle: Optional[float]
    obj_id: Optional[int]
    lane_status: str
    inter_type: Optional[str]


def convert_lane_result(p_le, p_ls, p_is, p_it):
    """
    파일 내부 결과를 최종 lane class 체계로 변환
      0 = 차선 없음
      1 = 일반 차선
      6 = left_t
      7 = right_t
      8 = down_t
      9 = cross
    """
    if p_is:
        inter_map = {
            "left_t": 6,
            "right_t": 7,
            "down_t": 8,
            "cross": 9,
        }
        if p_it in inter_map:
            return inter_map[p_it], None

    if p_ls == "lost":
        return 0, None

    return 1, float(p_le)


def convert_object_result(obs_list):
    """
    파일 객체 결과를 최종 obj class 체계로 변환
      SL      -> 2
      person  -> 3
      car     -> 4
      parking -> 5
      없음    -> None
    """
    if not obs_list:
        return None

    name = obs_list[0]["class_name"]
    obj_map = {
        "SL": 2,
        "person": 3,
        "car": 4,
        "parking": 5,
    }
    return obj_map.get(name)


class DualModelRunner:
    """
    dual_model_edgetpu_v6.py를 외부 main.py에서 import해서 쓰기 위한 래퍼
    """

    def __init__(
        self,
        lane_model,
        obs_model,
        source=0,
        coral=2,
        use_edgetpu=True,
        cam_w=640,
        cam_h=480,
    ):
        self.coral = coral

        if coral == 2 and use_edgetpu:
            print("[INFO] lane 모델 -> usb:0 로드")
            self.lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=True, device="usb:0")
            time.sleep(1.0)
            print("[INFO] obs 모델 -> usb:1 로드")
            self.obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=True, device="usb:1")
        else:
            self.lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=use_edgetpu)
            self.obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=use_edgetpu)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"source open failed: {source}")

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.fsm = IntersectionFSM()

    def step(self) -> Optional[DualInferenceResult]:
        ok, frame = self.cap.read()
        if not ok:
            return None

        H, W = frame.shape[:2]

        lane_outs, _ = self.lane_eng.infer(frame)
        lane_shapes = parse_lane(lane_outs, H, W)
        p_le, p_ls = compute_lane_error(lane_shapes)
        (p_is, p_it), _ = self.fsm.update(lane_shapes)

        obs_outs, _ = self.obs_eng.infer(frame)
        obs_list = parse_obstacle(obs_outs, H, W)

        line_id, angle = convert_lane_result(p_le, p_ls, p_is, p_it)
        obj_id = convert_object_result(obs_list)

        return DualInferenceResult(
            line_id=line_id,
            angle=angle,
            obj_id=obj_id,
            lane_status=p_ls,
            inter_type=p_it if p_is else None,
        )

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


###===============================================================
## 팀장코드~ 종료지점~
###===============================================================

# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
FPS 높이는 조합 (추천):
  --coral 2                       Coral 2개 병렬         ★★★
  --cam-w 320 --cam-h 240         해상도 낮추기           ★★★
  --obs-skip 2  (1-Coral만)       obs 추론 빈도 줄이기   ★★☆
  --no-display                    시각화 완전 제거        ★★☆
  --disp-scale 0.5                화면 렌더링 절반        ★☆☆
        """,
    )
    ap.add_argument("--lane-model", default="best_int8_edgetpu.tflite")
    ap.add_argument("--obs-model", default="obs_int8_edgetpu.tflite")
    ap.add_argument("--source", default="0")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--coral", type=int, default=1, choices=[1, 2])
    ap.add_argument(
        "--obs-skip", type=int, default=2, help="[1-Coral] obs를 N프레임마다 1회 실행"
    )
    ap.add_argument("--no-edgetpu", action="store_true")
    ap.add_argument(
        "--save", action="store_true", help="결과 영상 저장 (result_cam.mp4)"
    )
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--disp-scale", type=float, default=1.0)
    ap.add_argument("--cam-w", type=int, default=640)
    ap.add_argument("--cam-h", type=int, default=480)
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run(
        lane_model=args.lane_model,
        obs_model=args.obs_model,
        source=src,
        fps=args.fps,
        coral=args.coral,
        use_edgetpu=not args.no_edgetpu,
        obs_skip=args.obs_skip,
        save_video=args.save,
        no_display=args.no_display,
        disp_scale=args.disp_scale,
        cam_w=args.cam_w,
        cam_h=args.cam_h,
    )
