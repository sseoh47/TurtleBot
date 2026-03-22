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
LANE_ROI_TOP = 0.30


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
    shapes = []
    if len(outputs) < 2:
        return shapes

    proto_raw = None
    det_raw = None
    for o in outputs:
        if o.ndim == 4:
            proto_raw = o[0]
        elif o.ndim == 3:
            det_raw = o[0]

    if proto_raw is None or det_raw is None:
        return shapes

    det = det_raw if det_raw.shape[0] < det_raw.shape[1] else det_raw.T
    nm = proto_raw.shape[2]
    nc = det.shape[0] - 4 - nm
    if nc <= 0:
        return shapes

    boxes = det[:4, :].T
    scores = det[4 : 4 + nc, :].T
    mask_coefs = det[4 + nc :, :].T

    cids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)
    # parse_lane 함수에서 line 228 바로 위에 추가
    # print(f"[LANE_DEBUG] max_conf={confs.max():.3f}, mean_conf={confs.mean():.3f}, above_thresh={np.sum(confs > LANE_CONF_THRESH)}")
    keep = confs > LANE_CONF_THRESH
    if not np.any(keep):
        return shapes

    boxes = boxes[keep]
    confs = confs[keep]
    cids = cids[keep]
    mask_coefs = mask_coefs[keep]

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

        # ROI: bbox 중심이 상단 LANE_ROI_TOP 영역 안에 있으면 제거
        cy_norm = float(((by1 + by2) / 2) / H)
        if cy_norm < LANE_ROI_TOP:
            continue

        mask_bin = None
        if DRAW_LANE_MASK:
            raw_mask = proto_raw @ mask_coefs[i]
            raw_mask = 1.0 / (1.0 + np.exp(-raw_mask))
            mask_full = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_LINEAR)
            mask_bin = np.zeros((H, W), dtype=bool)
            mask_bin[by1:by2, bx1:bx2] = mask_full[by1:by2, bx1:bx2] > 0.5

        shapes.append(
            {
                "class_id": cid,
                "class_name": LANE_NAMES.get(cid, "?"),
                "cx_norm": float(((bx1 + bx2) / 2) / W),
                "cy_norm": cy_norm,
                "bbox": (bx1, by1, bx2, by2),
                "conf": float(confs[i]),
                "mask": mask_bin,
            }
        )
    return _merge_lane_shapes(shapes, H, W)


def _merge_lane_shapes(shapes, H, W, cx_gap=0.15):
    """
    같은 클래스끼리 cx_norm 차이가 cx_gap 이내인 박스들을 하나로 합침.
    위아래로 쪼개진 박스를 하나의 대표 박스로 통합.
    cx_gap: 같은 차선으로 볼 cx_norm 최대 거리 (기본 0.15 = 화면 너비의 15%)
    """
    if not shapes:
        return shapes

    # 클래스별로 분리
    merged = []
    by_class = {}
    for s in shapes:
        by_class.setdefault(s["class_name"], []).append(s)

    for cname, group in by_class.items():
        # cx_norm 기준 정렬 후 그룹핑
        group = sorted(group, key=lambda s: s["cx_norm"])
        clusters = []
        cur = [group[0]]
        for s in group[1:]:
            if s["cx_norm"] - cur[-1]["cx_norm"] <= cx_gap:
                cur.append(s)
            else:
                clusters.append(cur)
                cur = [s]
        clusters.append(cur)

        for cluster in clusters:
            # bbox는 전체를 감싸는 union box
            bx1 = min(s["bbox"][0] for s in cluster)
            by1 = min(s["bbox"][1] for s in cluster)
            bx2 = max(s["bbox"][2] for s in cluster)
            by2 = max(s["bbox"][3] for s in cluster)
            best = max(cluster, key=lambda s: s["conf"])
            merged.append(
                {
                    "class_id": best["class_id"],
                    "class_name": cname,
                    "cx_norm": float((bx1 + bx2) / 2 / W),
                    "cy_norm": float((by1 + by2) / 2 / H),
                    "bbox": (bx1, by1, bx2, by2),
                    "conf": best["conf"],
                    "mask": best["mask"],
                }
            )

    return merged


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
    keep = confs > OBS_CONF_THRESH
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
        OBS_CONF_THRESH,
        OBS_NMS_IOU,
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

    dets.sort(key=lambda d: (d["area"], d["conf"]), reverse=True)
    return dets[:1]


# ══════════════════════════════════════════════
# Lane 로직
# ══════════════════════════════════════════════
def compute_lane_error(shapes):
    mid = 0.5
    lines = [s for s in shapes if s["class_name"] == "line"]

    if not lines:
        return 0.0, "lost"

    # line 박스가 위아래로 쪼개져서 여러 개 나올 수 있음
    # → cx_norm < 0.5 인 것들의 평균 = 왼쪽 차선 cx
    # → cx_norm >= 0.5 인 것들의 평균 = 오른쪽 차선 cx
    ll = [s["cx_norm"] for s in lines if s["cx_norm"] < mid]
    rl = [s["cx_norm"] for s in lines if s["cx_norm"] >= mid]

    if ll and rl:
        center = (np.mean(ll) + np.mean(rl)) / 2
        err = float(np.clip((center - mid) / mid * 10, -10, 10))
        return err, "ok"

    if ll:
        err = float(np.clip((np.mean(ll) - mid) / mid * 10, -10, 10))
        return err, "left_only"

    if rl:
        err = float(np.clip((np.mean(rl) - mid) / mid * 10, -10, 10))
        return err, "right_only"

    return 0.0, "lost"


# cy_norm 기준: 0.0=상단, 1.0=하단
# "상단"  : cy_norm < 0.45  (ROI 제거 후 기준)
# "하단"  : cy_norm >= 0.55
# "오른쪽": cx_norm >= 0.5
# "왼쪽"  : cx_norm < 0.5

_CY_TOP = 0.45  # 이 값보다 작으면 "상단" 탐지
_CY_BOTTOM = 0.55  # 이 값 이상이면 "하단" 탐지
_CX_RIGHT = 0.5  # 이 값 이상이면 "오른쪽"


def _raw_classify(shapes):
    """
    현재 프레임 shapes만 보고 raw 유형 반환.
    반환값:
      'no_lane' | 'straight' | 'curve_left' | 'left_t'
      | 'cross_or_down_t'  <- curve 가 좌하단+우하단 동시에 존재
      | 'pending_eeu'       <- eeu만 단독
      | None
    """
    lines = [s for s in shapes if s["class_name"] == "line"]
    curves = [s for s in shapes if s["class_name"] == "curve"]
    eeus = [s for s in shapes if s["class_name"] == "eeu"]

    if not lines and not curves and not eeus:
        return "no_lane"

    # ── straight: line만 존재
    if lines and not curves and not eeus:
        return "straight"

    # ── curve_left: eeu(상단) + line(오른쪽)
    if eeus and lines and not curves:
        eeu_top = any(s["cy_norm"] < _CY_TOP for s in eeus)
        line_right = any(s["cx_norm"] >= _CX_RIGHT for s in lines)
        if eeu_top and line_right:
            return "curve_left"

    # ── left_t: curve가 왼쪽에만 있고 + line이 오른쪽에 있음
    #    curve가 오른쪽에 하나도 없어야 함 (cross_or_down_t와 구분)
    if curves and lines and not eeus:
        has_curve_left = any(s["cx_norm"] < _CX_RIGHT for s in curves)
        has_curve_right = any(s["cx_norm"] >= _CX_RIGHT for s in curves)
        has_line_right = any(s["cx_norm"] >= _CX_RIGHT for s in lines)
        # curve가 오른쪽엔 없고 왼쪽에만 있어야 left_t
        if has_curve_left and not has_curve_right and has_line_right:
            return "left_t"

    # ── cross_or_down_t: curve가 반드시 좌측 + 우측 동시에 존재해야 함
    #    (line 유무 무관)
    if curves:
        has_curve_left = any(s["cx_norm"] < _CX_RIGHT for s in curves)
        has_curve_right = any(s["cx_norm"] >= _CX_RIGHT for s in curves)
        if has_curve_left and has_curve_right:
            return "cross_or_down_t"

    # ── eeu만 단독 등장
    if eeus and not curves and not lines:
        return "pending_eeu"

    return None


def _resolve_cross_down(shapes):
    """
    cross_or_down_t 이후 단일 프레임으로 판별.
    - eeu 등장(위치 무관) → down_t  (클래스 불균형으로 cy 낮게 나와도 처리)
    - curve 2개(좌 상단 + 우 상단) → cross
    - 그 외 → None (계속 대기)
    """
    eeus = [s for s in shapes if s["class_name"] == "eeu"]
    curves = [s for s in shapes if s["class_name"] == "curve"]

    if eeus:
        return "down_t"

    curve_left_top = any(
        s["cx_norm"] < _CX_RIGHT and s["cy_norm"] < _CY_TOP for s in curves
    )
    curve_right_top = any(
        s["cx_norm"] >= _CX_RIGHT and s["cy_norm"] < _CY_TOP for s in curves
    )
    if curve_left_top and curve_right_top:
        return "cross"

    return None


class IntersectionFSM:
    """
    cross_or_down_t 진입 후엔 [1, err](직진) 계속 내보내면서
    eeu(상단) 또는 curve 2개(상단 좌+우) 프레임이 올 때까지 무한 대기.
    나머지 유형은 confirm 프레임 연속 확인 후 hold 동안 유지.
    """

    def __init__(self, confirm=2, hold=3, maxhist=30):
        self.confirm = confirm
        self.hold = hold
        self.maxhist = maxhist
        self.consec = 0
        self.hold_until = 0.0
        self.cur = None
        self.hist = []
        self._pending = False  # cross_or_down_t 대기 중

    def update(self, shapes):
        self.hist.append(shapes)
        if len(self.hist) > self.maxhist:
            self.hist.pop(0)

        raw = _raw_classify(shapes)

        # ── cross_or_down_t 대기 중: 결정적 프레임만 받으면 확정
        if self._pending:
            resolved = _resolve_cross_down(shapes)
            if resolved:
                self._pending = False
                self.cur = resolved
                self.hold_until = time.time() + self.hold
                return (True, self.cur), raw
            # 아직 판별 불가 → 직진 신호([1, err]) 계속 출력
            return (False, None), raw

        # ── cross_or_down_t 또는 pending_eeu 진입
        if raw in ("cross_or_down_t", "pending_eeu"):
            self._pending = True
            self.consec = 0
            return (False, None), raw

        return self._fsm(raw), raw

    def _fsm(self, raw):
        now = time.time()

        # hold 중이라도 다른 확실한 패턴이 오면 즉시 갱신
        if now < self.hold_until:
            if raw and raw not in ("no_lane", "straight", None) and raw != self.cur:
                # 새 패턴으로 교체 — hold 리셋
                self.hold_until = 0.0
                self.consec = 1
                self.cur = raw
            else:
                return True, self.cur

        if raw in ("no_lane", "straight", None):
            self.consec = 0
            return False, None

        self.consec = self.consec + 1 if raw == self.cur else 1
        self.cur = raw
        if self.consec >= self.confirm:
            self.hold_until = now + self.hold
            self.consec = 0
            return True, self.cur

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
    scale = d.get("scale", 1.0)
    progress = d.get("progress", None)

    vis = frame.copy()
    h, w = vis.shape[:2]

    if DRAW_LANE_MASK:
        overlay = vis.copy()
        for s in shapes:
            col = LANE_COLOR.get(s["class_id"], (200, 200, 200))
            mask = s.get("mask")
            if mask is not None and mask.shape == (h, w):
                overlay[mask] = (
                    overlay[mask].astype(np.float32) * 0.45
                    + np.array(col, dtype=np.float32) * 0.55
                ).astype(np.uint8)

            x1, y1, x2, y2 = s["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                overlay,
                f"{s['class_name']} {s['conf']:.2f}",
                (x1, max(y1 - 5, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                col,
                1,
            )
        vis = overlay
    else:
        for s in shapes:
            col = LANE_COLOR.get(s["class_id"], (200, 200, 200))
            x1, y1, x2, y2 = s["bbox"]
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

    if progress is not None:
        cur, total = progress
        pct = cur / total if total > 0 else 0
        bar_w = int(w * pct)
        cv2.rectangle(vis, (0, h - 6), (w, h), (50, 50, 50), -1)
        cv2.rectangle(vis, (0, h - 6), (bar_w, h), (0, 200, 100), -1)
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


OBS_OUTPUT_ID = {
    "SL": 2,
    "person": 3,
    "car": 4,
    "parking": 5,
}


def make_output(le, ls, is_inter, itype, obs_list):
    err = round(le, 2)
    result = []

    if is_inter:
        inter_id = {
            "left_t": 6,
            "curve_left": 6,
            "down_t": 8,
            "cross": 7,
        }.get(itype)
        if inter_id is not None:
            result.append([inter_id, err])
        else:
            result.append([1, err])
    elif ls == "lost":
        result.append([0, None])
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
