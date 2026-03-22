import time
from pathlib import Path

import cv2
import numpy as np

try:
    from pycoral.utils.edgetpu import make_interpreter

    EDGETPU_AVAILABLE = True
except ImportError:
    EDGETPU_AVAILABLE = False

try:
    import tflite_runtime.interpreter as tflite

    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite

        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

LANE_CONF_THRESH = 0.75
LANE_NMS_IOU = 0.35
OBS_CONF_THRESH = 0.60
OBS_NMS_IOU = 0.35
LANE_ROI_TOP = 0.10
LANE_DEBUG_ONCE = True

LANE_NAMES = {0: "curve", 1: "eeu", 2: "line"}
OBS_CLASS_NAMES = ["KNU", "SL", "box", "car", "parking", "person"]

_LANE_MID_X = 0.5


class EdgeTPUEngine:
    """
    runner.py 에서 import 해서 쓰는 최소 엔진.
    device='usb:0' / 'usb:1' 지원.
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

        if self.in_dtype == np.int8:
            if self.in_scale > 0:
                lut = (
                    np.arange(256, dtype=np.float32) / 255.0 / self.in_scale
                    + self.in_zp
                )
                self._lut = np.clip(lut, -128, 127).astype(np.int8)
            else:
                lut = np.arange(256, dtype=np.float32) - 128.0
                self._lut = np.clip(lut, -128, 127).astype(np.int8)
        elif self.in_dtype == np.uint8:
            if self.in_scale > 0:
                lut = (
                    np.arange(256, dtype=np.float32) / 255.0 / self.in_scale
                    + self.in_zp
                )
                self._lut = np.clip(lut, 0, 255).astype(np.uint8)
            else:
                self._lut = None
        else:
            self._lut = None

        self.pre_ms = 0.0
        self.invoke_ms = 0.0

        print(
            f"  [{tag}] {Path(model_path).name} ({self.img_h}x{self.img_w}) {self.in_dtype.__name__}"
        )
        for i, od in enumerate(self.outd):
            print(f"    out[{i}] {od['shape']} {od['dtype'].__name__}")

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(
            rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
        )

        if self._lut is not None:
            mapped = cv2.LUT(small, self._lut.view(np.uint8))
            return np.expand_dims(mapped.view(self.in_dtype), 0)

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
        total_ms = (t2 - t0) * 1000.0

        outs = [
            self.dequant(self.interp.get_tensor(od["index"]), i)
            for i, od in enumerate(self.outd)
        ]
        return outs, total_ms


def _nms_indices(x1, y1, x2, y2, confs, conf_thresh, iou_thresh):
    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], 1).tolist()
    idx = cv2.dnn.NMSBoxes(boxes, confs.tolist(), conf_thresh, iou_thresh)
    if len(idx) == 0:
        return []
    idx = np.array(idx).reshape(-1)
    return [int(i) for i in idx]


def _merge_lane_shapes(shapes, H, W, cx_gap=0.15):
    if not shapes:
        return shapes

    merged = []
    by_class = {}
    for s in shapes:
        by_class.setdefault(s["class_name"], []).append(s)

    for cname, group in by_class.items():
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
                    "conf": float(best["conf"]),
                }
            )

    return merged


def parse_lane(outputs, H, W):
    """
    detection 전용 lane parser.
    기존 proto/mask 기대 로직 제거.
    """
    global LANE_DEBUG_ONCE

    shapes = []
    if not outputs:
        return shapes

    det = outputs[0][0]

    if LANE_DEBUG_ONCE:
        print(f"[LANE DEBUG] raw output shape: {outputs[0].shape}")
        print(f"[LANE DEBUG] det shape before T: {det.shape}")
        _d = det.T if det.shape[0] > det.shape[1] else det
        _nc = _d.shape[0] - 4
        if _nc > 0:
            _scores = _d[4:, :].T
            _confs = np.max(_scores, axis=1)
            print(f"[LANE DEBUG] nc={_nc}  anchors={_d.shape[1]}")
            print(
                f"[LANE DEBUG] score max={_confs.max():.4f}  mean={_confs.mean():.4f}  >0.35: {(_confs > 0.35).sum()}  >0.50: {(_confs > 0.50).sum()}  >0.70: {(_confs > 0.70).sum()}"
            )
        LANE_DEBUG_ONCE = False

    if det.shape[0] > det.shape[1]:
        det = det.T

    nc = det.shape[0] - 4
    if nc <= 0:
        return shapes

    boxes = det[:4, :].T
    scores = det[4:, :].T

    cids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)
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

    idxs = _nms_indices(x1, y1, x2, y2, confs, LANE_CONF_THRESH, LANE_NMS_IOU)
    for i in idxs:
        bx1 = int(np.clip(x1[i], 0, W - 1))
        by1 = int(np.clip(y1[i], 0, H - 1))
        bx2 = int(np.clip(x2[i], 0, W - 1))
        by2 = int(np.clip(y2[i], 0, H - 1))
        cy_norm = float(((by1 + by2) / 2) / H)
        if cy_norm < LANE_ROI_TOP:
            continue
        cid = int(cids[i])
        shapes.append(
            {
                "class_id": cid,
                "class_name": LANE_NAMES.get(cid, "?"),
                "cx_norm": float(((bx1 + bx2) / 2) / W),
                "cy_norm": cy_norm,
                "bbox": (bx1, by1, bx2, by2),
                "conf": float(confs[i]),
            }
        )

    return _merge_lane_shapes(shapes, H, W)


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

    idxs = _nms_indices(x1, y1, x2, y2, confs, OBS_CONF_THRESH, OBS_NMS_IOU)
    for i in idxs:
        cid = int(cids[i])
        bx1 = int(np.clip(x1[i], 0, W - 1))
        by1 = int(np.clip(y1[i], 0, H - 1))
        bx2 = int(np.clip(x2[i], 0, W - 1))
        by2 = int(np.clip(y2[i], 0, H - 1))
        area = max(0, bx2 - bx1) * max(0, by2 - by1)
        dets.append(
            {
                "class_id": cid,
                "class_name": (
                    OBS_CLASS_NAMES[cid] if cid < len(OBS_CLASS_NAMES) else str(cid)
                ),
                "bbox": (bx1, by1, bx2, by2),
                "conf": float(confs[i]),
                "area": area,
            }
        )

    dets.sort(key=lambda d: (d["area"], d["conf"]), reverse=True)
    return dets[:1]


def _zone_weight(cy: float) -> float:
    if cy >= 0.66:
        return 3.0
    if cy >= 0.33:
        return 2.0
    return 1.0


def _x_zone(cx: float) -> str:
    if cx < 0.40:
        return "left"
    if cx > 0.60:
        return "right"
    return "center"


def _y_zone(cy: float) -> str:
    if cy < 0.33:
        return "top"
    if cy < 0.66:
        return "mid"
    return "bottom"


def _has_class(shapes, class_name: str, x=None, y=None) -> bool:
    for s in shapes:
        if s["class_name"] != class_name:
            continue
        if x is not None and _x_zone(float(s["cx_norm"])) != x:
            continue
        if y is not None and _y_zone(float(s["cy_norm"])) != y:
            continue
        return True
    return False


def _count_class(shapes, class_name: str, x=None, y=None) -> int:
    cnt = 0
    for s in shapes:
        if s["class_name"] != class_name:
            continue
        if x is not None and _x_zone(float(s["cx_norm"])) != x:
            continue
        if y is not None and _y_zone(float(s["cy_norm"])) != y:
            continue
        cnt += 1
    return cnt


def compute_lane_error(shapes):
    """
    line 기반 angle 계산.
    반환 범위는 대략 -10 ~ +10.
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
        return float(np.clip(err, -10.0, 10.0)), "ok"

    if left_mean is not None:
        err = ((_LANE_MID_X - left_mean) / 0.5) * 10.0
        return float(np.clip(err, -10.0, 10.0)), "left_only"

    err = ((_LANE_MID_X - right_mean) / 0.5) * 10.0
    return float(np.clip(err, -10.0, 10.0)), "right_only"


def _raw_classify(shapes):
    has_line_left = _has_class(shapes, "line", x="left")
    right_line_any = _has_class(shapes, "line", x="right")

    left_bottom_curve = _has_class(shapes, "curve", x="left", y="bottom")
    mid_eeu = _has_class(shapes, "eeu", y="mid")
    bottom_eeu = _has_class(shapes, "eeu", y="bottom")

    eeu_bottom_left = _count_class(shapes, "eeu", x="left", y="bottom")
    eeu_bottom_center = _count_class(shapes, "eeu", x="center", y="bottom")
    eeu_bottom_right = _count_class(shapes, "eeu", x="right", y="bottom")
    eeu_bottom_wide = (
        sum([eeu_bottom_left > 0, eeu_bottom_center > 0, eeu_bottom_right > 0]) >= 2
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


class IntersectionFSM:
    """
    얇은 현재 프레임 기반 분류기.
    기존 인터페이스(update 반환형) 유지.
    """

    def __init__(self, *args, **kwargs):
        self.cur = None

    def update(self, shapes):
        raw = _raw_classify(shapes)
        self.cur = raw
        if raw is None:
            return (False, None), raw
        return (True, raw), raw
