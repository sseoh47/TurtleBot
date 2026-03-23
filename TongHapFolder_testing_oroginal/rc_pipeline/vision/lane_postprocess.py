import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# ======================================
# names / thresholds
# ======================================
LANE_CLASS_NAMES = ["curve", "eeu", "line"]
OBS_CLASS_NAMES = ["KNU", "SL", "box", "car", "parking", "person"]

LANE_CONF_THRESH = 0.45
LANE_NMS_IOU = 0.45
OBS_CONF_THRESH = 0.75
OBS_NMS_IOU = 0.45

# straight는 상/중/하 전부 사용하므로 상단 ROI는 자르지 않음
LANE_ROI_TOP = 0.0

# 3분할 기준 (정확히 1/3, 2/3)
_X1 = 1.0 / 3.0
_X2 = 2.0 / 3.0
_Y1 = 1.0 / 3.0
_Y2 = 2.0 / 3.0

_LANE_MID_X = 0.5

# straight angle 계산용 y 가중치: 하단 우선
_ZONE_W_TOP = 0.20
_ZONE_W_MID = 0.30
_ZONE_W_BOTTOM = 0.50

OBS_OUTPUT_ID = {
    "KNU": 10,
    "box": 10,
    "SL": 2,
    "person": 3,
    "car": 4,
    "parking": 5,
}


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


def _has_class(
    shapes: List[Dict[str, Any]],
    cls_name: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
) -> bool:
    for s in shapes:
        if s.get("class_name") != cls_name:
            continue
        if x is not None and _x_zone(float(s["cx_norm"])) != x:
            continue
        if y is not None and _y_zone(float(s["cy_norm"])) != y:
            continue
        return True
    return False


def _count_class(
    shapes: List[Dict[str, Any]],
    cls_name: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
) -> int:
    c = 0
    for s in shapes:
        if s.get("class_name") != cls_name:
            continue
        if x is not None and _x_zone(float(s["cx_norm"])) != x:
            continue
        if y is not None and _y_zone(float(s["cy_norm"])) != y:
            continue
        c += 1
    return c


# ======================================
# lane parse: detection-only model
# 기대 출력 예시: [1, 7, N]  (cx, cy, w, h, class logits...)
# class 수는 LANE_CLASS_NAMES 길이로 계산
# ======================================
def parse_lane(outputs: List[np.ndarray], H: int, W: int) -> List[Dict[str, Any]]:
    shapes: List[Dict[str, Any]] = []
    if not outputs:
        return shapes

    det_raw = None
    for o in outputs:
        if getattr(o, "ndim", 0) == 3:
            det_raw = o
            break

    if det_raw is None:
        return shapes

    det = det_raw[0]
    if det.ndim != 2:
        return shapes

    nc = len(LANE_CLASS_NAMES)
    if det.shape[0] < 4 + nc:
        return shapes

    cls_logits = det[4 : 4 + nc, :]
    confs = cls_logits.max(axis=0)
    clss = cls_logits.argmax(axis=0)

    keep = confs > LANE_CONF_THRESH
    if not np.any(keep):
        return shapes

    det = det[:, keep]
    confs = confs[keep]
    clss = clss[keep]

    boxes = det[:4, :].T
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    bw = boxes[:, 2]
    bh = boxes[:, 3]

    # 출력은 정규화된 xywh라고 가정
    x1 = ((cx - bw / 2.0) * W).clip(0, W - 1)
    y1 = ((cy - bh / 2.0) * H).clip(0, H - 1)
    x2 = ((cx + bw / 2.0) * W).clip(0, W - 1)
    y2 = ((cy + bh / 2.0) * H).clip(0, H - 1)

    bboxes_xywh = np.stack(
        [x1, y1, np.maximum(1, x2 - x1), np.maximum(1, y2 - y1)], axis=1
    ).tolist()
    idxs = cv2.dnn.NMSBoxes(
        bboxes_xywh,
        confs.astype(float).tolist(),
        LANE_CONF_THRESH,
        LANE_NMS_IOU,
    )
    if len(idxs) == 0:
        return shapes

    idxs = np.array(idxs).reshape(-1)

    for i in idxs:
        bx1, by1, bx2, by2 = map(int, [x1[i], y1[i], x2[i], y2[i]])
        cls_id = int(clss[i])
        if cls_id < 0 or cls_id >= len(LANE_CLASS_NAMES):
            continue

        cx_norm = ((bx1 + bx2) / 2.0) / W
        cy_norm = ((by1 + by2) / 2.0) / H

        if cy_norm < LANE_ROI_TOP:
            continue

        shapes.append(
            {
                "type": "lane_shape",
                "class_id": cls_id,
                "class_name": LANE_CLASS_NAMES[cls_id],
                "conf": float(confs[i]),
                "bbox": [bx1, by1, bx2, by2],
                "cx_norm": float(cx_norm),
                "cy_norm": float(cy_norm),
            }
        )

    return shapes


# ======================================
# obstacle parse: detection-only model
# 기대 출력: [1, 4 + nc, N]
# ======================================
def parse_obstacle(outputs: List[np.ndarray], H: int, W: int) -> List[Dict[str, Any]]:
    obs: List[Dict[str, Any]] = []
    if not outputs:
        return obs

    det_raw = None
    for o in outputs:
        if getattr(o, "ndim", 0) == 3:
            det_raw = o
            break

    if det_raw is None:
        return obs

    det = det_raw[0]
    if det.ndim != 2:
        return obs

    nc = len(OBS_CLASS_NAMES)
    if det.shape[0] < 4 + nc:
        return obs

    cls_logits = det[4 : 4 + nc, :]
    confs = cls_logits.max(axis=0)
    clss = cls_logits.argmax(axis=0)

    keep = confs > OBS_CONF_THRESH
    if not np.any(keep):
        return obs

    det = det[:, keep]
    confs = confs[keep]
    clss = clss[keep]

    boxes = det[:4, :].T
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    bw = boxes[:, 2]
    bh = boxes[:, 3]

    x1 = ((cx - bw / 2.0) * W).clip(0, W - 1)
    y1 = ((cy - bh / 2.0) * H).clip(0, H - 1)
    x2 = ((cx + bw / 2.0) * W).clip(0, W - 1)
    y2 = ((cy + bh / 2.0) * H).clip(0, H - 1)

    bboxes_xywh = np.stack(
        [x1, y1, np.maximum(1, x2 - x1), np.maximum(1, y2 - y1)], axis=1
    ).tolist()
    idxs = cv2.dnn.NMSBoxes(
        bboxes_xywh,
        confs.astype(float).tolist(),
        OBS_CONF_THRESH,
        OBS_NMS_IOU,
    )
    if len(idxs) == 0:
        return obs

    idxs = np.array(idxs).reshape(-1)

    items = []
    for i in idxs:
        bx1, by1, bx2, by2 = map(int, [x1[i], y1[i], x2[i], y2[i]])
        cls_id = int(clss[i])
        if cls_id < 0 or cls_id >= len(OBS_CLASS_NAMES):
            continue
        area = max(1, bx2 - bx1) * max(1, by2 - by1)
        items.append(
            {
                "type": "obstacle",
                "class_id": cls_id,
                "class_name": OBS_CLASS_NAMES[cls_id],
                "conf": float(confs[i]),
                "bbox": [bx1, by1, bx2, by2],
                "area": int(area),
            }
        )

    items.sort(key=lambda x: (-x["area"], -x["conf"]))
    return items


# ======================================
# straight angle
# ======================================
def compute_lane_error(shapes: List[Dict[str, Any]]) -> Tuple[float, str]:
    """
    straight용 angle 계산
    - line만 사용
    - 상/중/하 전체 사용
    - center only는 제외
    - left only / right only 허용
    - 하단에 더 큰 가중치 부여
    """
    lines = [s for s in shapes if s.get("class_name") == "line"]
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
        # center only는 angle 계산에서 제외

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


# ======================================
# intersection / branch classify
# ======================================
def _raw_classify(shapes: List[Dict[str, Any]]) -> Optional[str]:
    # ---------- left_t ----------
    left_mid_curve = _has_class(shapes, "curve", x="left", y="mid")
    left_bottom_curve = _has_class(shapes, "curve", x="left", y="bottom")

    mid_eeu = _has_class(shapes, "eeu", y="mid")
    bottom_eeu = _has_class(shapes, "eeu", y="bottom")
    right_line_any = _has_class(shapes, "line", x="right")
    has_line_left = _has_class(shapes, "line", x="left")

    left_t_curve_cond = left_bottom_curve or left_mid_curve
    left_t_eeu_cond = (mid_eeu or bottom_eeu) and right_line_any and not has_line_left

    # ---------- down_t ----------
    eeu_bottom_left = _count_class(shapes, "eeu", x="left", y="bottom")
    eeu_bottom_center = _count_class(shapes, "eeu", x="center", y="bottom")
    eeu_bottom_right = _count_class(shapes, "eeu", x="right", y="bottom")

    eeu_mid_left = _count_class(shapes, "eeu", x="left", y="mid")
    eeu_mid_center = _count_class(shapes, "eeu", x="center", y="mid")
    eeu_mid_right = _count_class(shapes, "eeu", x="right", y="mid")

    eeu_bottom_count = sum(
        [
            eeu_bottom_left > 0,
            eeu_bottom_center > 0,
            eeu_bottom_right > 0,
        ]
    )
    eeu_mid_any = (eeu_mid_left > 0) or (eeu_mid_center > 0) or (eeu_mid_right > 0)

    down_t_cond = (eeu_bottom_count >= 2) or (eeu_bottom_count >= 1 and eeu_mid_any)

    # ---------- cross ----------
    curve_left_any = _has_class(shapes, "curve", x="left", y="mid") or _has_class(
        shapes, "curve", x="left", y="bottom"
    )
    curve_right_any = _has_class(shapes, "curve", x="right", y="mid") or _has_class(
        shapes, "curve", x="right", y="bottom"
    )

    cross_cond = curve_left_any and curve_right_any

    # ---------- 우선순위 ----------
    if left_t_curve_cond:
        return "left_t"

    if down_t_cond:
        return "down_t"

    if cross_cond and not down_t_cond:
        return "cross"

    if left_t_eeu_cond:
        return "left_t"

    return None


def _resolve_cross_down(shapes: List[Dict[str, Any]]) -> Optional[str]:
    raw = _raw_classify(shapes)
    if raw in ("left_t", "down_t", "cross"):
        return raw
    return None


class IntersectionFSM:
    """현재 프레임 기반의 얇은 분류기."""

    def __init__(self, *args, **kwargs):
        self.cur = None

    def update(self, shapes: List[Dict[str, Any]]):
        raw = _raw_classify(shapes)
        resolved = _resolve_cross_down(shapes)
        self.cur = resolved
        if resolved is None:
            return (False, None), raw
        return (True, resolved), raw


# ======================================
# result converters
# ======================================
def convert_lane_result(p_le: float, p_ls: str, p_is: bool, p_it: Optional[str]):
    angle = float(round(p_le, 2))
    if p_is:
        inter_map = {
            "left_t": 6,
            "down_t": 8,
            "cross": 9,
        }
        if p_it in inter_map:
            return inter_map[p_it], angle
    if p_ls == "lost":
        return 0, 0.0
    return 1, angle


def convert_object_result(obs_list: List[Dict[str, Any]]) -> Optional[int]:
    if not obs_list:
        return None
    name = obs_list[0].get("class_name")
    return OBS_OUTPUT_ID.get(name)


def make_output(
    le: float,
    ls: str,
    is_inter: bool,
    itype: Optional[str],
    obs_list: List[Dict[str, Any]],
):
    """
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
        oid = OBS_OUTPUT_ID.get(o.get("class_name"))
        if oid is not None:
            result.append([oid, None])

    return result
