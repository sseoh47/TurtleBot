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

## 라즈베리파이에서 2개의 EdgeTPU 써서 차선 인식 + 장애물 인식 실행,
## 결과를 주행 제어용값으로 바꿔주는 통합 추론 코드
## 입력 : 카메라
## 모델 : 1) lane model(차선/교차로 관련 객체 검출), 2) obs model(장애물/사람/차 등 검출)
## 출력 : 주행 제어에 넘길 수 있는 ID 체계 값
## 영상 -> AI추론 -> 차선/장애물 판단 -> 주행 로직용 숫자 결과 생성

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
## 모듈 import 안전하게 시도

## pycoral 있으면 EdgeTPU용 기능 키고
try:
    from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus

    EDGETPU_AVAILABLE = True
except ImportError:
    EDGETPU_AVAILABLE = False

    def list_edge_tpus():
        return []


## TFLite 인터프리터를 가져온다
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
## 모델/시각화에 쓰는 설정값
# CONF_THRESH = 0.35  # 모델이 예측한 박스의 신뢰도 해당 값 이상일 때만 채택
# NMS_IOU = 0.35  # 겹치는 박스 적극 제거
# LANE_DEBUG_ONCE = True  # 차선 모델 디버그 정보를 첫 프레임에서 한 번, 출력 (확인 후 False로 변경)
LANE_CONF_THRESH = 0.75
LANE_NMS_IOU = 0.35

OBS_CONF_THRESH = 0.60
OBS_NMS_IOU = 0.35

DRAW_LANE_MASK = False
# ROI: 상단 몇 % 를 제거할지 (0.0 ~ 1.0)
LANE_ROI_TOP = 0.10

LANE_NAMES = {0: "curve", 1: "eeu", 2: "line"}  # 곡선, 가로선, 세로선
LANE_COLOR = {
    0: (0, 255, 255),
    1: (0, 255, 0),
    2: (0, 0, 255),
}  # 차선 클래스 색 (노랑, 초록, 빨강)
OBS_PALETTE = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 200, 80),
    (200, 80, 255),
    (80, 255, 200),
]  # 장애물 클래스에 쓸 색상 목록

# ★ 학습한 6개 클래스 (data.yaml names 순서와 동일)
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
}  ## 교차로/상황 설명(내부용 상태 키 -> 화면 표시용 영어 문구로 바꿈)


# ══════════════════════════════════════════════
# EdgeTPU 엔진
# ══════════════════════════════════════════════
## 1. 모델 실행 엔진, 2. 차선/장애물 탐지 결과 후처리, 3. 차선 기반 주행 판단 로직
## 카메라 프레임 -> TPU/TFLite로 추론 -> 박스 결과 정리 -> 차선 형태 해석 -> 조향 오차 계산
class EdgeTPUEngine:
    """
    device 파라미터 (공식 형식 coral.ai/docs/edgetpu/multiple-edgetpu):
      'usb:0'  -> 첫 번째 USB Coral
      'usb:1'  -> 두 번째 USB Coral
      ''       -> 자동 선택 (1개일 때)
    """

    ## 해당 모델을 어떤 장치에서 어떤 입력 형식으로 돌릴지 준비
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

        # preprocess LUT 미리 계산
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
            f"  [{tag}] {Path(model_path).name} "
            f"({self.img_h}x{self.img_w}) {self.in_dtype.__name__}"
        )
        for i, od in enumerate(self.outd):
            print(f"    out[{i}] {od['shape']} {od['dtype'].__name__}")

        if self.on_tpu:
            try:
                dummy = np.zeros((1, self.img_h, self.img_w, 3), dtype=self.in_dtype)
                self.interp.set_tensor(self.ind[0]["index"], dummy)
                t0 = time.perf_counter()
                self.interp.invoke()
                ms = (time.perf_counter() - t0) * 1000.0
                print(
                    f"    [TPU diag] 워밍업 invoke={ms:.1f}ms "
                    f"({'EdgeTPU 정상' if ms < 50 else '⚠ CPU fallback 의심 >50ms'})"
                )
            except Exception as e:
                print(f"    [TPU diag] 워밍업 실패: {e}")

    ## 카에 따라 변환
    ## int8 모델이면 quantization 맞춰서 int8, uint8모델이면 uint8로, float는 0~1 정규화한 float32
    ## =>원본 openCV 이미지 -> 모델에 줄 tensor
    ## 전처리 부분메라 프레임을 모델 입력 형태로 바꾸는 함수
    ## BGR -> RGB 변환
    ## 모델 입력 크기로 resize
    ## 입력 dtype 코드 짧아짐!!
    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(
            rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
        )

        if self._lut is not None:
            mapped = cv2.LUT(small, self._lut.view(np.uint8))
            return np.expand_dims(mapped.view(self.in_dtype), 0)

        return np.expand_dims(small.astype(np.float32) * (1.0 / 255.0), 0)

    ## 출력 텐서가 int8/uint8 양자화 모델일때, float값으로 복원
    def dequant(self, t, idx):
        sc, zp = self.outd[idx]["quantization"]
        if self.outd[idx]["dtype"] in (np.uint8, np.int8) and sc > 0:
            return (t.astype(np.float32) - zp) * sc
        return t.astype(np.float32)

    ## 실제 추론 함수 : 프레임 1장 넣으면 추론 결과와 추론 시간 반환
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


# ══════════════════════════════════════════════
# Lane 후처리
# ══════════════════════════════════════════════
## 차선 탐지 모델 출력 후처리
## yolov는 [cx, cy, w, h, class scores...] -> 의미있는 detection 리스트로 반환
def parse_lane(outputs, H, W):
    """
    YOLOv8-det (detection 전용) 출력 파싱.
    output[0]: [1, 4+nc, num_anchors] 또는 [1, num_anchors, 4+nc]
    반환: shapes 리스트 (overlay 없음)
    """
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
## 차선 탐지 결과 보고 현재 도로 상황이 직선인지, T자 인지, 곡선인지 분류
## 왜 없지??????????????????????????????????????????????????????
# def classify_intersection(shapes, history):
#     lines = [s for s in shapes if s["class_name"] == "line"]
#     curves = [s for s in shapes if s["class_name"] == "curve"]
#     eeus = [s for s in shapes if s["class_name"] == "eeu"]

#     has_eeu, has_curve, has_line = bool(eeus), bool(curves), bool(lines)

#     if has_eeu:
#         if has_curve:
#             cx = float(np.mean([s["cx_norm"] for s in curves]))
#             return "curve_right" if cx >= 0.5 else "curve_left"
#         rec = history[-11:]
#         pr = sum(
#             1
#             for f in rec
#             if any(s["class_name"] == "curve" and s["cx_norm"] >= 0.5 for s in f)
#         )
#         pl = sum(
#             1
#             for f in rec
#             if any(s["class_name"] == "curve" and s["cx_norm"] < 0.5 for s in f)
#         )
#         if pr > pl and pr >= 2:
#             return "curve_right"
#         if pl > pr and pl >= 2:
#             return "curve_left"
#         return "down_t"
#     if has_curve:
#         lc = [s for s in curves if s["cx_norm"] < 0.5]
#         rc = [s for s in curves if s["cx_norm"] >= 0.5]
#         ll = [s for s in lines if s["cx_norm"] < 0.5]
#         rl = [s for s in lines if s["cx_norm"] >= 0.5]
#         if lc and rc:
#             return "cross"
#         if rc and ll:
#             return "right_t"
#         if lc and rl:
#             return "left_t"
#         return "approaching"
#     if has_line:
#         return "straight"
#     return None

# =========================
# 3x3 zone helper
# =========================
_X1 = 1.0 / 3.0
_X2 = 2.0 / 3.0
_Y1 = 1.0 / 3.0
_Y2 = 2.0 / 3.0

_LANE_MID_X = 0.5

# straight angle 계산용 y 가중치
# straight는 상/중/하 다 보되, 하단 가중치를 더 줌
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


## 조향 제어용 : 차량이 차선 중앙에서 얼마나 벗어났는지 계산
## return : (error, status) : -` ~ 1 정도 범위의 좌우 오차, 차선인식상태`
def compute_lane_error(shapes):
    """
    straight용 angle 원재료 계산
    - line만 사용
    - top/mid/bottom 모두 사용
    - center only는 제외
    - left only / right only 허용
    - 반환: (error, status)
      error 범위: 대략 -10 ~ +10
    """
    lines = [s for s in shapes if s["class_name"] == "line"]
    if not lines:
        return 0.0, "lost"

    left_pts = []
    right_pts = []
    center_pts = []

    for s in lines:
        cx = float(s["cx_norm"])
        cy = float(s["cy_norm"])
        w = _zone_weight(cy)
        xz = _x_zone(cx)

        if xz == "left":
            left_pts.append((cx, w))
        elif xz == "right":
            right_pts.append((cx, w))
        else:
            center_pts.append((cx, w))

    def weighted_mean(items):
        if not items:
            return None
        ws = sum(w for _, w in items)
        if ws <= 1e-6:
            return None
        return sum(x * w for x, w in items) / ws

    left_mean = weighted_mean(left_pts)
    right_mean = weighted_mean(right_pts)

    # center only는 제외
    if left_mean is None and right_mean is None:
        return 0.0, "lost"

    # 좌우 둘 다 있으면 lane center 사용
    if left_mean is not None and right_mean is not None:
        lane_center = (left_mean + right_mean) * 0.5
        err = ((_LANE_MID_X - lane_center) / 0.5) * 10.0
        err = float(np.clip(err, -10.0, 10.0))
        return err, "ok"

    # 왼쪽만
    if left_mean is not None:
        err = ((_LANE_MID_X - left_mean) / 0.5) * 10.0
        err = float(np.clip(err, -10.0, 10.0))
        return err, "left_only"

    # 오른쪽만
    err = ((_LANE_MID_X - right_mean) / 0.5) * 10.0
    err = float(np.clip(err, -10.0, 10.0))
    return err, "right_only"


def _raw_classify(shapes):
    """
    현재 프레임 기준 특수상황 분류.

    규칙 요약
    - straight 판단용 lane은 상/중/하 모두 사용한다. (angle 계산은 compute_lane_error 담당)
    - left_t / down_t는 가까운 구간 우선이므로 mid/bottom을 더 강하게 본다.
    - cross는 eeu가 아니라 curve 쌍 기반 직진 상황으로 본다.
    - 현재 프레임만 사용한다.

    반환:
      - "left_t"
      - "down_t"
      - "cross"
      - None
    """

    # ---------- straight 참고용 lane 존재 ----------
    has_line_left = _has_class(shapes, "line", x="left")
    right_line_any = _has_class(shapes, "line", x="right")

    # ---------- left_t ----------
    # 1) 왼쪽 하단 curve 가 보이면 좌회전
    left_bottom_curve = _has_class(shapes, "curve", x="left", y="bottom")

    # 2) curve처럼 보여도 실제 검출은 eeu + right line 으로 나올 수 있음
    #    단, 상단은 너무 멀어서 제외하고 mid/bottom만 사용
    mid_eeu = _has_class(shapes, "eeu", y="mid")
    bottom_eeu = _has_class(shapes, "eeu", y="bottom")

    # ---------- down_t ----------
    # 하단에 eeu가 길게 깔린 상황
    # => bottom에서 eeu가 좌/중/우 중 2영역 이상 차지하면 wide로 해석
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

    # ---------- cross ----------
    # curve 기반 직진 상황
    # top은 너무 멀어서 제외하고 mid/bottom만 사용
    curve_mid_left = _has_class(shapes, "curve", x="left", y="mid")
    curve_mid_right = _has_class(shapes, "curve", x="right", y="mid")
    curve_bottom_left = _has_class(shapes, "curve", x="left", y="bottom")
    curve_bottom_right = _has_class(shapes, "curve", x="right", y="bottom")

    cross_cond = (curve_mid_left and curve_mid_right) or (
        curve_bottom_left and curve_bottom_right
    )

    # ---------- 우선순위 ----------
    # 1) 하단 curve -> left_t
    if left_bottom_curve:
        return "left_t"

    # 2) 하단 eeu wide -> down_t
    if eeu_bottom_wide:
        return "down_t"

    # 3) cross는 curve pair
    #    단, eeu 기반 down_t와 겹치면 down_t 우선
    if cross_cond and not eeu_bottom_wide:
        return "cross"

    # 4) mid/bottom eeu + right line 기반 left_t
    #    상단 제외, 중단/하단 우선
    if (mid_eeu or bottom_eeu) and right_line_any and not has_line_left:
        return "left_t"

    return None


def _resolve_cross_down(shapes):
    raw = _raw_classify(shapes)
    if raw in ("left_t", "down_t", "cross"):
        return raw
    return None


class IntersectionFSM:
    """
    현재 프레임 기반의 얇은 분류기.
    기존 인터페이스(update 반환형)는 유지.
    """

    def __init__(self, *args, **kwargs):
        self.cur = None

    def update(self, shapes):
        raw = _raw_classify(shapes)
        resolved = _resolve_cross_down(shapes)

        self.cur = resolved
        if resolved is None:
            return (False, None), raw
        return (True, resolved), raw


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════
def _draw_frame(d: dict) -> np.ndarray:
    ## 이부분도 수정. 줄어들었다.
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


# 객체 클래스명 -> 출력 ID 맵핑 (0~10 체계)
# KNU, 는 10으로 출력 box는 아직 미정
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

    반환: 패킷 리스트
      예) [[1, 0.123], [4, None], [3, None]]
    """
    err = round(le, 4)
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
class FrameGrabber(threading.Thread):
    """카메라 캡처 전담 - 항상 최신 프레임 유지. 영상 끝나면 eof=True"""

    def __init__(self, cap, is_file=False):
        super().__init__(daemon=True)
        self.cap = cap
        self.is_file = is_file
        self._f = None
        self._lk = threading.Lock()
        self._q = queue.Queue(maxsize=8) if is_file else None
        self.alive = True
        self.eof = False

    def run(self):
        while self.alive:
            if self.is_file:
                ok, f = self.cap.read()
                if ok:
                    try:
                        self._q.put(f, timeout=2.0)
                    except queue.Full:
                        pass
                else:
                    self.eof = True
                    break
            else:
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

    is_file = Path(str(source)).exists()
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
## 조금 줄었다.
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

    is_file = Path(str(source)).exists()

    if not is_file:
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, fps)
    else:
        cap = cv2.VideoCapture(str(source))

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    save_fps = src_fps if is_file and src_fps > 0 else fps
    print(
        f"[INFO] 해상도: {orig_w}x{orig_h}  obs_skip={obs_skip}  disp_scale={disp_scale}"
    )
    print(f"[INFO] 저장 FPS: {save_fps:.2f}  (원본: {src_fps:.2f})\n")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0

    grabber = FrameGrabber(cap, is_file=is_file)
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
    최종 lane class 체계
      0  = 차선 없음
      1  = 일반 차선
      6  = left_t
      8  = down_t
      9  = cross
      10 = 물류 pass 계열 (현재 lane에서는 미사용)
    """
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


def convert_object_result(obs_list):
    """
    파일 객체 결과를 최종 obj class 체계로 변환
      2  = SL
      3  = person
      4  = car
      5  = parking
      10 = KNU / box / 물류 pass 계열
      없음 = None
    """
    if not obs_list:
        return None

    name = obs_list[0]["class_name"]
    obj_map = {
        "SL": 2,
        "person": 3,
        "car": 4,
        "parking": 5,
        "KNU": 10,
        "box": 10,
        # "pass": 10,
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
        cam_w=320,
        cam_h=320,
    ):
        self.coral = coral
        self.source = source  ## 츠가
        self.fsm = IntersectionFSM()  ## 추가

        ## 추가 여기부터
        if use_edgetpu and EDGETPU_AVAILABLE:
            tpus = list_edge_tpus()
            print(f"[INFO] 연결된 EdgeTPU 장치: {len(tpus)}개")
            for i, t in enumerate(tpus):
                print(f"  [{i}] type={t.get('type', '?')}  path={t.get('path', '?')}")
        ## 추가 여기까지

        if coral == 2 and use_edgetpu:
            print("[INFO] lane 모델 -> usb:0 로드")
            self.lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=True, device="usb:0")
            time.sleep(1.0)
            print("[INFO] obs 모델 -> usb:1 로드")
            self.obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=True, device="usb:1")
        else:
            self.lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=use_edgetpu)
            self.obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=use_edgetpu)

        # self.cap = cv2.VideoCapture(source)
        ## ========================
        ## 실시간 카메라 지연 줄이려면 보통 V4L2 지정한 쪽이 더 나음
        ## ========================
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"source open failed: {source}")

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # self.fsm = IntersectionFSM() ## 주석 처리

    ## 추가
    def _read_latest_frame(self):
        if self.cap is None:
            return None

        if isinstance(self.source, int):
            ok = self.cap.grab()
            if not ok:
                return None
            ok, frame = self.cap.retrieve()
            if not ok:
                return None
            return frame

        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def step(self) -> Optional[DualInferenceResult]:
        # ok, frame = self.cap.read()
        # if not ok:
        #     return None

        frame = self._read_latest_frame()
        if frame is None:
            return None

        H, W = frame.shape[:2]

        lane_outs, _ = self.lane_eng.infer(frame)
        lane_shapes = parse_lane(lane_outs, H, W)
        p_le, p_ls = compute_lane_error(lane_shapes)
        (p_is, p_it), _ = self.fsm.update(lane_shapes)

        # obs_outs, _ = self.obs_eng.infer(frame)
        obs_outs, obs_ms = self.obs_eng.infer(frame)
        obs_list = parse_obstacle(obs_outs, H, W)

        line_id, angle = convert_lane_result(p_le, p_ls, p_is, p_it)
        obj_id = convert_object_result(obs_list)

        # print(
        #     f"[AI] line_id={line_id} angle={angle} obj_id={obj_id} "
        #     f"lane_status={p_ls} inter={p_it if p_is else None} "
        #     f"lane={lane_ms:.0f}ms(pre={self.lane_eng.pre_ms:.1f}/inv={self.lane_eng.invoke_ms:.1f}) "
        #     f"obs={obs_ms:.0f}ms raw={raw}"
        # )

        return DualInferenceResult(
            line_id=line_id,
            angle=angle,
            obj_id=obj_id,
            lane_status=p_ls,
            inter_type=p_it if p_is else None,
        )

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


###===============================================================
## 팀장코드~ 종료지점~
###===============================================================

# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
## 기존 코드와 바뀜(add_argument쪽이 좀 더 줄어들었다.)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
FPS/지연 줄이기 추천:
  --coral 2
  --cam-w 320 --cam-h 240
  DRAW_LANE_MASK = False
        """,
    )
    ap.add_argument("--lane-model", default="best_int8_edgetpu.tflite")
    ap.add_argument("--obs-model", default="obs_int8_edgetpu.tflite")
    ap.add_argument("--source", default="0")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--coral", type=int, default=1, choices=[1, 2])
    ap.add_argument("--obs-skip", type=int, default=2)
    ap.add_argument("--no-edgetpu", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--disp-scale", type=float, default=1.0)
    ap.add_argument("--cam-w", type=int, default=640)
    ap.add_argument("--cam-h", type=int, default=480)
    args = ap.parse_args()

    src = int(args.source) if str(args.source).isdigit() else args.source
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

## 라즈베리파이에서 2개의 EdgeTPU 써서 차선 인식 + 장애물 인식 실행,
## 결과를 주행 제어용값으로 바꿔주는 통합 추론 코드
## 입력 : 카메라
## 모델 : 1) lane model(차선/교차로 관련 객체 검출), 2) obs model(장애물/사람/차 등 검출)
## 출력 : 주행 제어에 넘길 수 있는 ID 체계 값
## 영상 -> AI추론 -> 차선/장애물 판단 -> 주행 로직용 숫자 결과 생성

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
## 모듈 import 안전하게 시도

## pycoral 있으면 EdgeTPU용 기능 키고
try:
    from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus

    EDGETPU_AVAILABLE = True
except ImportError:
    EDGETPU_AVAILABLE = False

    def list_edge_tpus():
        return []


## TFLite 인터프리터를 가져온다
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
## 모델/시각화에 쓰는 설정값
# CONF_THRESH = 0.35  # 모델이 예측한 박스의 신뢰도 해당 값 이상일 때만 채택
# NMS_IOU = 0.35  # 겹치는 박스 적극 제거
# LANE_DEBUG_ONCE = True  # 차선 모델 디버그 정보를 첫 프레임에서 한 번, 출력 (확인 후 False로 변경)
LANE_CONF_THRESH = 0.75
LANE_NMS_IOU = 0.35

OBS_CONF_THRESH = 0.60
OBS_NMS_IOU = 0.35

DRAW_LANE_MASK = False
# ROI: 상단 몇 % 를 제거할지 (0.0 ~ 1.0)
LANE_ROI_TOP = 0.10

LANE_NAMES = {0: "curve", 1: "eeu", 2: "line"}  # 곡선, 가로선, 세로선
LANE_COLOR = {
    0: (0, 255, 255),
    1: (0, 255, 0),
    2: (0, 0, 255),
}  # 차선 클래스 색 (노랑, 초록, 빨강)
OBS_PALETTE = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 200, 80),
    (200, 80, 255),
    (80, 255, 200),
]  # 장애물 클래스에 쓸 색상 목록

# ★ 학습한 6개 클래스 (data.yaml names 순서와 동일)
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
}  ## 교차로/상황 설명(내부용 상태 키 -> 화면 표시용 영어 문구로 바꿈)


# ══════════════════════════════════════════════
# EdgeTPU 엔진
# ══════════════════════════════════════════════
## 1. 모델 실행 엔진, 2. 차선/장애물 탐지 결과 후처리, 3. 차선 기반 주행 판단 로직
## 카메라 프레임 -> TPU/TFLite로 추론 -> 박스 결과 정리 -> 차선 형태 해석 -> 조향 오차 계산
class EdgeTPUEngine:
    """
    device 파라미터 (공식 형식 coral.ai/docs/edgetpu/multiple-edgetpu):
      'usb:0'  -> 첫 번째 USB Coral
      'usb:1'  -> 두 번째 USB Coral
      ''       -> 자동 선택 (1개일 때)
    """

    ## 해당 모델을 어떤 장치에서 어떤 입력 형식으로 돌릴지 준비
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

        # preprocess LUT 미리 계산
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
            f"  [{tag}] {Path(model_path).name} "
            f"({self.img_h}x{self.img_w}) {self.in_dtype.__name__}"
        )
        for i, od in enumerate(self.outd):
            print(f"    out[{i}] {od['shape']} {od['dtype'].__name__}")

        if self.on_tpu:
            try:
                dummy = np.zeros((1, self.img_h, self.img_w, 3), dtype=self.in_dtype)
                self.interp.set_tensor(self.ind[0]["index"], dummy)
                t0 = time.perf_counter()
                self.interp.invoke()
                ms = (time.perf_counter() - t0) * 1000.0
                print(
                    f"    [TPU diag] 워밍업 invoke={ms:.1f}ms "
                    f"({'EdgeTPU 정상' if ms < 50 else '⚠ CPU fallback 의심 >50ms'})"
                )
            except Exception as e:
                print(f"    [TPU diag] 워밍업 실패: {e}")

    ## 카에 따라 변환
    ## int8 모델이면 quantization 맞춰서 int8, uint8모델이면 uint8로, float는 0~1 정규화한 float32
    ## =>원본 openCV 이미지 -> 모델에 줄 tensor
    ## 전처리 부분메라 프레임을 모델 입력 형태로 바꾸는 함수
    ## BGR -> RGB 변환
    ## 모델 입력 크기로 resize
    ## 입력 dtype 코드 짧아짐!!
    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(
            rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
        )

        if self._lut is not None:
            mapped = cv2.LUT(small, self._lut.view(np.uint8))
            return np.expand_dims(mapped.view(self.in_dtype), 0)

        return np.expand_dims(small.astype(np.float32) * (1.0 / 255.0), 0)

    ## 출력 텐서가 int8/uint8 양자화 모델일때, float값으로 복원
    def dequant(self, t, idx):
        sc, zp = self.outd[idx]["quantization"]
        if self.outd[idx]["dtype"] in (np.uint8, np.int8) and sc > 0:
            return (t.astype(np.float32) - zp) * sc
        return t.astype(np.float32)

    ## 실제 추론 함수 : 프레임 1장 넣으면 추론 결과와 추론 시간 반환
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


# ══════════════════════════════════════════════
# Lane 후처리
# ══════════════════════════════════════════════
## 차선 탐지 모델 출력 후처리
## yolov는 [cx, cy, w, h, class scores...] -> 의미있는 detection 리스트로 반환
def parse_lane(outputs, H, W):
    """
    YOLOv8-det (detection 전용) 출력 파싱.
    output[0]: [1, 4+nc, num_anchors] 또는 [1, num_anchors, 4+nc]
    반환: shapes 리스트 (overlay 없음)
    """
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
## 차선 탐지 결과 보고 현재 도로 상황이 직선인지, T자 인지, 곡선인지 분류
## 왜 없지??????????????????????????????????????????????????????
# def classify_intersection(shapes, history):
#     lines = [s for s in shapes if s["class_name"] == "line"]
#     curves = [s for s in shapes if s["class_name"] == "curve"]
#     eeus = [s for s in shapes if s["class_name"] == "eeu"]

#     has_eeu, has_curve, has_line = bool(eeus), bool(curves), bool(lines)

#     if has_eeu:
#         if has_curve:
#             cx = float(np.mean([s["cx_norm"] for s in curves]))
#             return "curve_right" if cx >= 0.5 else "curve_left"
#         rec = history[-11:]
#         pr = sum(
#             1
#             for f in rec
#             if any(s["class_name"] == "curve" and s["cx_norm"] >= 0.5 for s in f)
#         )
#         pl = sum(
#             1
#             for f in rec
#             if any(s["class_name"] == "curve" and s["cx_norm"] < 0.5 for s in f)
#         )
#         if pr > pl and pr >= 2:
#             return "curve_right"
#         if pl > pr and pl >= 2:
#             return "curve_left"
#         return "down_t"
#     if has_curve:
#         lc = [s for s in curves if s["cx_norm"] < 0.5]
#         rc = [s for s in curves if s["cx_norm"] >= 0.5]
#         ll = [s for s in lines if s["cx_norm"] < 0.5]
#         rl = [s for s in lines if s["cx_norm"] >= 0.5]
#         if lc and rc:
#             return "cross"
#         if rc and ll:
#             return "right_t"
#         if lc and rl:
#             return "left_t"
#         return "approaching"
#     if has_line:
#         return "straight"
#     return None

# =========================
# 3x3 zone helper
# =========================
_X1 = 1.0 / 3.0
_X2 = 2.0 / 3.0
_Y1 = 1.0 / 3.0
_Y2 = 2.0 / 3.0

_LANE_MID_X = 0.5

# straight angle 계산용 y 가중치
# straight는 상/중/하 다 보되, 하단 가중치를 더 줌
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


## 조향 제어용 : 차량이 차선 중앙에서 얼마나 벗어났는지 계산
## return : (error, status) : -` ~ 1 정도 범위의 좌우 오차, 차선인식상태`
def compute_lane_error(shapes):
    """
    straight용 angle 원재료 계산
    - line만 사용
    - top/mid/bottom 모두 사용
    - center only는 제외
    - left only / right only 허용
    - 반환: (error, status)
      error 범위: 대략 -10 ~ +10
    """
    lines = [s for s in shapes if s["class_name"] == "line"]
    if not lines:
        return 0.0, "lost"

    left_pts = []
    right_pts = []
    center_pts = []

    for s in lines:
        cx = float(s["cx_norm"])
        cy = float(s["cy_norm"])
        w = _zone_weight(cy)
        xz = _x_zone(cx)

        if xz == "left":
            left_pts.append((cx, w))
        elif xz == "right":
            right_pts.append((cx, w))
        else:
            center_pts.append((cx, w))

    def weighted_mean(items):
        if not items:
            return None
        ws = sum(w for _, w in items)
        if ws <= 1e-6:
            return None
        return sum(x * w for x, w in items) / ws

    left_mean = weighted_mean(left_pts)
    right_mean = weighted_mean(right_pts)

    # center only는 제외
    if left_mean is None and right_mean is None:
        return 0.0, "lost"

    # 좌우 둘 다 있으면 lane center 사용
    if left_mean is not None and right_mean is not None:
        lane_center = (left_mean + right_mean) * 0.5
        err = ((_LANE_MID_X - lane_center) / 0.5) * 10.0
        err = float(np.clip(err, -10.0, 10.0))
        return err, "ok"

    # 왼쪽만
    if left_mean is not None:
        err = ((_LANE_MID_X - left_mean) / 0.5) * 10.0
        err = float(np.clip(err, -10.0, 10.0))
        return err, "left_only"

    # 오른쪽만
    err = ((_LANE_MID_X - right_mean) / 0.5) * 10.0
    err = float(np.clip(err, -10.0, 10.0))
    return err, "right_only"


def _raw_classify(shapes):
    """
    현재 프레임 기준 특수상황 분류.

    규칙 요약
    - straight 판단용 lane은 상/중/하 모두 사용한다. (angle 계산은 compute_lane_error 담당)
    - left_t / down_t는 가까운 구간 우선이므로 mid/bottom을 더 강하게 본다.
    - cross는 eeu가 아니라 curve 쌍 기반 직진 상황으로 본다.
    - 현재 프레임만 사용한다.

    반환:
      - "left_t"
      - "down_t"
      - "cross"
      - None
    """

    # ---------- straight 참고용 lane 존재 ----------
    has_line_left = _has_class(shapes, "line", x="left")
    right_line_any = _has_class(shapes, "line", x="right")

    # ---------- left_t ----------
    # 1) 왼쪽 하단 curve 가 보이면 좌회전
    left_bottom_curve = _has_class(shapes, "curve", x="left", y="bottom")

    # 2) curve처럼 보여도 실제 검출은 eeu + right line 으로 나올 수 있음
    #    단, 상단은 너무 멀어서 제외하고 mid/bottom만 사용
    mid_eeu = _has_class(shapes, "eeu", y="mid")
    bottom_eeu = _has_class(shapes, "eeu", y="bottom")

    # ---------- down_t ----------
    # 하단에 eeu가 길게 깔린 상황
    # => bottom에서 eeu가 좌/중/우 중 2영역 이상 차지하면 wide로 해석
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

    # ---------- cross ----------
    # curve 기반 직진 상황
    # top은 너무 멀어서 제외하고 mid/bottom만 사용
    curve_mid_left = _has_class(shapes, "curve", x="left", y="mid")
    curve_mid_right = _has_class(shapes, "curve", x="right", y="mid")
    curve_bottom_left = _has_class(shapes, "curve", x="left", y="bottom")
    curve_bottom_right = _has_class(shapes, "curve", x="right", y="bottom")

    cross_cond = (curve_mid_left and curve_mid_right) or (
        curve_bottom_left and curve_bottom_right
    )

    # ---------- 우선순위 ----------
    # 1) 하단 curve -> left_t
    if left_bottom_curve:
        return "left_t"

    # 2) 하단 eeu wide -> down_t
    if eeu_bottom_wide:
        return "down_t"

    # 3) cross는 curve pair
    #    단, eeu 기반 down_t와 겹치면 down_t 우선
    if cross_cond and not eeu_bottom_wide:
        return "cross"

    # 4) mid/bottom eeu + right line 기반 left_t
    #    상단 제외, 중단/하단 우선
    if (mid_eeu or bottom_eeu) and right_line_any and not has_line_left:
        return "left_t"

    return None


def _resolve_cross_down(shapes):
    raw = _raw_classify(shapes)
    if raw in ("left_t", "down_t", "cross"):
        return raw
    return None


class IntersectionFSM:
    """
    현재 프레임 기반의 얇은 분류기.
    기존 인터페이스(update 반환형)는 유지.
    """

    def __init__(self, *args, **kwargs):
        self.cur = None

    def update(self, shapes):
        raw = _raw_classify(shapes)
        resolved = _resolve_cross_down(shapes)

        self.cur = resolved
        if resolved is None:
            return (False, None), raw
        return (True, resolved), raw


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════
def _draw_frame(d: dict) -> np.ndarray:
    ## 이부분도 수정. 줄어들었다.
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


# 객체 클래스명 -> 출력 ID 맵핑 (0~10 체계)
# KNU, 는 10으로 출력 box는 아직 미정
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

    반환: 패킷 리스트
      예) [[1, 0.123], [4, None], [3, None]]
    """
    err = round(le, 4)
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
class FrameGrabber(threading.Thread):
    """카메라 캡처 전담 - 항상 최신 프레임 유지. 영상 끝나면 eof=True"""

    def __init__(self, cap, is_file=False):
        super().__init__(daemon=True)
        self.cap = cap
        self.is_file = is_file
        self._f = None
        self._lk = threading.Lock()
        self._q = queue.Queue(maxsize=8) if is_file else None
        self.alive = True
        self.eof = False

    def run(self):
        while self.alive:
            if self.is_file:
                ok, f = self.cap.read()
                if ok:
                    try:
                        self._q.put(f, timeout=2.0)
                    except queue.Full:
                        pass
                else:
                    self.eof = True
                    break
            else:
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

    is_file = Path(str(source)).exists()
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
## 조금 줄었다.
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

    is_file = Path(str(source)).exists()

    if not is_file:
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, fps)
    else:
        cap = cv2.VideoCapture(str(source))

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    save_fps = src_fps if is_file and src_fps > 0 else fps
    print(
        f"[INFO] 해상도: {orig_w}x{orig_h}  obs_skip={obs_skip}  disp_scale={disp_scale}"
    )
    print(f"[INFO] 저장 FPS: {save_fps:.2f}  (원본: {src_fps:.2f})\n")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0

    grabber = FrameGrabber(cap, is_file=is_file)
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
    최종 lane class 체계
      0  = 차선 없음
      1  = 일반 차선
      6  = left_t
      8  = down_t
      9  = cross
      10 = 물류 pass 계열 (현재 lane에서는 미사용)
    """
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


def convert_object_result(obs_list):
    """
    파일 객체 결과를 최종 obj class 체계로 변환
      2  = SL
      3  = person
      4  = car
      5  = parking
      10 = KNU / box / 물류 pass 계열
      없음 = None
    """
    if not obs_list:
        return None

    name = obs_list[0]["class_name"]
    obj_map = {
        "SL": 2,
        "person": 3,
        "car": 4,
        "parking": 5,
        "KNU": 10,
        "box": 10,
        # "pass": 10,
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
        cam_w=320,
        cam_h=320,
    ):
        self.coral = coral
        self.source = source  ## 츠가
        self.fsm = IntersectionFSM()  ## 추가

        ## 추가 여기부터
        if use_edgetpu and EDGETPU_AVAILABLE:
            tpus = list_edge_tpus()
            print(f"[INFO] 연결된 EdgeTPU 장치: {len(tpus)}개")
            for i, t in enumerate(tpus):
                print(f"  [{i}] type={t.get('type', '?')}  path={t.get('path', '?')}")
        ## 추가 여기까지

        if coral == 2 and use_edgetpu:
            print("[INFO] lane 모델 -> usb:0 로드")
            self.lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=True, device="usb:0")
            time.sleep(1.0)
            print("[INFO] obs 모델 -> usb:1 로드")
            self.obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=True, device="usb:1")
        else:
            self.lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=use_edgetpu)
            self.obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=use_edgetpu)

        # self.cap = cv2.VideoCapture(source)
        ## ========================
        ## 실시간 카메라 지연 줄이려면 보통 V4L2 지정한 쪽이 더 나음
        ## ========================
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"source open failed: {source}")

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # self.fsm = IntersectionFSM() ## 주석 처리

    ## 추가
    def _read_latest_frame(self):
        if self.cap is None:
            return None

        if isinstance(self.source, int):
            ok = self.cap.grab()
            if not ok:
                return None
            ok, frame = self.cap.retrieve()
            if not ok:
                return None
            return frame

        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def step(self) -> Optional[DualInferenceResult]:
        # ok, frame = self.cap.read()
        # if not ok:
        #     return None

        frame = self._read_latest_frame()
        if frame is None:
            return None

        H, W = frame.shape[:2]

        lane_outs, _ = self.lane_eng.infer(frame)
        lane_shapes = parse_lane(lane_outs, H, W)
        p_le, p_ls = compute_lane_error(lane_shapes)
        (p_is, p_it), _ = self.fsm.update(lane_shapes)

        # obs_outs, _ = self.obs_eng.infer(frame)
        obs_outs, obs_ms = self.obs_eng.infer(frame)
        obs_list = parse_obstacle(obs_outs, H, W)

        line_id, angle = convert_lane_result(p_le, p_ls, p_is, p_it)
        obj_id = convert_object_result(obs_list)

        # print(
        #     f"[AI] line_id={line_id} angle={angle} obj_id={obj_id} "
        #     f"lane_status={p_ls} inter={p_it if p_is else None} "
        #     f"lane={lane_ms:.0f}ms(pre={self.lane_eng.pre_ms:.1f}/inv={self.lane_eng.invoke_ms:.1f}) "
        #     f"obs={obs_ms:.0f}ms raw={raw}"
        # )

        return DualInferenceResult(
            line_id=line_id,
            angle=angle,
            obj_id=obj_id,
            lane_status=p_ls,
            inter_type=p_it if p_is else None,
        )

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


###===============================================================
## 팀장코드~ 종료지점~
###===============================================================

# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
## 기존 코드와 바뀜(add_argument쪽이 좀 더 줄어들었다.)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
FPS/지연 줄이기 추천:
  --coral 2
  --cam-w 320 --cam-h 240
  DRAW_LANE_MASK = False
        """,
    )
    ap.add_argument("--lane-model", default="best_int8_edgetpu.tflite")
    ap.add_argument("--obs-model", default="obs_int8_edgetpu.tflite")
    ap.add_argument("--source", default="0")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--coral", type=int, default=1, choices=[1, 2])
    ap.add_argument("--obs-skip", type=int, default=2)
    ap.add_argument("--no-edgetpu", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--disp-scale", type=float, default=1.0)
    ap.add_argument("--cam-w", type=int, default=640)
    ap.add_argument("--cam-h", type=int, default=480)
    args = ap.parse_args()

    src = int(args.source) if str(args.source).isdigit() else args.source
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
