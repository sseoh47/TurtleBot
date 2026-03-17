import time
from dataclasses import dataclass
from typing import Optional

import cv2

from vision.camera import RPiMJPEGCamera
from vision.postprocess import convert_lane_result, convert_object_result

from dual_model_edgetpu_v6 import (
    EdgeTPUEngine,
    parse_lane,
    parse_obstacle,
    compute_lane_error,
    IntersectionFSM,
)


@dataclass
class DualInferenceResult:
    line_id: int
    angle: Optional[float]
    obj_id: Optional[int]
    lane_status: str
    inter_type: Optional[str]
    frame_id: Optional[int] = None
    frame_age_start: Optional[float] = None
    frame_age_end: Optional[float] = None
    step_dt: Optional[float] = None
    lane_ms: Optional[float] = None
    obs_ms: Optional[float] = None


class DualModelRunner:
    """
    프로젝트 전용 듀얼 모델 래퍼
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
        cam_fps=10,
    ):
        self.coral = coral
        self.use_rpicam = False

        if coral == 2 and use_edgetpu:
            self.lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=True, device="usb:0")
            time.sleep(1.0)
            self.obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=True, device="usb:1")
        else:
            self.lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=use_edgetpu)
            self.obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=use_edgetpu)

        if isinstance(source, int):
            self.cam = RPiMJPEGCamera(
                width=cam_w,
                height=cam_h,
                framerate=cam_fps,
            )
            self.use_rpicam = True
        else:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"source open failed: {source}")

        self.fsm = IntersectionFSM()

    def step(self) -> Optional[DualInferenceResult]:
        step_start_mono = time.monotonic()

        if self.use_rpicam:
            ok, cam_data = self.cam.read(wait_timeout=2.0)
            if not ok:
                return None

            frame = cam_data["frame"]
            frame_id = cam_data["frame_id"]
            rx_done_mono = cam_data["rx_done_mono"]
        else:
            ok, frame = self.cap.read()
            if not ok:
                return None

            frame_id = None
            rx_done_mono = None

        h, w = frame.shape[:2]

        lane_outs, lane_ms = self.lane_eng.infer(frame)
        obs_outs, obs_ms = self.obs_eng.infer(frame)

        lane_shapes = parse_lane(lane_outs, h, w)
        obs_list = parse_obstacle(obs_outs, h, w)

        p_le, p_ls = compute_lane_error(lane_shapes)
        (p_is, p_it), _ = self.fsm.update(lane_shapes)

        line_id, angle = convert_lane_result(p_le, p_ls, p_is, p_it)
        obj_id = convert_object_result(obs_list)

        step_end_mono = time.monotonic()

        frame_age_start = None
        frame_age_end = None
        if rx_done_mono is not None:
            frame_age_start = step_start_mono - rx_done_mono
            frame_age_end = step_end_mono - rx_done_mono

        return DualInferenceResult(
            line_id=line_id,
            angle=angle,
            obj_id=obj_id,
            lane_status=p_ls,
            inter_type=p_it if p_is else None,
            frame_id=frame_id,
            frame_age_start=frame_age_start,
            frame_age_end=frame_age_end,
            step_dt=step_end_mono - step_start_mono,
            lane_ms=lane_ms,
            obs_ms=obs_ms,
        )

    def close(self):
        try:
            if self.use_rpicam:
                self.cam.release()
            else:
                self.cap.release()
        except Exception:
            pass
