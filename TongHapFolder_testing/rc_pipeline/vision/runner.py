import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

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

    방향:
    - 카메라에서 raw frame 획득
    - lane / obs 입력을 각각 320x320으로 생성
    - 디버깅용 raw / lane_input / obs_input 저장
    - parse에는 원본 해상도(raw_h, raw_w)를 그대로 넘김
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
        model_input_size=(320, 320),
        save_debug_frames=False,
        debug_dir="debug_frames",
        debug_save_interval=10,
        max_debug_saves=20,
        lane_use_rgb=False,
        obs_use_rgb=False,
    ):
        self.coral = coral
        self.use_rpicam = False

        self.model_input_w = int(model_input_size[0])
        self.model_input_h = int(model_input_size[1])

        self.save_debug_frames = save_debug_frames
        self.debug_dir = Path(debug_dir)
        self.debug_save_interval = max(1, int(debug_save_interval))
        self.max_debug_saves = max(0, int(max_debug_saves))
        self.saved_debug_count = 0
        self.debug_counter = 0

        self.lane_use_rgb = lane_use_rgb
        self.obs_use_rgb = obs_use_rgb

        self._printed_input_info = False

        if self.save_debug_frames:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "raw").mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "lane_input").mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "obs_input").mkdir(parents=True, exist_ok=True)

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

    def _resize_for_model(self, frame):
        return cv2.resize(
            frame,
            (self.model_input_w, self.model_input_h),
            interpolation=cv2.INTER_LINEAR,
        )

    def _prepare_lane_input(self, raw_frame):
        x = self._resize_for_model(raw_frame)
        if self.lane_use_rgb:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x

    def _prepare_obs_input(self, raw_frame):
        x = self._resize_for_model(raw_frame)
        if self.obs_use_rgb:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x

    def _save_debug_images(self, frame_id, raw_frame, lane_input, obs_input):
        if not self.save_debug_frames:
            return

        if self.saved_debug_count >= self.max_debug_saves:
            return

        self.debug_counter += 1
        if self.debug_counter % self.debug_save_interval != 0:
            return

        ts_ms = int(time.time() * 1000)
        fid = frame_id if frame_id is not None else self.debug_counter

        raw_path = self.debug_dir / "raw" / f"frame_{fid}_{ts_ms}.jpg"
        lane_path = self.debug_dir / "lane_input" / f"frame_{fid}_{ts_ms}.jpg"
        obs_path = self.debug_dir / "obs_input" / f"frame_{fid}_{ts_ms}.jpg"

        # 저장은 보기 쉽게 BGR로 맞춤
        lane_save = lane_input
        obs_save = obs_input

        if self.lane_use_rgb:
            lane_save = cv2.cvtColor(lane_input, cv2.COLOR_RGB2BGR)
        if self.obs_use_rgb:
            obs_save = cv2.cvtColor(obs_input, cv2.COLOR_RGB2BGR)

        ok_raw = cv2.imwrite(str(raw_path), raw_frame)
        ok_lane = cv2.imwrite(str(lane_path), lane_save)
        ok_obs = cv2.imwrite(str(obs_path), obs_save)

        if ok_raw and ok_lane and ok_obs:
            self.saved_debug_count += 1
            print(
                f"[DEBUG_SAVE] saved set {self.saved_debug_count}/{self.max_debug_saves} "
                f"(frame_id={fid})"
            )
            if self.saved_debug_count >= self.max_debug_saves:
                print(
                    "[DEBUG_SAVE] reached max debug saves, no more images will be saved."
                )
        else:
            print(
                f"[DEBUG_SAVE] save failed "
                f"(raw={ok_raw}, lane={ok_lane}, obs={ok_obs}, frame_id={fid})"
            )

    def step(self) -> Optional[DualInferenceResult]:
        step_start_mono = time.monotonic()

        if self.use_rpicam:
            ok, cam_data = self.cam.read(wait_timeout=2.0)
            if not ok:
                return None

            raw_frame = cam_data["frame"]
            frame_id = cam_data["frame_id"]
            rx_done_mono = cam_data["rx_done_mono"]
        else:
            ok, raw_frame = self.cap.read()
            if not ok:
                return None

            frame_id = None
            rx_done_mono = None

        raw_h, raw_w = raw_frame.shape[:2]

        lane_input = self._prepare_lane_input(raw_frame)
        obs_input = self._prepare_obs_input(raw_frame)

        if not self._printed_input_info:
            print(
                f"[RUNNER] raw={raw_frame.shape}, "
                f"lane_input={lane_input.shape}, obs_input={obs_input.shape}, "
                f"lane_rgb={self.lane_use_rgb}, obs_rgb={self.obs_use_rgb}"
            )
            self._printed_input_info = True

        self._save_debug_images(
            frame_id=frame_id,
            raw_frame=raw_frame,
            lane_input=lane_input,
            obs_input=obs_input,
        )

        lane_outs, lane_ms = self.lane_eng.infer(lane_input)
        obs_outs, obs_ms = self.obs_eng.infer(obs_input)

        # parse에는 원본 해상도 유지
        lane_shapes = parse_lane(lane_outs, raw_h, raw_w)
        obs_list = parse_obstacle(obs_outs, raw_h, raw_w)

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
