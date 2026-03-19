import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import cv2

from vision.camera import RPiMJPEGCamera
from vision.postprocess import convert_lane_result, convert_object_result

from dual_model_edgetpu_v6_origin import (
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

    기능:
    - PiCamera MJPEG 프레임 수신
    - 모델 입력 320x320 resize
    - 디버깅용 raw / lane_input / obs_input 이미지 저장
    - 저장은 세트 기준 최대 max_debug_saves번만 수행
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
    ):
        self.coral = coral
        self.use_rpicam = False

        self.model_input_w = model_input_size[0]
        self.model_input_h = model_input_size[1]

        self.save_debug_frames = save_debug_frames
        self.debug_dir = Path(debug_dir)
        self.debug_save_interval = max(1, int(debug_save_interval))
        self.debug_counter = 0

        # 저장 세트 수 제한
        self.max_debug_saves = max(0, int(max_debug_saves))
        self.saved_debug_count = 0

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

    def _build_model_input(self, frame):
        """
        모델에 실제 넣을 320x320 입력 이미지 생성
        현재는 단순 resize만 수행
        """
        model_input = cv2.resize(
            frame,
            (self.model_input_w, self.model_input_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return model_input

    def _save_debug_images(self, frame_id, raw_frame, lane_input, obs_input):
        """
        디버깅용 이미지 저장
        - raw: 카메라 원본
        - lane_input: lane 모델에 넣은 320x320 이미지
        - obs_input: obstacle 모델에 넣은 320x320 이미지

        저장은 debug_save_interval 주기마다 수행하며,
        최대 max_debug_saves 세트까지만 저장한다.
        """
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

        ok_raw = cv2.imwrite(str(raw_path), raw_frame)
        ok_lane = cv2.imwrite(str(lane_path), lane_input)
        ok_obs = cv2.imwrite(str(obs_path), obs_input)

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

        # 모델 입력 320x320 생성
        lane_input = self._build_model_input(raw_frame)
        obs_input = lane_input.copy()

        # 디버깅 이미지 저장
        self._save_debug_images(
            frame_id=frame_id,
            raw_frame=raw_frame,
            lane_input=lane_input,
            obs_input=obs_input,
        )

        # 실제 추론
        lane_outs, lane_ms = self.lane_eng.infer(lane_input)
        obs_outs, obs_ms = self.obs_eng.infer(obs_input)

        # 후처리는 현재 모델 입력 크기(320x320) 기준으로 진행
        in_h, in_w = lane_input.shape[:2]

        lane_shapes = parse_lane(lane_outs, in_h, in_w)
        obs_list = parse_obstacle(obs_outs, in_h, in_w)

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
