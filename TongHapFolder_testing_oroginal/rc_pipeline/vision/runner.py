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
    기존에 잘 되던 dual_model_edgetpu_v6_origin.py 경로에 맞춘 러너

    핵심:
    - runner에서 미리 320x320 resize 하지 않음
    - 원본 frame 그대로 infer()에 넣음
    - resize / RGB 변환 / quantize는 EdgeTPUEngine.preprocess() 내부에 맡김
    - parse는 원본 해상도(H, W) 기준으로 수행
    - 디버깅용 저장은 raw / lane_view / obs_view 20세트만 저장
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
        save_debug_frames=False,
        debug_dir="debug_frames",
        debug_save_interval=10,
        max_debug_saves=20,
    ):
        self.coral = coral
        self.use_rpicam = False

        self.save_debug_frames = save_debug_frames
        self.debug_dir = Path(debug_dir)
        self.debug_save_interval = max(1, int(debug_save_interval))
        self.max_debug_saves = max(0, int(max_debug_saves))
        self.saved_debug_count = 0
        self.debug_counter = 0

        if self.save_debug_frames:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "raw").mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "lane_view").mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "obs_view").mkdir(parents=True, exist_ok=True)

        if coral == 2 and use_edgetpu:
            print("[INFO] lane 모델 -> usb:0 로드")
            self.lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=True, device="usb:0")
            time.sleep(1.0)
            print("[INFO] obs 모델 -> usb:1 로드")
            self.obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=True, device="usb:1")
        else:
            self.lane_eng = EdgeTPUEngine(lane_model, use_edgetpu=use_edgetpu)
            self.obs_eng = EdgeTPUEngine(obs_model, use_edgetpu=use_edgetpu)

        if isinstance(source, int):
            print("[INFO] camera source detected -> use rpicam-vid MJPEG bridge")
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
        self._printed_input_info = False

    def _make_engine_view(self, engine, frame):
        """
        디버깅용: 엔진 preprocess와 최대한 같은 화면을 사람이 볼 수 있게 만든다.
        실제 infer 입력은 여기서 만든 걸 쓰지 않고, raw frame을 그대로 infer()에 넘긴다.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(
            rgb,
            (engine.img_w, engine.img_h),
            interpolation=cv2.INTER_NEAREST,
        )
        # 저장용으로 다시 BGR 변환
        view = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        return view

    def _save_debug_images(self, frame_id, raw_frame, lane_view, obs_view):
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
        lane_path = self.debug_dir / "lane_view" / f"frame_{fid}_{ts_ms}.jpg"
        obs_path = self.debug_dir / "obs_view" / f"frame_{fid}_{ts_ms}.jpg"

        ok_raw = cv2.imwrite(str(raw_path), raw_frame)
        ok_lane = cv2.imwrite(str(lane_path), lane_view)
        ok_obs = cv2.imwrite(str(obs_path), obs_view)

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

            frame = cam_data["frame"]
            frame_id = cam_data["frame_id"]
            rx_done_mono = cam_data["rx_done_mono"]
        else:
            ok, frame = self.cap.read()
            if not ok:
                return None

            frame_id = None
            rx_done_mono = None

        H, W = frame.shape[:2]

        if not self._printed_input_info:
            print(
                f"[RUNNER] raw frame shape={frame.shape}, "
                f"lane_model_input=({self.lane_eng.img_w}x{self.lane_eng.img_h}), "
                f"obs_model_input=({self.obs_eng.img_w}x{self.obs_eng.img_h})"
            )
            self._printed_input_info = True

        # 디버깅 저장용 뷰 생성
        lane_view = self._make_engine_view(self.lane_eng, frame)
        obs_view = self._make_engine_view(self.obs_eng, frame)

        self._save_debug_images(
            frame_id=frame_id,
            raw_frame=frame,
            lane_view=lane_view,
            obs_view=obs_view,
        )

        # 핵심: 원본 frame 그대로 infer()
        lane_outs, lane_ms = self.lane_eng.infer(frame)
        obs_outs, obs_ms = self.obs_eng.infer(frame)

        # 핵심: parse는 원본 해상도 기준
        lane_shapes = parse_lane(lane_outs, H, W)
        obs_list = parse_obstacle(obs_outs, H, W)

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
