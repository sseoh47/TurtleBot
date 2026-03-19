## 병렬화적용완
import time
import threading
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


class InferWorker(threading.Thread):
    """
    EdgeTPU 추론 전용 워커
    push(frame_id, frame) 하면 가장 최신 요청만 유지하고 infer 수행
    get_result_for(frame_id)로 해당 프레임 결과를 기다려서 가져온다.
    """

    def __init__(self, engine, name="infer-worker"):
        super().__init__(daemon=True, name=name)
        self.engine = engine
        self.alive = True

        self.req_lock = threading.Lock()
        self.req_cv = threading.Condition(self.req_lock)
        self.pending_frame = None
        self.pending_frame_id = None
        self.has_request = False

        self.res_lock = threading.Lock()
        self.res_cv = threading.Condition(self.res_lock)
        self.last_result_frame_id = None
        self.last_outs = None
        self.last_ms = None

    def push(self, frame_id, frame):
        with self.req_cv:
            self.pending_frame_id = frame_id
            self.pending_frame = frame
            self.has_request = True
            self.req_cv.notify()

    def get_result_for(self, frame_id, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self.res_cv:
            while self.alive:
                if self.last_result_frame_id == frame_id:
                    return self.last_outs, self.last_ms

                remain = deadline - time.monotonic()
                if remain <= 0:
                    return None, None
                self.res_cv.wait(timeout=remain)

        return None, None

    def run(self):
        while self.alive:
            with self.req_cv:
                while self.alive and not self.has_request:
                    self.req_cv.wait(timeout=0.5)

                if not self.alive:
                    break

                frame_id = self.pending_frame_id
                frame = self.pending_frame
                self.has_request = False

            if frame is None:
                continue

            outs, ms = self.engine.infer(frame)

            with self.res_cv:
                self.last_result_frame_id = frame_id
                self.last_outs = outs
                self.last_ms = ms
                self.res_cv.notify_all()

    def stop(self):
        self.alive = False
        with self.req_cv:
            self.req_cv.notify_all()
        with self.res_cv:
            self.res_cv.notify_all()


class DualModelRunner:
    """
    기존에 잘 되던 dual_model_edgetpu_v6_origin.py 경로에 맞춘 러너

    핵심:
    - runner에서 미리 320x320 resize 하지 않음
    - 원본 frame 그대로 infer()에 넣음
    - resize / RGB 변환 / quantize는 EdgeTPUEngine.preprocess() 내부에 맡김
    - parse는 원본 해상도(H, W) 기준으로 수행
    - coral 2개면 lane/obs 추론을 내부 worker thread로 병렬 수행
    - obs_interval 적용 가능
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
        obs_interval=2,
    ):
        self.coral = coral
        self.use_edgetpu = use_edgetpu
        self.use_rpicam = False

        self.save_debug_frames = save_debug_frames
        self.debug_dir = Path(debug_dir)
        self.debug_save_interval = max(1, int(debug_save_interval))
        self.max_debug_saves = max(0, int(max_debug_saves))
        self.saved_debug_count = 0
        self.debug_counter = 0

        self.obs_interval = max(1, int(obs_interval))
        self.step_count = 0
        self.prev_obs_list = []
        self.prev_obs_ms = 0.0

        self.parallel_mode = bool(coral == 2 and use_edgetpu)

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
        print(f"[RUNNER_CAM_CFG] cam_w={cam_w}, cam_h={cam_h}, cam_fps={cam_fps}")
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

        self.lane_worker = None
        self.obs_worker = None

        if self.parallel_mode:
            self.lane_worker = InferWorker(self.lane_eng, name="lane-worker")
            self.obs_worker = InferWorker(self.obs_eng, name="obs-worker")
            self.lane_worker.start()
            self.obs_worker.start()
            print("[INFO] parallel TPU workers started")

    def _make_engine_view(self, engine, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(
            rgb,
            (engine.img_w, engine.img_h),
            interpolation=cv2.INTER_NEAREST,
        )
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

    def _infer_serial(self, frame, H, W):
        lane_outs, lane_ms = self.lane_eng.infer(frame)
        lane_shapes = parse_lane(lane_outs, H, W)

        if self.step_count % self.obs_interval == 0:
            obs_outs, obs_ms = self.obs_eng.infer(frame)
            obs_list = parse_obstacle(obs_outs, H, W)
            self.prev_obs_list = obs_list
            self.prev_obs_ms = obs_ms
        else:
            obs_list = self.prev_obs_list
            obs_ms = self.prev_obs_ms

        return lane_shapes, lane_ms, obs_list, obs_ms

    def _infer_parallel(self, frame_id, frame, H, W):
        # lane은 항상 수행
        self.lane_worker.push(frame_id, frame)

        # obs는 주기마다만 수행
        run_obs = self.step_count % self.obs_interval == 0
        if run_obs:
            self.obs_worker.push(frame_id, frame)

        lane_outs, lane_ms = self.lane_worker.get_result_for(frame_id, timeout=2.0)
        if lane_outs is None:
            raise RuntimeError("lane worker timeout")

        lane_shapes = parse_lane(lane_outs, H, W)

        if run_obs:
            obs_outs, obs_ms = self.obs_worker.get_result_for(frame_id, timeout=2.0)
            if obs_outs is None:
                raise RuntimeError("obs worker timeout")

            obs_list = parse_obstacle(obs_outs, H, W)
            self.prev_obs_list = obs_list
            self.prev_obs_ms = obs_ms
        else:
            obs_list = self.prev_obs_list
            obs_ms = self.prev_obs_ms

        return lane_shapes, lane_ms, obs_list, obs_ms

    def step(self) -> Optional[DualInferenceResult]:
        step_start_mono = time.monotonic()
        self.step_count += 1

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

            frame_id = self.step_count
            rx_done_mono = None

        H, W = frame.shape[:2]

        if not self._printed_input_info:
            print(
                f"[RUNNER] raw frame shape={frame.shape}, "
                f"lane_model_input=({self.lane_eng.img_w}x{self.lane_eng.img_h}), "
                f"obs_model_input=({self.obs_eng.img_w}x{self.obs_eng.img_h}), "
                f"obs_interval={self.obs_interval}, "
                f"parallel_mode={self.parallel_mode}"
            )
            self._printed_input_info = True

        if self.save_debug_frames:
            lane_view = self._make_engine_view(self.lane_eng, frame)
            obs_view = self._make_engine_view(self.obs_eng, frame)

            self._save_debug_images(
                frame_id=frame_id,
                raw_frame=frame,
                lane_view=lane_view,
                obs_view=obs_view,
            )

        if self.parallel_mode:
            lane_shapes, lane_ms, obs_list, obs_ms = self._infer_parallel(
                frame_id, frame, H, W
            )
        else:
            lane_shapes, lane_ms, obs_list, obs_ms = self._infer_serial(frame, H, W)

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
            if self.lane_worker is not None:
                self.lane_worker.stop()
            if self.obs_worker is not None:
                self.obs_worker.stop()
        except Exception:
            pass

        try:
            if self.use_rpicam:
                self.cam.release()
            else:
                self.cap.release()
        except Exception:
            pass
