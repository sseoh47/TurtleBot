import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vision.camera import RPiMJPEGCamera
from vision.postprocess import convert_lane_result, convert_object_result
from dual_model_edgetpu_v6_origin import EdgeTPUEngine
from vision.lane_postprocess import (
    IntersectionFSM,
    compute_lane_error,
    parse_lane,
    parse_obstacle,
)
from vision.postprocess import convert_lane_result, convert_object_result


CV_LANE_PROC_SIZE = 320
CV_LANE_ROI_TOP_RATIO = 0.6
CV_LANE_THRESHOLD = 200
CV_LANE_ROW_HALF_HEIGHT = 3
CV_LANE_MIN_SIDE_PIXELS = 8
CV_LANE_ROW_RATIOS = (0.75, 0.50, 0.25)
CV_LANE_SIDE_MIN_PIXELS = 60
CV_LANE_MAX_STEER = 2.0
CV_LANE_SEARCH_STEER = 2.0
CV_LANE_SLOPE_SUM_LIMIT = 0.30
CV_CENTER_BLACK_Y_START_RATIO = 0.60
CV_CENTER_BLACK_Y_END_RATIO = 0.70
CV_CENTER_BLACK_HALF_WIDTH_RATIO = 0.08
CV_CENTER_TAPE_R_MAX = 80
CV_CENTER_TAPE_G_MIN = 150
CV_CENTER_TAPE_G_MAX = 200
CV_CENTER_TAPE_B_MIN = 150
CV_CENTER_TAPE_MIN_PIXELS = 1


def _resize_for_cv_lane(frame):
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((CV_LANE_PROC_SIZE, CV_LANE_PROC_SIZE, 3), dtype=np.uint8)

    scale = min(CV_LANE_PROC_SIZE / float(width), CV_LANE_PROC_SIZE / float(height))
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (out_width, out_height), interpolation=interp)

    canvas = np.zeros((CV_LANE_PROC_SIZE, CV_LANE_PROC_SIZE, 3), dtype=np.uint8)
    offset_x = (CV_LANE_PROC_SIZE - out_width) // 2
    offset_y = (CV_LANE_PROC_SIZE - out_height) // 2
    canvas[offset_y:offset_y + out_height, offset_x:offset_x + out_width] = resized
    return canvas


def _extract_cv_lane_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, CV_LANE_THRESHOLD, 255, cv2.THRESH_BINARY)

    top = int(mask.shape[0] * CV_LANE_ROI_TOP_RATIO)
    mask[:top, :] = 0
    return mask


def _scan_cv_lane_row(mask, row):
    height, width = mask.shape
    band = mask[max(0, row - CV_LANE_ROW_HALF_HEIGHT):min(height, row + CV_LANE_ROW_HALF_HEIGHT + 1), :]
    _, xs = np.nonzero(band)

    if xs.size == 0:
        return None, None

    mid = width // 2
    left_xs = xs[xs < mid]
    right_xs = xs[xs >= mid]

    left = float(left_xs.mean()) if left_xs.size >= CV_LANE_MIN_SIDE_PIXELS else None
    right = float(right_xs.mean()) if right_xs.size >= CV_LANE_MIN_SIDE_PIXELS else None
    return left, right


def _fit_cv_lane_slope(points):
    if len(points) < 2:
        return None

    ys = np.array([point[0] for point in points], dtype=np.float32)
    xs = np.array([point[1] for point in points], dtype=np.float32)
    slope, _ = np.polyfit(ys, xs, 1)
    return float(slope)


def _normalize_cv_lane_steer(value):
    if CV_LANE_SLOPE_SUM_LIMIT <= 0:
        return 0.0

    steer = (value / CV_LANE_SLOPE_SUM_LIMIT) * CV_LANE_MAX_STEER
    return float(np.clip(steer, -CV_LANE_MAX_STEER, CV_LANE_MAX_STEER))


def detect_cv_center_black(frame):
    if frame is None or frame.size == 0:
        return False

    lane_frame = _resize_for_cv_lane(frame)
    height, width = lane_frame.shape[:2]

    y1 = int(height * CV_CENTER_BLACK_Y_START_RATIO)
    y2 = int(height * CV_CENTER_BLACK_Y_END_RATIO)
    mid = width // 2
    half_w = max(1, int(width * CV_CENTER_BLACK_HALF_WIDTH_RATIO))
    x1 = max(0, mid - half_w)
    x2 = min(width, mid + half_w)

    center_roi = lane_frame[y1:y2, x1:x2]
    if center_roi.size == 0:
        return False

    r = center_roi[:, :, 0]
    g = center_roi[:, :, 1]
    b = center_roi[:, :, 2]
    in_range = (
        (r <= CV_CENTER_TAPE_R_MAX)
        & (g >= CV_CENTER_TAPE_G_MIN)
        & (g <= CV_CENTER_TAPE_G_MAX)
        & (b >= CV_CENTER_TAPE_B_MIN)
    )
    return int(np.count_nonzero(in_range)) >= CV_CENTER_TAPE_MIN_PIXELS


def compute_cv_lane_angle(frame, fallback_angle):
    if frame is None or frame.size == 0:
        return float(fallback_angle)

    lane_frame = _resize_for_cv_lane(frame)
    mask = _extract_cv_lane_mask(lane_frame)

    height, width = mask.shape
    roi_top = int(height * CV_LANE_ROI_TOP_RATIO)
    roi_height = max(1, height - roi_top)
    left_points = []
    right_points = []

    for ratio in CV_LANE_ROW_RATIOS:
        row = roi_top + int(roi_height * ratio)
        row = min(height - 1, max(roi_top, row))
        left, right = _scan_cv_lane_row(mask, row)

        if left is not None:
            left_points.append((float(row), left))
        if right is not None:
            right_points.append((float(row), right))

    mid = width // 2
    left_count = cv2.countNonZero(mask[:, :mid])
    right_count = cv2.countNonZero(mask[:, mid:])

    left_detected = left_count >= CV_LANE_SIDE_MIN_PIXELS
    right_detected = right_count >= CV_LANE_SIDE_MIN_PIXELS

    if left_detected and not right_detected:
        return CV_LANE_SEARCH_STEER
    if right_detected and not left_detected:
        return -CV_LANE_SEARCH_STEER
    if not left_detected and not right_detected:
        return 0.0

    if len(left_points) >= 2 and len(right_points) >= 2:
        left_slope = _fit_cv_lane_slope(left_points)
        right_slope = _fit_cv_lane_slope(right_points)

        if left_slope is not None and right_slope is not None:
            return _normalize_cv_lane_steer(left_slope + right_slope)

    return float(fallback_angle)


@dataclass
class DualInferenceResult:
    line_id: int
    angle: Optional[float]
    obj_id: Optional[int]
    lane_status: str
    inter_type: Optional[str]
    center_tape: bool = False
    frame_id: Optional[int] = None
    frame_age_start: Optional[float] = None
    frame_age_end: Optional[float] = None
    step_dt: Optional[float] = None
    lane_ms: Optional[float] = None
    obs_ms: Optional[float] = None


class InferWorker(threading.Thread):
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
        self.cv_lane_last_angle = 0.0

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

        if isinstance(source, int):
            print("[INFO] camera source detected -> use rpicam-vid MJPEG bridge")
            self.cam = RPiMJPEGCamera(width=cam_w, height=cam_h, framerate=cam_fps)
            self.use_rpicam = True
        else:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"source open failed: {source}")

        self.fsm = IntersectionFSM()
        self._printed_input_info = True

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
            rgb, (engine.img_w, engine.img_h), interpolation=cv2.INTER_NEAREST
        )
        return cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

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
                f"[DEBUG_SAVE] saved set {self.saved_debug_count}/{self.max_debug_saves} (frame_id={fid})"
            )
        else:
            print(
                f"[DEBUG_SAVE] save failed (raw={ok_raw}, lane={ok_lane}, obs={ok_obs}, frame_id={fid})"
            )

    def _infer_serial(self, frame, INF_H, INF_W):
        lane_outs, lane_ms = self.lane_eng.infer(frame)
        lane_shapes = parse_lane(lane_outs, INF_H, INF_W)

        if self.step_count % self.obs_interval == 0:
            obs_outs, obs_ms = self.obs_eng.infer(frame)
            obs_list = parse_obstacle(obs_outs, INF_H, INF_W)
            self.prev_obs_list = obs_list
            self.prev_obs_ms = obs_ms
        else:
            obs_list = self.prev_obs_list
            obs_ms = self.prev_obs_ms

        return lane_shapes, lane_ms, obs_list, obs_ms

    def _infer_parallel(self, frame_id, frame, INF_H, INF_W):
        self.lane_worker.push(frame_id, frame)
        run_obs = self.step_count % self.obs_interval == 0
        if run_obs:
            self.obs_worker.push(frame_id, frame)

        lane_outs, lane_ms = self.lane_worker.get_result_for(frame_id, timeout=2.0)
        # print("\n=== LANE RAW OUTPUT ===")
        # if lane_outs is None:
        #     print("lane_outs is None")
        # else:
        #     for i, o in enumerate(lane_outs):
        #         print(f"out[{i}] shape={o.shape}, min={o.min()}, max={o.max()}")

        if lane_outs is None:
            raise RuntimeError("lane worker timeout")
        lane_shapes = parse_lane(lane_outs, INF_H, INF_W)

        if run_obs:
            obs_outs, obs_ms = self.obs_worker.get_result_for(frame_id, timeout=2.0)
            if obs_outs is None:
                raise RuntimeError("obs worker timeout")
            obs_list = parse_obstacle(obs_outs, INF_H, INF_W)
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

        INF_H = self.lane_eng.img_h
        INF_W = self.lane_eng.img_w

        if not self._printed_input_info:
            print(
                f"[RUNNER] raw frame shape={frame.shape}, lane_model_input=({self.lane_eng.img_w}x{self.lane_eng.img_h}), obs_model_input=({self.obs_eng.img_w}x{self.obs_eng.img_h}), obs_interval={self.obs_interval}, parallel_mode={self.parallel_mode}"
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
                frame_id, frame, INF_H, INF_W
            )
        else:
            lane_shapes, lane_ms, obs_list, obs_ms = self._infer_serial(
                frame, INF_H, INF_W
            )

        p_le, p_ls = compute_lane_error(lane_shapes)
        (p_is, p_it), _ = self.fsm.update(lane_shapes)

        line_id, angle = convert_lane_result(p_le, p_ls, p_is, p_it)
        center_tape = detect_cv_center_black(frame)
        if center_tape:
            line_id = 7
            angle = 0.0
        elif line_id == 1:
            angle = compute_cv_lane_angle(frame, self.cv_lane_last_angle)
            self.cv_lane_last_angle = angle
        obj_id = convert_object_result(obs_list)

        # print(
        #     f"[RUNNER DBG] frame_id={frame_id}, lane_shapes={len(lane_shapes)}, obs={len(obs_list)}, p_le={p_le:.2f}, p_ls={p_ls}, p_is={p_is}, p_it={p_it}, line_id={line_id}, angle={angle}, obj_id={obj_id}"
        # )

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
            center_tape=center_tape,
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
                self.lane_worker.join(timeout=1.0)
            if self.obs_worker is not None:
                self.obs_worker.stop()
                self.obs_worker.join(timeout=1.0)
        except Exception:
            pass

        try:
            if self.use_rpicam:
                self.cam.release()
            else:
                self.cap.release()
        except Exception:
            pass
