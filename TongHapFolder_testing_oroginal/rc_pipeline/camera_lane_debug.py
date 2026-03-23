#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from config import CAM_FPS, CAM_H, CAM_W, CAMERA_SOURCE
from vision.camera import RPiMJPEGCamera


PROC_WIDTH = 320
DISPLAY_SIZE = 320
ROI_TOP_RATIO = 0.6
THRESHOLD_VALUE = 200
ROW_HALF_HEIGHT = 3
MIN_SIDE_PIXELS = 8
ROI_SAMPLE_RATIOS = (0.75, 0.50, 0.25)
ROI_SIDE_MIN_PIXELS = 60

MAX_STEER = 10.0
SEARCH_STEER = 5.0
SLOPE_SUM_LIMIT = 0.30
DEBUG_PRINT_INTERVAL = 0.5


def build_camera(source, width: int, height: int, fps: int):
    if isinstance(source, int):
        camera = RPiMJPEGCamera(width=width, height=height, framerate=fps)
        return "rpicam", camera

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"source open failed: {source}")
    return "opencv", cap


def read_frame(camera_kind: str, camera):
    if camera_kind == "rpicam":
        ok, cam_data = camera.read(wait_timeout=2.0)
        if not ok:
            return False, None
        return True, cam_data["frame"]

    ok, frame = camera.read()
    return ok, frame


def release_camera(camera_kind: str, camera) -> None:
    if camera_kind == "rpicam":
        camera.release()
    else:
        camera.release()


def resize_for_lane(frame_bgr: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return frame_bgr

    out_width = min(target_width, width)
    out_height = max(1, int(round(height * (out_width / float(width)))))
    interp = cv2.INTER_AREA if out_width < width else cv2.INTER_LINEAR
    return cv2.resize(frame_bgr, (out_width, out_height), interpolation=interp)


def extract_mask(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    top = int(mask.shape[0] * ROI_TOP_RATIO)
    mask[:top, :] = 0
    return mask


def scan_row(mask: np.ndarray, row: int) -> tuple[float | None, float | None]:
    height, width = mask.shape
    band = mask[max(0, row - ROW_HALF_HEIGHT):min(height, row + ROW_HALF_HEIGHT + 1), :]
    _, xs = np.nonzero(band)

    if xs.size == 0:
        return None, None

    mid = width // 2
    left_xs = xs[xs < mid]
    right_xs = xs[xs >= mid]

    left = float(left_xs.mean()) if left_xs.size >= MIN_SIDE_PIXELS else None
    right = float(right_xs.mean()) if right_xs.size >= MIN_SIDE_PIXELS else None
    return left, right


def fit_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None

    ys = np.array([point[0] for point in points], dtype=np.float32)
    xs = np.array([point[1] for point in points], dtype=np.float32)
    slope, _ = np.polyfit(ys, xs, 1)
    return float(slope)


def normalize_steer(value: float) -> float:
    if SLOPE_SUM_LIMIT <= 0:
        return 0.0
    steer = (value / SLOPE_SUM_LIMIT) * MAX_STEER
    return float(np.clip(steer, -MAX_STEER, MAX_STEER))


def collect_lane_samples(
    mask: np.ndarray,
) -> tuple[list[int], list[tuple[float, float]], list[tuple[float, float]], int, int]:
    height, width = mask.shape
    roi_top = int(height * ROI_TOP_RATIO)
    roi_height = max(1, height - roi_top)
    sample_rows: list[int] = []
    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []

    for ratio in ROI_SAMPLE_RATIOS:
        row = roi_top + int(roi_height * ratio)
        row = min(height - 1, max(roi_top, row))
        sample_rows.append(int(row))
        left, right = scan_row(mask, row)

        if left is not None:
            left_points.append((float(row), left))
        if right is not None:
            right_points.append((float(row), right))

    mid = width // 2
    left_count = cv2.countNonZero(mask[:, :mid])
    right_count = cv2.countNonZero(mask[:, mid:])
    return sample_rows, left_points, right_points, left_count, right_count


def compute_steer(
    mask: np.ndarray, last_steer: float
) -> tuple[float, int, int, list[int], list[tuple[float, float]], list[tuple[float, float]]]:
    sample_rows, left_points, right_points, left_count, right_count = collect_lane_samples(mask)

    left_detected = left_count >= ROI_SIDE_MIN_PIXELS
    right_detected = right_count >= ROI_SIDE_MIN_PIXELS

    if left_detected and not right_detected:
        return SEARCH_STEER, left_count, right_count, sample_rows, left_points, right_points
    if right_detected and not left_detected:
        return -SEARCH_STEER, left_count, right_count, sample_rows, left_points, right_points
    if not left_detected and not right_detected:
        return last_steer, left_count, right_count, sample_rows, left_points, right_points

    if len(left_points) >= 2 and len(right_points) >= 2:
        left_slope = fit_slope(left_points)
        right_slope = fit_slope(right_points)

        if left_slope is not None and right_slope is not None:
            return (
                normalize_steer(left_slope + right_slope),
                left_count,
                right_count,
                sample_rows,
                left_points,
                right_points,
            )

    return last_steer, left_count, right_count, sample_rows, left_points, right_points


def fit_to_square(frame_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = frame_bgr.shape[:2]
    if height <= 0 or width <= 0:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        return canvas, 1.0, 0, 0

    scale = min(size / float(width), size / float(height))
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_bgr, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    offset_x = (size - out_width) // 2
    offset_y = (size - out_height) // 2
    canvas[offset_y:offset_y + out_height, offset_x:offset_x + out_width] = resized
    return canvas, scale, offset_x, offset_y


def build_debug_frame(
    lane_frame_bgr: np.ndarray,
    mask: np.ndarray,
    steer: float,
    left_count: int,
    right_count: int,
    sample_rows: list[int],
    left_points: list[tuple[float, float]],
    right_points: list[tuple[float, float]],
) -> np.ndarray:
    debug_src = lane_frame_bgr.copy()
    overlay = debug_src.copy()
    overlay[mask > 0] = (0, 255, 0)
    debug_src = cv2.addWeighted(overlay, 0.35, debug_src, 0.65, 0.0)

    debug, scale, offset_x, offset_y = fit_to_square(debug_src, DISPLAY_SIZE)

    height, width = lane_frame_bgr.shape[:2]
    center_x = width // 2
    roi_top = int(height * ROI_TOP_RATIO)

    def map_point(x: float, y: float) -> tuple[int, int]:
        return (
            int(round(x * scale)) + offset_x,
            int(round(y * scale)) + offset_y,
        )

    top_pt = map_point(center_x, 0)
    bottom_pt = map_point(center_x, height - 1)
    cv2.line(debug, top_pt, bottom_pt, (255, 200, 0), 2)

    roi_left = map_point(0, roi_top)
    roi_right = map_point(width - 1, roi_top)
    cv2.line(debug, roi_left, roi_right, (0, 255, 255), 1)

    for row in sample_rows:
        row_left = map_point(0, row)
        row_right = map_point(width - 1, row)
        cv2.line(debug, row_left, row_right, (255, 0, 255), 1)

    for row, x in left_points:
        cx, cy = map_point(x, row)
        cv2.circle(debug, (cx, cy), 5, (0, 120, 255), -1)

    for row, x in right_points:
        cx, cy = map_point(x, row)
        cv2.circle(debug, (cx, cy), 5, (255, 120, 0), -1)

    cv2.putText(
        debug,
        f"angle={steer:.1f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        debug,
        f"L={left_count} R={right_count}",
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        debug,
        f"proc={lane_frame_bgr.shape[1]}x{lane_frame_bgr.shape[0]} -> view={DISPLAY_SIZE}x{DISPLAY_SIZE}",
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )
    return debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone camera lane debug")
    parser.add_argument("--width", type=int, default=CAM_W)
    parser.add_argument("--height", type=int, default=CAM_H)
    parser.add_argument("--fps", type=int, default=CAM_FPS)
    parser.add_argument("--proc-width", type=int, default=PROC_WIDTH)
    parser.add_argument("--source", default=CAMERA_SOURCE)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    source = args.source
    if isinstance(CAMERA_SOURCE, int):
        try:
            source = int(args.source)
        except (TypeError, ValueError):
            source = args.source

    camera_kind, camera = build_camera(source, args.width, args.height, args.fps)
    last_steer = 0.0
    last_debug_time = 0.0

    try:
        while True:
            ok, frame_bgr = read_frame(camera_kind, camera)
            if not ok or frame_bgr is None:
                print("[INFO] frame read failed, stop.")
                break

            lane_frame = resize_for_lane(frame_bgr, args.proc_width)
            mask = extract_mask(lane_frame)
            (
                last_steer,
                left_count,
                right_count,
                sample_rows,
                left_points,
                right_points,
            ) = compute_steer(mask, last_steer)

            if not args.headless:
                debug_frame = build_debug_frame(
                    lane_frame,
                    mask,
                    last_steer,
                    left_count,
                    right_count,
                    sample_rows,
                    left_points,
                    right_points,
                )
                cv2.imshow("lane-debug", debug_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            now = time.monotonic()
            if now - last_debug_time >= DEBUG_PRINT_INTERVAL:
                print(
                    f"raw={frame_bgr.shape[1]}x{frame_bgr.shape[0]} "
                    f"proc={lane_frame.shape[1]}x{lane_frame.shape[0]} "
                    f"left_pixels={left_count} right_pixels={right_count} steer={last_steer:.1f}",
                    flush=True,
                )
                last_debug_time = now
    finally:
        release_camera(camera_kind, camera)
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
