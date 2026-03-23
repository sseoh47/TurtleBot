#!/usr/bin/env python3

from __future__ import annotations

import argparse
import struct
import time

import cv2
import numpy as np
import serial
from picamera2 import Picamera2


START_BYTE = 0xAC
CLASS_LINE_FOLLOW = 1
CLASS_CENTER_BLACK = 7
CLASS_STARTUP = 9
ACTION_NONE = 0

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
PROC_SIZE = 320
FPS = 30
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 38400

ROI_TOP_RATIO = 0.6
THRESHOLD_VALUE = 200
ROW_HALF_HEIGHT = 3
MIN_SIDE_PIXELS = 8
ROI_SAMPLE_RATIOS = (0.75, 0.50, 0.25)
ROI_SIDE_MIN_PIXELS = 60

MAX_STEER = 5.0
SEARCH_STEER = 5.0
SLOPE_SUM_LIMIT = 0.30
DEBUG_PRINT_INTERVAL = 0.5
SHOW_DEBUG = True
STARTUP_SIGNAL_SECONDS = 3.0
STARTUP_SEND_INTERVAL = 0.05
CENTER_BLACK_Y_START_RATIO = 0.6
CENTER_BLACK_Y_END_RATIO = 0.7
CENTER_BLACK_HALF_WIDTH_RATIO = 0.08
CENTER_TAPE_R_MAX = 50
CENTER_TAPE_G_MIN = 100
CENTER_TAPE_G_MAX = 200
CENTER_TAPE_B_MIN = 200
CENTER_TAPE_MIN_PIXELS = 1

# Send steering in the -10 to +10 range.
TX_ANGLE_SCALE = 2.0


def send_packet(
    ser: serial.Serial,
    angle: float,
    class_id: int = CLASS_LINE_FOLLOW,
    action: int = ACTION_NONE,
) -> None:
    payload = struct.pack("<hfh", int(class_id), float(angle), int(action))
    crc = 0
    for byte in payload:
        crc ^= byte
    ser.write(bytes([START_BYTE]) + payload + bytes([crc]))


def build_camera(width: int, height: int, fps: int) -> Picamera2:
    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)},
        controls={"FrameRate": fps},
    )
    camera.configure(config)
    camera.start()
    time.sleep(1.0)
    return camera


def resize_letterbox(frame_rgb: np.ndarray, target_size: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    scale = min(target_size / float(width), target_size / float(height))
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_rgb, (out_width, out_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    offset_x = (target_size - out_width) // 2
    offset_y = (target_size - out_height) // 2
    canvas[offset_y:offset_y + out_height, offset_x:offset_x + out_width] = resized
    return canvas


def extract_mask(frame_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
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

    left_xs = xs[xs < (width // 2)]
    right_xs = xs[xs >= (width // 2)]

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


def normalize_steer(value: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return float(np.clip((value / limit) * MAX_STEER, -MAX_STEER, MAX_STEER))


def detect_center_tape(
    frame_rgb: np.ndarray,
) -> tuple[bool, tuple[float, float, float], int, tuple[int, int, int, int]]:
    height, width = frame_rgb.shape[:2]
    y1 = int(height * CENTER_BLACK_Y_START_RATIO)
    y2 = int(height * CENTER_BLACK_Y_END_RATIO)
    mid = width // 2
    half_w = max(1, int(width * CENTER_BLACK_HALF_WIDTH_RATIO))
    x1 = max(0, mid - half_w)
    x2 = min(width, mid + half_w)

    center_roi = frame_rgb[y1:y2, x1:x2]
    if center_roi.size == 0:
        return False, (0.0, 0.0, 0.0), 0, (x1, y1, x2, y2)

    mean_rgb = tuple(float(v) for v in center_roi.reshape(-1, 3).mean(axis=0))
    r = center_roi[:, :, 0]
    g = center_roi[:, :, 1]
    b = center_roi[:, :, 2]
    in_range = (
        (r <= CENTER_TAPE_R_MAX)
        & (g >= CENTER_TAPE_G_MIN)
        & (g <= CENTER_TAPE_G_MAX)
        & (b >= CENTER_TAPE_B_MIN)
    )
    match_count = int(np.count_nonzero(in_range))
    return (
        match_count >= CENTER_TAPE_MIN_PIXELS,
        mean_rgb,
        match_count,
        (x1, y1, x2, y2),
    )


def build_debug_frame(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    steer: float,
    tx_class_id: int,
    center_tape: bool,
    center_rgb_mean: tuple[float, float, float],
    center_match_count: int,
    center_rect: tuple[int, int, int, int],
    left_count: int,
    right_count: int,
    capture_width: int,
    capture_height: int,
) -> np.ndarray:
    debug = frame_rgb.copy()
    overlay = debug.copy()
    overlay[mask > 0] = (0, 255, 0)
    debug = cv2.addWeighted(overlay, 0.35, debug, 0.65, 0.0)

    height, width = mask.shape
    center_x = width // 2
    roi_top = int(height * ROI_TOP_RATIO)
    roi_height = max(1, height - roi_top)

    cv2.line(debug, (center_x, 0), (center_x, height - 1), (255, 200, 0), 2)
    cv2.line(debug, (0, roi_top), (width - 1, roi_top), (0, 255, 255), 1)

    for ratio in ROI_SAMPLE_RATIOS:
        row = roi_top + int(roi_height * ratio)
        row = min(height - 1, max(roi_top, row))
        cv2.line(debug, (0, row), (width - 1, row), (255, 0, 255), 1)

        left, right = scan_row(mask, row)
        if left is not None:
            cv2.circle(debug, (int(round(left)), row), 5, (0, 120, 255), -1)
        if right is not None:
            cv2.circle(debug, (int(round(right)), row), 5, (255, 120, 0), -1)

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
        (
            f"case={tx_class_id} center_tape={'Y' if center_tape else 'N'} "
            f"rgb=({center_rgb_mean[0]:.0f},{center_rgb_mean[1]:.0f},{center_rgb_mean[2]:.0f}) "
            f"hit={center_match_count}"
        ),
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        debug,
        f"raw={capture_width}x{capture_height} -> proc={width}x{height}",
        (10, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )

    x1, y1, x2, y2 = center_rect
    rect_color = (0, 0, 255) if center_tape else (255, 255, 0)
    cv2.rectangle(debug, (x1, y1), (x2, y2), rect_color, 2)
    return debug


def compute_steer(mask: np.ndarray, last_steer: float) -> tuple[float, int, int]:
    height, width = mask.shape
    roi_top = int(height * ROI_TOP_RATIO)
    roi_height = max(1, height - roi_top)
    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []

    for ratio in ROI_SAMPLE_RATIOS:
        row = roi_top + int(roi_height * ratio)
        row = min(height - 1, max(roi_top, row))
        left, right = scan_row(mask, row)

        if left is not None:
            left_points.append((float(row), left))
        if right is not None:
            right_points.append((float(row), right))

    left_count = cv2.countNonZero(mask[:, :width // 2])
    right_count = cv2.countNonZero(mask[:, width // 2:])

    left_detected = left_count >= ROI_SIDE_MIN_PIXELS
    right_detected = right_count >= ROI_SIDE_MIN_PIXELS

    if left_detected and not right_detected:
        return SEARCH_STEER, left_count, right_count
    if right_detected and not left_detected:
        return -SEARCH_STEER, left_count, right_count
    if not left_detected and not right_detected:
        return 0.0, left_count, right_count
    if left_detected and right_detected and len(left_points) >= 2 and len(right_points) >= 2:
        left_slope = fit_slope(left_points)
        right_slope = fit_slope(right_points)

        if left_slope is not None and right_slope is not None:
            return (
                normalize_steer(left_slope + right_slope, SLOPE_SUM_LIMIT),
                left_count,
                right_count,
            )
        return last_steer, left_count, right_count

    return last_steer, left_count, right_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Picamera2 lane sender for MAIN_V2")
    parser.add_argument("--serial-port", default=SERIAL_PORT)
    parser.add_argument("--baudrate", type=int, default=BAUDRATE)
    parser.add_argument("--capture-width", type=int, default=CAPTURE_WIDTH)
    parser.add_argument("--capture-height", type=int, default=CAPTURE_HEIGHT)
    parser.add_argument("--proc-size", type=int, default=PROC_SIZE)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-serial", action="store_true")
    args = parser.parse_args()

    ser = None if args.no_serial else serial.Serial(args.serial_port, args.baudrate, timeout=0)
    camera = build_camera(args.capture_width, args.capture_height, args.fps)
    last_steer = 0.0
    last_debug_time = 0.0
    startup_end_time = time.monotonic() + STARTUP_SIGNAL_SECONDS

    try:
        if ser is not None:
            while time.monotonic() < startup_end_time:
                send_packet(ser, 0.0, class_id=CLASS_STARTUP, action=ACTION_NONE)
                time.sleep(STARTUP_SEND_INTERVAL)

        while True:
            frame_rgb = camera.capture_array()
            proc_rgb = resize_letterbox(frame_rgb, args.proc_size)
            mask = extract_mask(proc_rgb)
            last_steer, left_count, right_count = compute_steer(mask, last_steer)
            center_tape, center_rgb_mean, center_match_count, center_rect = detect_center_tape(proc_rgb)
            tx_class_id = CLASS_CENTER_BLACK if center_tape else CLASS_LINE_FOLLOW
            tx_angle = 0.0 if center_tape else last_steer * TX_ANGLE_SCALE

            if ser is not None:
                send_packet(ser, tx_angle, class_id=tx_class_id)

            if SHOW_DEBUG and not args.headless:
                debug_frame = build_debug_frame(
                    proc_rgb,
                    mask,
                    last_steer,
                    tx_class_id,
                    center_tape,
                    center_rgb_mean,
                    center_match_count,
                    center_rect,
                    left_count,
                    right_count,
                    args.capture_width,
                    args.capture_height,
                )
                cv2.imshow("lane-debug", debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            now = time.monotonic()
            if now - last_debug_time >= DEBUG_PRINT_INTERVAL:
                print(
                    f"raw={args.capture_width}x{args.capture_height} "
                    f"proc={args.proc_size}x{args.proc_size} "
                    f"left_pixels={left_count} right_pixels={right_count} "
                    f"steer={last_steer:.1f} class={tx_class_id} "
                    f"center_tape={center_tape} "
                    f"rgb=({center_rgb_mean[0]:.0f},{center_rgb_mean[1]:.0f},{center_rgb_mean[2]:.0f}) "
                    f"hit={center_match_count}",
                    flush=True,
                )
                last_debug_time = now
    finally:
        camera.stop()
        if ser is not None:
            ser.close()
        if SHOW_DEBUG and not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
