#!/usr/bin/env python3
"""
Raspberry Pi 5 + Camera Module 3 lane/intersection sender.

This script captures frames with Picamera2, detects white lane markings,
classifies simple junction shapes, and sends commands to the controller
using the binary packet format expected by MAIN_V2/communication.cpp.
"""

## 실행
## python3 vision_sender.py --serial-port /dev/ttyUSB0

from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import serial
from picamera2 import Picamera2


START_BYTE = 0xAC
ACTION_NONE = 0

CLASS_LINE_LOST = 0
CLASS_LINE_FOLLOW = 1
CLASS_CROSS = 7
CLASS_LEFT_TURN = 8


@dataclass
class VisionConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    min_lane_pixels: int = 1200
    angle_gain: float = 0.12
    max_angle: float = 100.0
    debug_view: bool = True


class CommandSender:
    def __init__(self, port: str, baudrate: int) -> None:
        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=0)

    def send(self, class_id: int, angle: float, action: int = ACTION_NONE) -> None:
        payload = struct.pack("<hfh", int(class_id), float(angle), int(action))
        crc = 0
        for byte in payload:
            crc ^= byte
        packet = bytes([START_BYTE]) + payload + bytes([crc])
        self._serial.write(packet)

    def close(self) -> None:
        if self._serial.is_open:
            self._serial.close()


class LaneIntersectionDetector:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[int, float, np.ndarray, Optional[dict]]:
        binary = self._extract_white_mask(frame_bgr)
        lane_mask, lane_info = self._select_main_lane(binary)

        if lane_info is None or lane_info["area"] < self.config.min_lane_pixels:
            debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            return CLASS_LINE_LOST, 0.0, debug, None

        branches = self._detect_branches(lane_mask)
        class_id = self._classify(branches)
        angle = self._compute_angle(lane_mask)
        debug = self._build_debug_view(frame_bgr, lane_mask, branches, class_id, angle)

        lane_info.update({"branches": branches})
        return class_id, angle, debug, lane_info

    def _extract_white_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 180], dtype=np.uint8)
        upper_white = np.array([180, 80, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _select_main_lane(self, binary: np.ndarray) -> Tuple[np.ndarray, Optional[dict]]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(binary), None

        height, width = binary.shape
        best_label = None
        best_score = -1.0

        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            cx = x + w / 2.0
            bottom_overlap = max(0, min(y + h, height) - max(y, int(height * 0.82)))
            reaches_bottom = (y + h) >= int(height * 0.92)
            score = area + bottom_overlap * 35 - abs(cx - width / 2.0) * 1.8
            if reaches_bottom:
                score += 2500

            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return np.zeros_like(binary), None

        lane_mask = np.zeros_like(binary)
        lane_mask[labels == best_label] = 255

        x, y, w, h, area = stats[best_label]
        return lane_mask, {"bbox": (x, y, w, h), "area": int(area)}

    def _detect_branches(self, lane_mask: np.ndarray) -> dict:
        height, width = lane_mask.shape

        left_band = lane_mask[int(height * 0.40):int(height * 0.80), :int(width * 0.15)]
        right_band = lane_mask[int(height * 0.40):int(height * 0.80), int(width * 0.85):]
        top_band = lane_mask[:int(height * 0.22), int(width * 0.25):int(width * 0.75)]
        bottom_band = lane_mask[int(height * 0.78):, int(width * 0.25):int(width * 0.75)]

        branches = {
            "left": cv2.countNonZero(left_band) > 180,
            "right": cv2.countNonZero(right_band) > 180,
            "top": cv2.countNonZero(top_band) > 180,
            "bottom": cv2.countNonZero(bottom_band) > 250,
        }
        return branches

    def _classify(self, branches: dict) -> int:
        left = branches["left"]
        right = branches["right"]
        top = branches["top"]
        bottom = branches["bottom"]

        if left and right and top and bottom:
            return CLASS_CROSS

        left_turn_patterns = [
            left and top and bottom and not right,   # ㅓ
            left and right and bottom and not top,   # ㅜ
            left and bottom and not top and not right,  # ㄱ
        ]
        if any(left_turn_patterns):
            return CLASS_LEFT_TURN

        return CLASS_LINE_FOLLOW

    def _compute_angle(self, lane_mask: np.ndarray) -> float:
        height, width = lane_mask.shape
        sample_rows = [
            int(height * 0.92),
            int(height * 0.78),
            int(height * 0.64),
        ]

        centers = []
        weights = [0.55, 0.30, 0.15]

        for row in sample_rows:
            band = lane_mask[max(0, row - 6):min(height, row + 6), :]
            xs = np.where(band > 0)[1]
            if xs.size == 0:
                centers.append(None)
                continue
            centers.append(float(np.mean(xs)))

        valid = [(center, weight) for center, weight in zip(centers, weights) if center is not None]
        if not valid:
            return 0.0

        weighted_center = sum(center * weight for center, weight in valid) / sum(
            weight for _, weight in valid
        )
        pixel_error = weighted_center - (width / 2.0)
        angle = float(np.clip(pixel_error * self.config.angle_gain, -self.config.max_angle, self.config.max_angle))
        return angle

    def _build_debug_view(
        self,
        frame_bgr: np.ndarray,
        lane_mask: np.ndarray,
        branches: dict,
        class_id: int,
        angle: float,
    ) -> np.ndarray:
        debug = frame_bgr.copy()
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)

        height, width = lane_mask.shape
        cv2.line(debug, (width // 2, 0), (width // 2, height), (255, 200, 0), 1)

        text = f"class={class_id} angle={angle:.1f}"
        branch_text = (
            f"L:{int(branches['left'])} R:{int(branches['right'])} "
            f"T:{int(branches['top'])} B:{int(branches['bottom'])}"
        )
        cv2.putText(debug, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(debug, branch_text, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return debug


def build_camera(config: VisionConfig) -> Picamera2:
    camera = Picamera2()
    preview_config = camera.create_preview_configuration(
        main={"format": "RGB888", "size": (config.width, config.height)},
        controls={"FrameRate": config.fps},
    )
    camera.configure(preview_config)
    camera.start()
    time.sleep(1.0)
    return camera


def parse_args() -> VisionConfig:
    parser = argparse.ArgumentParser(description="Lane and junction sender for TurtleBot")
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--angle-gain", type=float, default=0.12)
    parser.add_argument("--min-lane-pixels", type=int, default=1200)
    args = parser.parse_args()

    return VisionConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        angle_gain=args.angle_gain,
        min_lane_pixels=args.min_lane_pixels,
        debug_view=not args.headless,
    )


def main() -> None:
    config = parse_args()
    sender = CommandSender(config.serial_port, config.baudrate)
    detector = LaneIntersectionDetector(config)
    camera = build_camera(config)

    try:
        while True:
            frame_rgb = camera.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            class_id, angle, debug, _ = detector.process(frame_bgr)

            if class_id in (CLASS_CROSS, CLASS_LEFT_TURN):
                angle = 0.0
            sender.send(class_id, angle, ACTION_NONE)

            if config.debug_view:
                cv2.imshow("turtlebot-camera", debug)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
    finally:
        camera.stop()
        sender.close()
        if config.debug_view:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
