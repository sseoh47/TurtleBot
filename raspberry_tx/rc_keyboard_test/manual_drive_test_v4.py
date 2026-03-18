#!/usr/bin/python3

import curses
import os
import serial
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


SYSTEM_PYTHON = "/usr/bin/python3"
REEXEC_ENV_KEY = "MANUAL_DRIVE_V4_SYSTEM_PYTHON"


def ensure_system_python():
    # Re-launch under the Raspberry Pi system Python so apt-installed camera packages are available.
    if os.path.exists(SYSTEM_PYTHON) and sys.executable != SYSTEM_PYTHON:
        if os.environ.get(REEXEC_ENV_KEY) != "1":
            env = os.environ.copy()
            env[REEXEC_ENV_KEY] = "1"
            os.execve(SYSTEM_PYTHON, [SYSTEM_PYTHON, __file__, *sys.argv[1:]], env)


ensure_system_python()

from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder


SERIAL_PORT = "/dev/serial0"
BAUDRATE = 38400
SEND_HZ = 10
STEER_ANGLE = 18.0
FORWARD_ANGLE = 0.0
START_BYTE = 0xAC
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
VIDEO_SIZE = (1980, 1080)
VIDEO_FRAMERATE = 30
VIDEO_BITRATE = 10_000_000


@dataclass
class CarCommand:
    class_id: int
    angle: float
    action: int
    status: str


def build_packet(cmd: CarCommand) -> bytes:
    payload = struct.pack("<hfh", cmd.class_id, cmd.angle, cmd.action)
    crc = 0
    for value in payload:
        crc ^= value
    return bytes([START_BYTE]) + payload + bytes([crc])


def send_packet(ser: serial.Serial, cmd: CarCommand):
    ser.write(build_packet(cmd))
    ser.flush()


def key_to_command(key: int):
    if key == curses.KEY_UP:
        return CarCommand(1, FORWARD_ANGLE, 0, "FORWARD")

    if key == curses.KEY_DOWN:
        return CarCommand(1, FORWARD_ANGLE, 0, "FORWARD")

    if key == curses.KEY_LEFT:
        return CarCommand(1, -STEER_ANGLE, 0, "LEFT STEER")

    if key == curses.KEY_RIGHT:
        return CarCommand(1, STEER_ANGLE, 0, "RIGHT STEER")

    digit_map = {
        ord("1"): CarCommand(1, 0.0, 0, "1,0,0"),
        ord("2"): CarCommand(2, 0.0, 0, "2,0,0"),
        ord("3"): CarCommand(3, 0.0, 0, "3,0,0"),
        ord("4"): CarCommand(4, 0.0, 0, "4,0,0"),
        ord("5"): CarCommand(5, 0.0, 0, "5,0,0"),
        ord("6"): CarCommand(6, 0.0, 0, "6,0,0"),
        ord("7"): CarCommand(7, 0.0, 0, "7,0,0"),
        ord("8"): CarCommand(8, 0.0, 0, "8,0,0"),
        ord("9"): CarCommand(9, 0.0, 0, "9,0,0"),
        ord("0"): CarCommand(10, 0.0, 0, "10,0,0"),
    }
    if key in digit_map:
        return digit_map[key]

    alpha_map = {
        ord("p"): CarCommand(0, 0.0, 0, "0,0,0"),
        ord("q"): CarCommand(1, 0.0, 1, "1,0,1"),
        ord("w"): CarCommand(1, 0.0, 2, "1,0,2"),
        ord("e"): CarCommand(2, 0.0, 3, "2,0,3"),
        ord("r"): CarCommand(2, 0.0, 4, "2,0,4"),
        ord("t"): CarCommand(1, 0.0, 9, "1,0,9"),
        ord("y"): CarCommand(1, 0.0, 0, "1,0,0"),
        ord("f"): CarCommand(5, 0.0, 4, "5,0,4"),
        ord("v"): CarCommand(10, 0.0, 4, "10,0,4"),
    }
    if key in alpha_map:
        return alpha_map[key]

    return None


def preview_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class CameraController:
    def __init__(self):
        self.picam2 = Picamera2()
        self.encoder = H264Encoder(VIDEO_BITRATE)
        self.recording_path = None
        self.recording = False
        self.preview_running = False
        self.preview_enabled = preview_available()

        video_config = self.picam2.create_video_configuration(
            # Let Picamera2 choose a preview-compatible main format for QTGL preview.
            main={"size": VIDEO_SIZE},
            controls={"FrameDurationLimits": (33333, 33333)},
        )
        self.picam2.configure(video_config)

    def start_camera(self):
        if self.preview_enabled:
            self.picam2.start_preview(Preview.QTGL)
            self.preview_running = True
        self.picam2.start()

    def stop_camera(self):
        if self.recording:
            self.stop_recording()
        if self.preview_running:
            self.picam2.stop_preview()
            self.preview_running = False
        self.picam2.stop()

    def toggle_preview(self):
        if not self.preview_enabled:
            return False, "Preview unavailable: no desktop display"

        if self.preview_running:
            self.picam2.stop_preview()
            self.preview_running = False
            return True, "Preview OFF"

        self.picam2.start_preview(Preview.QTGL)
        self.preview_running = True
        return True, "Preview ON"

    def start_recording(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = DATA_DIR / f"video_{timestamp}.h264"
        self.picam2.start_recording(self.encoder, str(self.recording_path))
        self.recording = True
        return self.recording_path

    def stop_recording(self):
        if not self.recording:
            return None
        self.picam2.stop_recording()
        saved_path = self.recording_path
        self.recording = False
        self.recording_path = None
        return saved_path


def draw_ui(
    stdscr,
    cmd: CarCommand,
    port_open: bool,
    last_packet_hex: str,
    camera_ready: bool,
    preview_on: bool,
    preview_supported: bool,
    recording: bool,
    recording_path: Optional[Path],
    camera_message: str,
):
    stdscr.erase()
    stdscr.addstr(0, 0, "RC CAR KEYBOARD TEST V4")
    stdscr.addstr(1, 0, "======================")
    stdscr.addstr(3, 0, f"Serial Port : {SERIAL_PORT}")
    stdscr.addstr(4, 0, f"Baudrate    : {BAUDRATE}")
    stdscr.addstr(5, 0, f"Port Open   : {port_open}")
    stdscr.addstr(7, 0, f"Status      : {cmd.status}")
    stdscr.addstr(8, 0, f"class       : {cmd.class_id}")
    stdscr.addstr(9, 0, f"angle       : {cmd.angle:.1f}")
    stdscr.addstr(10, 0, f"action      : {cmd.action}")
    stdscr.addstr(11, 0, f"last packet : {last_packet_hex}")
    stdscr.addstr(13, 0, f"Camera      : {'READY' if camera_ready else 'NOT READY'}")
    stdscr.addstr(14, 0, f"Preview     : {'ON' if preview_on else 'OFF'}")
    stdscr.addstr(15, 0, f"Preview GUI : {'YES' if preview_supported else 'NO'}")
    stdscr.addstr(16, 0, f"Recording   : {'ON' if recording else 'OFF'}")
    stdscr.addstr(17, 0, f"Save Dir    : {DATA_DIR}")
    stdscr.addstr(18, 0, f"Video File  : {recording_path if recording_path else '-'}")
    stdscr.addstr(19, 0, f"Cam Msg     : {camera_message}")

    stdscr.addstr(21, 0, "[Arrow Keys]")
    stdscr.addstr(22, 0, "UP          : 1,0,0")
    stdscr.addstr(23, 0, "DOWN        : 1,0,0")
    stdscr.addstr(24, 0, "LEFT        : 1,-18,0")
    stdscr.addstr(25, 0, "RIGHT       : 1,18,0")

    stdscr.addstr(27, 0, "[Digits]")
    stdscr.addstr(28, 0, "1..9        : n,0,0")
    stdscr.addstr(29, 0, "0           : 10,0,0")

    stdscr.addstr(31, 0, "[Letters]")
    stdscr.addstr(32, 0, "P           : 0,0,0")
    stdscr.addstr(33, 0, "Q           : 1,0,1")
    stdscr.addstr(34, 0, "W           : 1,0,2")
    stdscr.addstr(35, 0, "E           : 2,0,3")
    stdscr.addstr(36, 0, "R           : 2,0,4")
    stdscr.addstr(37, 0, "T           : 1,0,9")
    stdscr.addstr(38, 0, "Y           : 1,0,0")
    stdscr.addstr(39, 0, "F           : 5,0,4")
    stdscr.addstr(40, 0, "V           : 10,0,4")
    stdscr.addstr(41, 0, "M           : video record toggle")
    stdscr.addstr(42, 0, "C           : preview on/off")

    stdscr.addstr(44, 0, "ESC / Shift+Q : quit")
    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(20)

    ser = None
    port_open = False
    camera = None
    camera_message = "Camera init pending"
    camera_ready = False

    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        port_open = True
    except Exception as exc:
        stdscr.addstr(0, 0, f"Serial open failed: {exc}")
        stdscr.refresh()
        stdscr.getch()
        return

    try:
        camera = CameraController()
        camera.start_camera()
        camera_ready = True
        if camera.preview_enabled:
            camera_message = "Preview window opened"
        else:
            camera_message = "Camera ready, preview disabled in headless mode"
    except Exception as exc:
        camera = None
        camera_message = f"Camera init failed: {exc}"

    current_cmd = CarCommand(1, 0.0, 0, "IDLE")
    last_send_time = 0.0
    last_packet_hex = ""

    try:
        while True:
            key = stdscr.getch()

            if key in (27, ord("Q")):
                break

            if key in (ord("m"), ord("M")):
                if camera is None:
                    camera_message = "Camera unavailable"
                elif not camera.recording:
                    try:
                        recording_path = camera.start_recording()
                        camera_message = f"REC START: {recording_path.name}"
                    except Exception as exc:
                        camera_message = f"REC START FAIL: {exc}"
                else:
                    try:
                        saved_path = camera.stop_recording()
                        if saved_path is not None:
                            camera_message = f"REC SAVED: {saved_path.name}"
                        else:
                            camera_message = "REC STOPPED"
                    except Exception as exc:
                        camera_message = f"REC STOP FAIL: {exc}"

            if key in (ord("c"), ord("C")):
                if camera is None:
                    camera_message = "Camera unavailable"
                else:
                    _, camera_message = camera.toggle_preview()

            cmd = key_to_command(key)
            if cmd is not None:
                current_cmd = cmd

            now = time.monotonic()
            if now - last_send_time >= 1.0 / SEND_HZ:
                packet = build_packet(current_cmd)
                last_packet_hex = " ".join(f"{byte:02X}" for byte in packet)
                send_packet(ser, current_cmd)
                last_send_time = now

            draw_ui(
                stdscr,
                current_cmd,
                port_open,
                last_packet_hex,
                camera_ready,
                camera.preview_running if camera else False,
                camera.preview_enabled if camera else False,
                camera.recording if camera else False,
                camera.recording_path if camera else None,
                camera_message,
            )

    finally:
        if camera is not None:
            try:
                camera.stop_camera()
            except Exception:
                pass
        if ser is not None:
            try:
                stop_cmd = CarCommand(0, 0.0, 0, "FINAL STOP")
                send_packet(ser, stop_cmd)
            except Exception:
                pass
            ser.close()


if __name__ == "__main__":
    curses.wrapper(main)
