import curses
import serial
import struct
import time
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


SERIAL_PORT = "/dev/serial0"
BAUDRATE = 38400
SEND_HZ = 10
STEER_ANGLE = 18.0
FORWARD_ANGLE = 0.0
START_BYTE = 0xAC

SAVE_DIR = Path.home() / "rc_data"


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
        return CarCommand(1, 0.0, 5, "REVERSE")

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
        ord("k"): CarCommand(1, 0.0, 3, "ACT_ROTATE_L"),
        ord("l"): CarCommand(1, 0.0, 4, "ACT_ROTATE_R"),
    }
    if key in alpha_map:
        return alpha_map[key]

    return None


class CameraRecorder:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.proc = None
        self.session_prefix = None
        self.video_file = None
        self.meta_file = None

        self.record_start_mono = None
        self.record_start_wall = None
        self.record_stop_mono = None
        self.record_stop_wall = None

    @property
    def is_recording(self):
        return self.proc is not None

    def start(self):
        if self.proc is not None:
            return self.session_prefix

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_prefix = f"drive_{ts}"
        self.video_file = self.save_dir / f"{self.session_prefix}.h264"
        self.meta_file = self.save_dir / f"{self.session_prefix}_meta.json"

        self.record_start_mono = time.monotonic()
        self.record_start_wall = time.time()
        self.record_stop_mono = None
        self.record_stop_wall = None

        cmd = [
            "rpicam-vid",
            "-t", "0",
            "--nopreview",
            "--width", "1920",
            "--height", "1080",
            "--framerate", "30",
            "-o", str(self.video_file)
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

        self._write_meta()
        return self.session_prefix

    def stop(self):
        if self.proc is None:
            return

        self.record_stop_mono = time.monotonic()
        self.record_stop_wall = time.time()

        try:
            self.proc.terminate()
            self.proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        finally:
            self.proc = None

        self._write_meta()

    def _write_meta(self):
        if self.meta_file is None:
            return

        data = {
            "session_prefix": self.session_prefix,
            "video_file": str(self.video_file) if self.video_file else None,
            "recording": self.is_recording,
            "record_start_mono": self.record_start_mono,
            "record_start_wall": self.record_start_wall,
            "record_stop_mono": self.record_stop_mono,
            "record_stop_wall": self.record_stop_wall,
            "width": 1920,
            "height": 1080,
            "framerate": 30,
            "format": "h264"
        }

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class CommandLogger:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.csv_fp = None
        self.csv_writer = None
        self.csv_file = None
        self.session_prefix = None

    @property
    def is_logging(self):
        return self.csv_writer is not None

    def start(self, session_prefix: str):
        self.stop()

        self.session_prefix = session_prefix
        self.csv_file = self.save_dir / f"{session_prefix}_cmd.csv"
        self.csv_fp = open(self.csv_file, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_fp)
        self.csv_writer.writerow([
            "mono_time",
            "wall_time",
            "class_id",
            "angle",
            "action",
            "status",
            "packet_hex"
        ])
        self.csv_fp.flush()

    def log(self, cmd: CarCommand, packet_hex: str):
        if self.csv_writer is None:
            return

        self.csv_writer.writerow([
            f"{time.monotonic():.6f}",
            f"{time.time():.6f}",
            cmd.class_id,
            f"{cmd.angle:.2f}",
            cmd.action,
            cmd.status,
            packet_hex
        ])
        self.csv_fp.flush()

    def stop(self):
        if self.csv_fp:
            try:
                self.csv_fp.flush()
                self.csv_fp.close()
            except Exception:
                pass

        self.csv_fp = None
        self.csv_writer = None
        self.csv_file = None
        self.session_prefix = None


def safe_addstr(stdscr, y, x, text):
    try:
        height, width = stdscr.getmaxyx()
        if y < height:
            stdscr.addstr(y, x, text[:max(0, width - x - 1)])
    except curses.error:
        pass


def draw_ui(stdscr, cmd: CarCommand, port_open: bool, last_packet_hex: str, cam: CameraRecorder, logger: CommandLogger):
    stdscr.erase()
    safe_addstr(stdscr, 0, 0, "RC CAR KEYBOARD TEST + VIDEO LOGGER")
    safe_addstr(stdscr, 1, 0, "==================================")
    safe_addstr(stdscr, 3, 0, f"Serial Port : {SERIAL_PORT}")
    safe_addstr(stdscr, 4, 0, f"Baudrate    : {BAUDRATE}")
    safe_addstr(stdscr, 5, 0, f"Port Open   : {port_open}")

    safe_addstr(stdscr, 7, 0, f"Status      : {cmd.status}")
    safe_addstr(stdscr, 8, 0, f"class       : {cmd.class_id}")
    safe_addstr(stdscr, 9, 0, f"angle       : {cmd.angle:.1f}")
    safe_addstr(stdscr, 10, 0, f"action      : {cmd.action}")
    safe_addstr(stdscr, 11, 0, f"last packet : {last_packet_hex}")

    safe_addstr(stdscr, 13, 0, "[Recording]")
    safe_addstr(stdscr, 14, 0, f"REC         : {cam.is_recording}")
    safe_addstr(stdscr, 15, 0, f"session     : {cam.session_prefix if cam.session_prefix else '-'}")
    safe_addstr(stdscr, 16, 0, f"video file  : {str(cam.video_file) if cam.video_file else '-'}")
    safe_addstr(stdscr, 17, 0, f"cmd log     : {str(logger.csv_file) if logger.csv_file else '-'}")
    safe_addstr(stdscr, 18, 0, "M           : Start / Stop recording")

    safe_addstr(stdscr, 20, 0, "[Arrow Keys]")
    safe_addstr(stdscr, 21, 0, "UP          : 1,0,0")
    safe_addstr(stdscr, 22, 0, "DOWN        : 1,0,0")
    safe_addstr(stdscr, 23, 0, "LEFT        : 1,-18,0")
    safe_addstr(stdscr, 24, 0, "RIGHT       : 1,18,0")

    safe_addstr(stdscr, 26, 0, "[Digits]")
    safe_addstr(stdscr, 27, 0, "1..9        : n,0,0")
    safe_addstr(stdscr, 28, 0, "0           : 10,0,0")

    safe_addstr(stdscr, 30, 0, "[Letters]")
    safe_addstr(stdscr, 31, 0, "P           : 0,0,0")
    safe_addstr(stdscr, 32, 0, "Q           : 1,0,1")
    safe_addstr(stdscr, 33, 0, "W           : 1,0,2")
    safe_addstr(stdscr, 34, 0, "E           : 2,0,3")
    safe_addstr(stdscr, 35, 0, "R           : 2,0,4")
    safe_addstr(stdscr, 36, 0, "T           : 1,0,9")
    safe_addstr(stdscr, 37, 0, "Y           : 1,0,0")
    safe_addstr(stdscr, 38, 0, "F           : 5,0,4")
    safe_addstr(stdscr, 39, 0, "V           : 10,0,4")

    safe_addstr(stdscr, 41, 0, "ESC / Shift+Q : quit")
    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(20)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    ser = None
    port_open = False

    cam = CameraRecorder(SAVE_DIR)
    logger = CommandLogger(SAVE_DIR)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        port_open = True
    except Exception as exc:
        safe_addstr(stdscr, 0, 0, f"Serial open failed: {exc}")
        stdscr.refresh()
        stdscr.getch()
        return

    current_cmd = CarCommand(1, 0.0, 0, "IDLE")
    last_send_time = 0.0
    last_packet_hex = ""

    try:
        while True:
            key = stdscr.getch()

            if key in (27, ord("Q")):
                break

            if key in (ord("m"), ord("M")):
                if not cam.is_recording:
                    session_prefix = cam.start()
                    logger.start(session_prefix)
                else:
                    logger.stop()
                    cam.stop()

            cmd = key_to_command(key)
            if cmd is not None:
                current_cmd = cmd

            now = time.monotonic()
            if now - last_send_time >= 1.0 / SEND_HZ:
                packet = build_packet(current_cmd)
                last_packet_hex = " ".join(f"{byte:02X}" for byte in packet)
                send_packet(ser, current_cmd)
                last_send_time = now

                if cam.is_recording:
                    logger.log(current_cmd, last_packet_hex)

            draw_ui(stdscr, current_cmd, port_open, last_packet_hex, cam, logger)

    finally:
        try:
            if logger.is_logging:
                logger.stop()
        except Exception:
            pass

        try:
            if cam.is_recording:
                cam.stop()
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
