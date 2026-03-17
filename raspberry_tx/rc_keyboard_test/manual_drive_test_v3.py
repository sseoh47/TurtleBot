import curses
import serial
import signal
import struct
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SERIAL_PORT = "/dev/serial0"
BAUDRATE = 38400
SEND_HZ = 10
STEER_ANGLE = 18.0
FORWARD_ANGLE = 0.0
START_BYTE = 0xAC
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RPICAM_CMD = [
    "rpicam-vid",
    "-t",
    "0",
    "--width",
    "1920",
    "--height",
    "1080",
    "--framerate",
    "30",
]


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


def start_recording():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = DATA_DIR / f"video_{timestamp}.h264"
    process = subprocess.Popen(
        [*RPICAM_CMD, "-o", str(output_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, output_path


def stop_recording(process: subprocess.Popen | None, timeout: float = 5.0) -> bool:
    if process is None or process.poll() is not None:
        return True

    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2.0)
            return True
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
            return False


def draw_ui(
    stdscr,
    cmd: CarCommand,
    port_open: bool,
    last_packet_hex: str,
    recording: bool,
    recording_path: Path | None,
    record_message: str,
):
    stdscr.erase()
    stdscr.addstr(0, 0, "RC CAR KEYBOARD TEST V3")
    stdscr.addstr(1, 0, "======================")
    stdscr.addstr(3, 0, f"Serial Port : {SERIAL_PORT}")
    stdscr.addstr(4, 0, f"Baudrate    : {BAUDRATE}")
    stdscr.addstr(5, 0, f"Port Open   : {port_open}")
    stdscr.addstr(7, 0, f"Status      : {cmd.status}")
    stdscr.addstr(8, 0, f"class       : {cmd.class_id}")
    stdscr.addstr(9, 0, f"angle       : {cmd.angle:.1f}")
    stdscr.addstr(10, 0, f"action      : {cmd.action}")
    stdscr.addstr(11, 0, f"last packet : {last_packet_hex}")
    stdscr.addstr(13, 0, f"Recording   : {'ON' if recording else 'OFF'}")
    stdscr.addstr(14, 0, f"Save Dir    : {DATA_DIR}")
    stdscr.addstr(15, 0, f"Video File  : {recording_path if recording_path else '-'}")
    stdscr.addstr(16, 0, f"Record Msg  : {record_message}")

    stdscr.addstr(18, 0, "[Arrow Keys]")
    stdscr.addstr(19, 0, "UP          : 1,0,0")
    stdscr.addstr(20, 0, "DOWN        : 1,0,0")
    stdscr.addstr(21, 0, "LEFT        : 1,-18,0")
    stdscr.addstr(22, 0, "RIGHT       : 1,18,0")

    stdscr.addstr(24, 0, "[Digits]")
    stdscr.addstr(25, 0, "1..9        : n,0,0")
    stdscr.addstr(26, 0, "0           : 10,0,0")

    stdscr.addstr(28, 0, "[Letters]")
    stdscr.addstr(29, 0, "P           : 0,0,0")
    stdscr.addstr(30, 0, "Q           : 1,0,1")
    stdscr.addstr(31, 0, "W           : 1,0,2")
    stdscr.addstr(32, 0, "E           : 2,0,3")
    stdscr.addstr(33, 0, "R           : 2,0,4")
    stdscr.addstr(34, 0, "T           : 1,0,9")
    stdscr.addstr(35, 0, "Y           : 1,0,0")
    stdscr.addstr(36, 0, "F           : 5,0,4")
    stdscr.addstr(37, 0, "V           : 10,0,4")
    stdscr.addstr(38, 0, "M           : video record toggle")

    stdscr.addstr(40, 0, "ESC / Shift+Q : quit")
    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(20)

    ser = None
    port_open = False
    record_process = None
    recording_path = None
    record_message = "READY"

    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        port_open = True
    except Exception as exc:
        stdscr.addstr(0, 0, f"Serial open failed: {exc}")
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
                if record_process is None or record_process.poll() is not None:
                    try:
                        record_process, recording_path = start_recording()
                        record_message = f"REC START: {recording_path.name}"
                    except FileNotFoundError:
                        record_process = None
                        recording_path = None
                        record_message = "rpicam-vid not found"
                    except Exception as exc:
                        record_process = None
                        recording_path = None
                        record_message = f"REC START FAIL: {exc}"
                else:
                    saved_path = recording_path
                    stopped_cleanly = stop_recording(record_process)
                    record_process = None
                    recording_path = None
                    if saved_path is not None:
                        if stopped_cleanly:
                            record_message = f"REC SAVED: {saved_path.name}"
                        else:
                            record_message = f"REC FORCED STOP: {saved_path.name}"
                    else:
                        record_message = "REC STOPPED"

            cmd = key_to_command(key)
            if cmd is not None:
                current_cmd = cmd

            now = time.monotonic()
            if now - last_send_time >= 1.0 / SEND_HZ:
                packet = build_packet(current_cmd)
                last_packet_hex = " ".join(f"{byte:02X}" for byte in packet)
                send_packet(ser, current_cmd)
                last_send_time = now

            is_recording = record_process is not None and record_process.poll() is None
            draw_ui(
                stdscr,
                current_cmd,
                port_open,
                last_packet_hex,
                is_recording,
                recording_path,
                record_message,
            )

    finally:
        if record_process is not None and record_process.poll() is None:
            stop_recording(record_process)
        if ser is not None:
            try:
                stop_cmd = CarCommand(0, 0.0, 0, "FINAL STOP")
                send_packet(ser, stop_cmd)
            except Exception:
                pass
            ser.close()


if __name__ == "__main__":
    curses.wrapper(main)
