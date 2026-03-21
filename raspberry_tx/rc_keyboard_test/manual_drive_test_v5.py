import curses
import serial
import struct
import time
from dataclasses import dataclass


SERIAL_PORT = "/dev/serial0"
BAUDRATE = 38400
SEND_HZ = 10
STEER_ANGLE = 18.0
FORWARD_ANGLE = 0.0
START_BYTE = 0xAC


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
        ord("v"): CarCommand(1, 0.0, 0, "1,0,0"),
        ord("c"): CarCommand(1, -10.0, 0, "1,-10,0"),
        ord("x"): CarCommand(1, -20.0, 0, "1,-20,0"),
        ord("z"): CarCommand(1, -30.0, 0, "1,-30,0"),
        ord("b"): CarCommand(1, 10.0, 0, "1,10,0"),
        ord("n"): CarCommand(1, 20.0, 0, "1,20,0"),
        ord("m"): CarCommand(1, 30.0, 0, "1,30,0"),
    }
    if key in alpha_map:
        return alpha_map[key]

    return None


def draw_ui(stdscr, cmd: CarCommand, port_open: bool, last_packet_hex: str):
    stdscr.erase()
    stdscr.addstr(0, 0, "RC CAR KEYBOARD TEST V5")
    stdscr.addstr(1, 0, "======================")
    stdscr.addstr(3, 0, f"Serial Port : {SERIAL_PORT}")
    stdscr.addstr(4, 0, f"Baudrate    : {BAUDRATE}")
    stdscr.addstr(5, 0, f"Port Open   : {port_open}")
    stdscr.addstr(7, 0, f"Status      : {cmd.status}")
    stdscr.addstr(8, 0, f"class       : {cmd.class_id}")
    stdscr.addstr(9, 0, f"angle       : {cmd.angle:.1f}")
    stdscr.addstr(10, 0, f"action      : {cmd.action}")
    stdscr.addstr(11, 0, f"last packet : {last_packet_hex}")

    stdscr.addstr(13, 0, "[Arrow Keys]")
    stdscr.addstr(14, 0, "UP          : 1,0,0")
    stdscr.addstr(15, 0, "DOWN        : 1,0,0")
    stdscr.addstr(16, 0, "LEFT        : 1,-18,0")
    stdscr.addstr(17, 0, "RIGHT       : 1,18,0")

    stdscr.addstr(19, 0, "[Digits]")
    stdscr.addstr(20, 0, "1..9        : n,0,0")
    stdscr.addstr(21, 0, "0           : 10,0,0")

    stdscr.addstr(23, 0, "[Letters]")
    stdscr.addstr(24, 0, "P           : 0,0,0")
    stdscr.addstr(25, 0, "Q           : 1,0,1")
    stdscr.addstr(26, 0, "W           : 1,0,2")
    stdscr.addstr(27, 0, "E           : 2,0,3")
    stdscr.addstr(28, 0, "R           : 2,0,4")
    stdscr.addstr(29, 0, "T           : 1,0,9")
    stdscr.addstr(30, 0, "Y           : 1,0,0")
    stdscr.addstr(31, 0, "F           : 5,0,4")
    stdscr.addstr(32, 0, "V           : 1,0,0")
    stdscr.addstr(33, 0, "C           : 1,-10,0")
    stdscr.addstr(34, 0, "X           : 1,-20,0")
    stdscr.addstr(35, 0, "Z           : 1,-30,0")
    stdscr.addstr(36, 0, "B           : 1,10,0")
    stdscr.addstr(37, 0, "N           : 1,20,0")
    stdscr.addstr(38, 0, "M           : 1,30,0")

    stdscr.addstr(40, 0, "ESC / Shift+Q : quit")
    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(20)

    ser = None
    port_open = False

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

            cmd = key_to_command(key)
            if cmd is not None:
                current_cmd = cmd

            now = time.monotonic()
            if now - last_send_time >= 1.0 / SEND_HZ:
                packet = build_packet(current_cmd)
                last_packet_hex = " ".join(f"{byte:02X}" for byte in packet)
                send_packet(ser, current_cmd)
                last_send_time = now

            draw_ui(stdscr, current_cmd, port_open, last_packet_hex)

    finally:
        if ser is not None:
            try:
                stop_cmd = CarCommand(0, 0.0, 0, "FINAL STOP")
                send_packet(ser, stop_cmd)
            except Exception:
                pass
            ser.close()


if __name__ == "__main__":
    curses.wrapper(main)
