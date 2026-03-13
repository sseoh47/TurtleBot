import curses
import serial
import time
from dataclasses import dataclass


# =========================================================
# 사용자 설정
# =========================================================
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 38400
SEND_HZ = 30

STEER_ANGLE = 18
FORWARD_ANGLE = 0

START = 0xAC


# =========================================================
# 명령 구조체
# =========================================================
@dataclass
class CarCommand:
    class_id: int
    angle: int
    action: int
    status: str


# =========================================================
# 시리얼 전송 (Binary Packet)
# =========================================================
def send_packet(ser: serial.Serial, cmd: CarCommand):

    class_b = cmd.class_id & 0xFF
    angle_b = int(cmd.angle) & 0xFF
    action_b = cmd.action & 0xFF

    crc = class_b ^ angle_b ^ action_b

    packet = bytes([
        START,
        class_b,
        angle_b,
        action_b,
        crc
    ])

    ser.write(packet)

    return " ".join(f"{b:02X}" for b in packet)


# =========================================================
# 키 입력 -> 명령 매핑
# =========================================================
def key_to_command(key: int):

    if key == curses.KEY_UP:
        return CarCommand(1, FORWARD_ANGLE, 0, "FORWARD")

    if key == curses.KEY_LEFT:
        return CarCommand(1, -STEER_ANGLE, 0, "LEFT STEER")

    if key == curses.KEY_RIGHT:
        return CarCommand(1, STEER_ANGLE, 0, "RIGHT STEER")

    if key == ord(" "):
        return CarCommand(3, 0, 0, "EMERGENCY STOP")

    if key in (ord("r"), ord("R")):
        return CarCommand(9, 0, 0, "RESUME")

    if key == ord("z"):
        return CarCommand(0, 0, 0, "LINE LOST")

    if key == ord("x"):
        return CarCommand(6, 0, 0, "CLASS 6")

    if key == ord("c"):
        return CarCommand(7, 0, 0, "CLASS 7")

    if key == ord("v"):
        return CarCommand(8, 0, 0, "CLASS 8")

    if key == ord("1"):
        return CarCommand(2, 0, 1, "CARGO ACTION1")

    if key == ord("2"):
        return CarCommand(2, 0, 2, "CARGO ROTATE")

    if key == ord("3"):
        return CarCommand(2, 0, 3, "CARGO STOP")

    if key == ord("4"):
        return CarCommand(2, 0, 4, "CARGO ROUTINE")

    if key == ord("5"):
        return CarCommand(5, 0, 1, "PARK ACTION1")

    if key == ord("6"):
        return CarCommand(5, 0, 2, "PARK ROTATE")

    if key == ord("7"):
        return CarCommand(5, 0, 3, "PARK STOP")

    if key == ord("8"):
        return CarCommand(5, 0, 4, "PARK ROUTINE")

    if key == ord("p"):
        return CarCommand(3, 0, 0, "PERSON")

    if key == ord("o"):
        return CarCommand(4, 0, 0, "CAR")

    return None


# =========================================================
# UI 출력
# =========================================================
def draw_ui(stdscr, cmd: CarCommand, port_open: bool, last_packet: str):

    stdscr.erase()

    stdscr.addstr(0, 0, "RC CAR KEYBOARD TEST")
    stdscr.addstr(1, 0, "====================")

    stdscr.addstr(3, 0, f"Serial Port : {SERIAL_PORT}")
    stdscr.addstr(4, 0, f"Baudrate    : {BAUDRATE}")
    stdscr.addstr(5, 0, f"Port Open   : {port_open}")

    stdscr.addstr(7, 0, f"Status      : {cmd.status}")
    stdscr.addstr(8, 0, f"class       : {cmd.class_id}")
    stdscr.addstr(9, 0, f"angle       : {cmd.angle}")
    stdscr.addstr(10, 0, f"action      : {cmd.action}")

    stdscr.addstr(11, 0, f"packet(hex) : {last_packet}")

    stdscr.addstr(13, 0, "[Driving]")
    stdscr.addstr(14, 0, "UP    : forward")
    stdscr.addstr(15, 0, "LEFT  : left steer")
    stdscr.addstr(16, 0, "RIGHT : right steer")

    stdscr.addstr(18, 0, "SPACE : emergency stop")
    stdscr.addstr(19, 0, "R     : resume")

    stdscr.addstr(21, 0, "Q     : quit")

    stdscr.refresh()


# =========================================================
# 메인
# =========================================================
def main(stdscr):

    curses.curs_set(0)
    stdscr.nodelay(True)

    ser = None
    port_open = False

    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        port_open = True
    except Exception as e:
        stdscr.addstr(0, 0, f"Serial open failed: {e}")
        stdscr.refresh()
        stdscr.getch()
        return

    current_cmd = CarCommand(1, 0, 0, "IDLE")

    last_send_time = 0
    last_packet = ""

    try:
        while True:

            key = stdscr.getch()

            if key in (ord("q"), ord("Q")):
                break

            cmd = key_to_command(key)
            if cmd:
                current_cmd = cmd

            now = time.monotonic()

            if now - last_send_time >= 1 / SEND_HZ:

                last_packet = send_packet(ser, current_cmd)
                last_send_time = now

            draw_ui(stdscr, current_cmd, port_open, last_packet)

    finally:

        try:
            stop_cmd = CarCommand(3, 0, 0, "FINAL STOP")
            send_packet(ser, stop_cmd)
        except:
            pass

        if ser:
            ser.close()


if __name__ == "__main__":
    curses.wrapper(main)