import curses
import serial
import time
from dataclasses import dataclass


# =========================================================
# 사용자 설정
# =========================================================
SERIAL_PORT = "/dev/serial0"  # 예: /dev/ttyUSB0, /dev/ttyACM0
BAUDRATE = 57600
SEND_HZ = 30  # 초당 전송 횟수
STEER_ANGLE = 18.0  # 좌/우 조향 테스트용 angle
FORWARD_ANGLE = 0.0


# =========================================================
# 명령 구조체
# =========================================================
@dataclass
class CarCommand:
    class_id: int
    angle: float
    action: int
    status: str


# =========================================================
# 시리얼 전송
# =========================================================
def send_packet(ser: serial.Serial, cmd: CarCommand):
    """
    MCU 파서 기준:
        class,angle,action\n
    예:
        1,-12.3,0
    """

    packet = f"{cmd.class_id},{cmd.angle:.1f},{cmd.action}\n"
    ser.write(packet.encode("utf-8"))


# =========================================================
# 키 입력 -> 명령 매핑
# =========================================================
def key_to_command(key: int):
    """
    방향키 / 테스트키를 class, angle, action으로 변환한다.
    반환:
        CarCommand 또는 None
    """

    # -------------------------
    # 기본 주행
    # -------------------------
    if key == curses.KEY_UP:
        return CarCommand(1, FORWARD_ANGLE, 0, "FORWARD")

    if key == curses.KEY_LEFT:
        return CarCommand(1, -STEER_ANGLE, 0, "LEFT STEER")

    if key == curses.KEY_RIGHT:
        return CarCommand(1, STEER_ANGLE, 0, "RIGHT STEER")

    # -------------------------
    # 긴급 정지 / 재출발
    # -------------------------
    if key == ord(" "):
        return CarCommand(3, 0.0, 0, "EMERGENCY STOP (class 3)")

    if key in (ord("r"), ord("R")):
        return CarCommand(9, 0.0, 0, "RESUME (class 9)")

    # -------------------------
    # class 테스트
    # -------------------------
    if key == ord("z"):
        return CarCommand(0, 0.0, 0, "LINE LOST (class 0)")

    if key == ord("x"):
        return CarCommand(6, 0.0, 0, "SPECIAL LINE LEFT (class 6)")

    if key == ord("c"):
        return CarCommand(7, 0.0, 0, "SPECIAL LINE STRAIGHT (class 7)")

    if key == ord("v"):
        return CarCommand(8, 0.0, 0, "SPECIAL LINE LEFT (class 8)")

    # -------------------------
    # 물류/주차 + action 테스트
    # -------------------------
    # 1 : 물류 + 서행
    if key == ord("1"):
        return CarCommand(2, 0.0, 1, "CARGO + ACTION1(SLOW)")

    # 2 : 물류 + 좌측 90도 회전
    if key == ord("2"):
        return CarCommand(2, 0.0, 2, "CARGO + ACTION2(ROTATE)")

    # 3 : 물류 + 정상 정지
    if key == ord("3"):
        return CarCommand(2, 0.0, 3, "CARGO + ACTION3(STOP)")

    # 4 : 물류 + 루틴
    if key == ord("4"):
        return CarCommand(2, 0.0, 4, "CARGO + ACTION4(ROUTINE)")

    # 5 : 주차 + 서행
    if key == ord("5"):
        return CarCommand(5, 0.0, 1, "PARK + ACTION1(SLOW)")

    # 6 : 주차 + 좌측 90도 회전
    if key == ord("6"):
        return CarCommand(5, 0.0, 2, "PARK + ACTION2(ROTATE)")

    # 7 : 주차 + 정상 정지
    if key == ord("7"):
        return CarCommand(5, 0.0, 3, "PARK + ACTION3(STOP)")

    # 8 : 주차 + 루틴
    if key == ord("8"):
        return CarCommand(5, 0.0, 4, "PARK + ACTION4(ROUTINE)")

    # -------------------------
    # 사람 / 자동차 테스트
    # -------------------------
    if key == ord("p"):
        return CarCommand(3, 0.0, 0, "PERSON (class 3)")

    if key == ord("o"):
        return CarCommand(4, 0.0, 0, "CAR (class 4)")

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
    stdscr.addstr(9, 0, f"angle       : {cmd.angle:.1f}")
    stdscr.addstr(10, 0, f"action      : {cmd.action}")
    stdscr.addstr(11, 0, f"last packet : {last_packet}")

    stdscr.addstr(13, 0, "[기본 주행]")
    stdscr.addstr(14, 0, "UP          : forward")
    stdscr.addstr(15, 0, "LEFT        : left steer")
    stdscr.addstr(16, 0, "RIGHT       : right steer")
    stdscr.addstr(17, 0, "SPACE       : emergency stop")
    stdscr.addstr(18, 0, "R           : resume")

    stdscr.addstr(20, 0, "[class 테스트]")
    stdscr.addstr(21, 0, "Z           : class 0 (line lost)")
    stdscr.addstr(22, 0, "X           : class 6")
    stdscr.addstr(23, 0, "C           : class 7")
    stdscr.addstr(24, 0, "V           : class 8")

    stdscr.addstr(26, 0, "[물류/주차 + action 테스트]")
    stdscr.addstr(27, 0, "1           : cargo + action1")
    stdscr.addstr(28, 0, "2           : cargo + action2")
    stdscr.addstr(29, 0, "3           : cargo + action3")
    stdscr.addstr(30, 0, "4           : cargo + action4")
    stdscr.addstr(31, 0, "5           : park  + action1")
    stdscr.addstr(32, 0, "6           : park  + action2")
    stdscr.addstr(33, 0, "7           : park  + action3")
    stdscr.addstr(34, 0, "8           : park  + action4")

    stdscr.addstr(36, 0, "[객체 테스트]")
    stdscr.addstr(37, 0, "P           : person (class 3)")
    stdscr.addstr(38, 0, "O           : car (class 4)")

    stdscr.addstr(40, 0, "Q           : quit")
    stdscr.refresh()


# =========================================================
# 메인
# =========================================================
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(20)

    ser = None
    port_open = False

    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        port_open = True
    except Exception as e:
        port_open = False
        stdscr.addstr(0, 0, f"Serial open failed: {e}")
        stdscr.addstr(2, 0, "포트 이름 확인 후 다시 실행해라.")
        stdscr.addstr(4, 0, "예: /dev/ttyUSB0, /dev/ttyACM0")
        stdscr.refresh()
        stdscr.getch()
        return

    current_cmd = CarCommand(1, 0.0, 0, "IDLE")
    last_send_time = 0.0
    last_packet = ""

    try:
        while True:
            key = stdscr.getch()

            if key in (ord("q"), ord("Q")):
                break

            cmd = key_to_command(key)
            if cmd is not None:
                current_cmd = cmd

            now = time.monotonic()
            if now - last_send_time >= 1.0 / SEND_HZ:
                packet = f"{current_cmd.class_id},{current_cmd.angle:.1f},{current_cmd.action}"
                last_packet = packet

                send_packet(ser, current_cmd)
                last_send_time = now

            draw_ui(stdscr, current_cmd, port_open, last_packet)

    finally:
        try:
            # 종료 시 정지 한 번 보내기
            stop_cmd = CarCommand(3, 0.0, 0, "FINAL STOP")
            send_packet(ser, stop_cmd)
        except Exception:
            pass

        if ser is not None:
            ser.close()


if __name__ == "__main__":
    curses.wrapper(main)
