import sys
import time
from pathlib import Path

# 키보드 테스트용
import select

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_PARENT = ROOT_DIR.parent
TOP_DIR = PROJECT_PARENT.parent


if str(PROJECT_PARENT) not in sys.path:
    sys.path.append(str(PROJECT_PARENT))
if str(TOP_DIR) not in sys.path:
    sys.path.append(str(TOP_DIR))

from config import (
    LANE_MODEL_PATH,
    OBS_MODEL_PATH,
    CAMERA_SOURCE,
    CORAL_COUNT,
    USE_EDGETPU,
    CAM_W,
    CAM_H,
    SEND_HZ,
    DEBUG,
    ARDUINO_PORT,
    ARDUINO_BAUDRATE,
    LIDAR_PORT,
    LIDAR_BAUDRATE,
    LIDAR_TIMEOUT,
    LIDAR_CONF_MIN,
    LIDAR_DIST_MIN,
    LIDAR_DIST_MAX,
    LIDAR_VALID_TIME,
)
from domain.types import FinalCommand
from decision.signal import signal_det
from lidar.lds02 import LDS02
from comm.arduino_serial import ArduinoSerial
from dual_model_edgetpu_v6 import DualModelRunner


# 키보드 테스트용
def check_start_key():
    """키보드 입력 체크: s + 엔터 누르면 True"""
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.readline().strip()
        return key.lower() == "s"
    return False


def main():
    # 듀얼모델 러너 클래스 생성
    runner = DualModelRunner(
        lane_model=LANE_MODEL_PATH,
        obs_model=OBS_MODEL_PATH,
        source=CAMERA_SOURCE,
        coral=CORAL_COUNT,
        use_edgetpu=USE_EDGETPU,
        cam_w=CAM_W,
        cam_h=CAM_H,
    )
    # 라이다 클래스 생성
    lidar = LDS02(
        port=LIDAR_PORT,
        baud=LIDAR_BAUDRATE,
        timeout=LIDAR_TIMEOUT,
        conf_min=LIDAR_CONF_MIN,
        dist_min=LIDAR_DIST_MIN,
        dist_max=LIDAR_DIST_MAX,
        valid_time=LIDAR_VALID_TIME,
    )
    # 라이다 시리얼 생성
    arduino = ArduinoSerial(
        port=ARDUINO_PORT,
        baudrate=ARDUINO_BAUDRATE,
    )
    # 전송속도 필터링을 위한 변수
    last_send_time = 0.0
    start_signal_until = 0.0

    try:
        while True:

            result = runner.step()
            if result is None:
                print("[INFO] inference result is None, stop.")
                break

            if check_start_key():
                start_signal_until = time.monotonic() + 1.0
                print("[KEY] start signal triggered")

            start_signal = time.monotonic() < start_signal_until
            lidar_action = lidar.check_action()

            # 마이크 붙이기 전까지 False
            # 마이크 모델 어디있는지 확인
            # start_signal = False

            final_class, final_angle, final_action = signal_det(
                obj_id=result.obj_id,
                line_id=result.line_id,
                angle=result.angle,
                lidar_action=lidar_action,
                start_signal=start_signal,
            )

            cmd = FinalCommand(
                class_id=final_class,
                angle=final_angle,
                action=final_action,
                status=(
                    f"obj={result.obj_id}, "
                    f"line={result.line_id}, "
                    f"angle={result.angle}, "
                    f"lane_status={result.lane_status}, "
                    f"inter={result.inter_type}, "
                    f"lidar={lidar_action}, "
                    f"start={start_signal}"
                ),
            )

            now = time.monotonic()
            if now - last_send_time >= 1.0 / SEND_HZ:
                packet = arduino.send(cmd)
                last_send_time = now

                if DEBUG:
                    print(
                        f"[SEND] class={cmd.class_id}, "
                        f"angle={cmd.angle}, "
                        f"action={cmd.action}, "
                        f"status={cmd.status}, "
                        f"packet={' '.join(f'{b:02X}' for b in packet)}"
                    )

    finally:
        try:
            runner.close()
        except Exception:
            pass

        try:
            lidar.close()
        except Exception:
            pass

        try:
            arduino.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
