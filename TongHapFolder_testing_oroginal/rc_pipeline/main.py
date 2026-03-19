import sys
import time
import select
import termios
import tty
from pathlib import Path
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

## 모델 돌리기 위한 import들
from domain.types import FinalCommand
from decision.signal import signal_det
from lidar.lds02 import LDS02
from comm.arduino_serial import ArduinoSerial

# from dual_model_edgetpu_v6 import DualModelRunner
from vision.runner import DualModelRunner

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


def get_key_nonblock():
    """
    엔터 없이 키 1개 읽기.
    입력이 없으면 None 반환.
    """
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def main():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    runner = None
    lidar = None
    arduino = None

    try:
        # print("[INFO] controls: s=start signal, q=quit")

        runner = DualModelRunner(
            lane_model=LANE_MODEL_PATH,
            obs_model=OBS_MODEL_PATH,
            source=CAMERA_SOURCE,
            coral=CORAL_COUNT,
            use_edgetpu=USE_EDGETPU,
            cam_w=CAM_W,
            cam_h=CAM_H,
            cam_fps=10,
            save_debug_frames=False,
            debug_dir="debug_frames",
            debug_save_interval=10,
            max_debug_saves=10,
            obs_interval=2,
        )

        lidar = LDS02(
            port=LIDAR_PORT,
            baud=LIDAR_BAUDRATE,
            timeout=LIDAR_TIMEOUT,
            conf_min=LIDAR_CONF_MIN,
            dist_min=LIDAR_DIST_MIN,
            dist_max=LIDAR_DIST_MAX,
            valid_time=LIDAR_VALID_TIME,
        )

        arduino = ArduinoSerial(
            port=ARDUINO_PORT,
            baudrate=ARDUINO_BAUDRATE,
        )

        last_send_time = 0.0
        start_signal_until = 0.0
        lidar_ignore_until = time.monotonic() + 5.0

        while True:
            key = get_key_nonblock()
            if key == "q":
                print("\n[INFO] quit requested")
                break
            elif key == "s":
                start_signal_until = time.monotonic() + 0.5
                print("\n[KEY] start signal triggered")
            ## debug
            loop_t0 = time.monotonic()
            result = runner.step()
            if result is None:
                print("[INFO] inference result is None, stop.")
                break

            if DEBUG:
                print(
                    f"[VISION] line={result.line_id}, obj={result.obj_id}, "
                    f"lane_status={result.lane_status}, inter={result.inter_type}, "
                    f"step_dt={result.step_dt:.3f}s, "
                    f"frame_age_end={result.frame_age_end:.3f}s"
                    if result.frame_age_end is not None
                    else f"[VISION] line={result.line_id}, obj={result.obj_id}"
                )

            # start_signal = time.monotonic() < start_signal_until
            # lidar_action = lidar.check_action()
            # lidar_action = 0
            ## debug ##
            now = time.monotonic()
            start_signal = now < start_signal_until

            if now < lidar_ignore_until:
                lidar_action = 0
            else:
                lidar_action = lidar.check_action()

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
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass

        if lidar is not None:
            try:
                lidar.close()
            except Exception:
                pass

        if arduino is not None:
            try:
                arduino.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
