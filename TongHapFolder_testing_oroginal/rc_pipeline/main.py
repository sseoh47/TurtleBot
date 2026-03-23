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
    CAM_FPS,
    SEND_HZ,
    DEBUG,
    DEBUGAVG,
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


ENABLE_ARDUINO_SEND_TIMING = True
ARDUINO_SEND_TIMING_EVERY = 20
ARDUINO_SEND_TIMING_PRINT_EACH = False
ENABLE_LIDAR_ACTION4_COOLDOWN = True
LIDAR_ACTION4_COOLDOWN_SEC = 10.0


class ArduinoSendTimingLogger:
    """Set ENABLE_ARDUINO_SEND_TIMING=False to disable runtime logs quickly."""

    def __init__(self, enabled: bool, report_every: int, print_each: bool):
        self.enabled = enabled
        self.report_every = max(1, int(report_every))
        self.print_each = print_each
        self.count = 0
        self.sum_ms = 0.0
        self.max_ms = 0.0

    def record(self, elapsed_ms: float) -> None:
        if not self.enabled:
            return

        self.count += 1
        self.sum_ms += elapsed_ms
        self.max_ms = max(self.max_ms, elapsed_ms)

        if self.print_each:
            print(f"[TX] arduino_send={elapsed_ms:.3f}ms")

        if self.count >= self.report_every:
            avg_ms = self.sum_ms / self.count
            print(
                f"[TXAVG{self.count}] arduino_send avg={avg_ms:.3f}ms, "
                f"last={elapsed_ms:.3f}ms, max={self.max_ms:.3f}ms"
            )
            self.count = 0
            self.sum_ms = 0.0
            self.max_ms = 0.0


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
        runner = DualModelRunner(
            lane_model=LANE_MODEL_PATH,
            obs_model=OBS_MODEL_PATH,
            source=CAMERA_SOURCE,
            coral=CORAL_COUNT,
            use_edgetpu=USE_EDGETPU,
            cam_w=CAM_W,
            cam_h=CAM_H,
            cam_fps=CAM_FPS,
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
        lidar_ignore_until = time.monotonic() + 10.0
        lidar_action4_lock_until = 0.0
        prev_lidar_action = 0
        send_timing_logger = ArduinoSendTimingLogger(
            enabled=ENABLE_ARDUINO_SEND_TIMING,
            report_every=ARDUINO_SEND_TIMING_EVERY,
            print_each=ARDUINO_SEND_TIMING_PRINT_EACH,
        )

        # 10루프 평균 timing 로그용
        avg_every = 10
        avg_count = 0

        sum_step_dt = 0.0
        sum_frame_age_start = 0.0
        sum_frame_age_end = 0.0
        sum_lane_ms = 0.0
        sum_obs_ms = 0.0

        count_frame_age_start = 0
        count_frame_age_end = 0
        count_lane_ms = 0
        count_obs_ms = 0

        while True:
            key = get_key_nonblock()
            if key == "q":
                print("\n[INFO] quit requested")
                break
            elif key == "s":
                start_signal_until = time.monotonic() + 0.5
                print("\n[KEY] start signal triggered")

            result = runner.step()
            if result is None:
                print("[INFO] inference result is None, stop.")
                break

            now = time.monotonic()
            start_signal = now < start_signal_until

            if now < lidar_ignore_until:
                lidar_action = 0
            elif (
                ENABLE_LIDAR_ACTION4_COOLDOWN
                and now < lidar_action4_lock_until
            ):
                lidar_action = 0
            else:
                lidar_action = lidar.check_action()
                if (
                    ENABLE_LIDAR_ACTION4_COOLDOWN
                    and prev_lidar_action == 4
                    and lidar_action != 4
                ):
                    lidar_action4_lock_until = now + LIDAR_ACTION4_COOLDOWN_SEC
                    lidar_action = 0
                    if DEBUG:
                        print(
                            f"[LIDAR] action 4 ended -> cooldown "
                            f"{LIDAR_ACTION4_COOLDOWN_SEC:.1f}s"
                        )

                prev_lidar_action = lidar_action

            final_class, final_angle, final_action = signal_det(
                obj_id=result.obj_id,
                line_id=result.line_id,
                angle=result.angle,
                lidar_action=lidar_action,
                start_signal=start_signal,
            )

            # 매 프레임 핵심 로그
            if DEBUG:
                print(
                    f"[DBG] line={result.line_id}, "
                    f"obj={result.obj_id}, "
                    f"angle={result.angle}, "
                    f"center_tape={result.center_tape}, "
                    f"class={final_class}, "
                    f"action={final_action}, "
                    f"lidar={lidar_action}, "
                    f"lane_status={result.lane_status}"
                )

            # 10루프 평균 timing 로그
            if DEBUGAVG:
                avg_count += 1

                if result.step_dt is not None:
                    sum_step_dt += result.step_dt

                if result.frame_age_start is not None:
                    sum_frame_age_start += result.frame_age_start
                    count_frame_age_start += 1

                if result.frame_age_end is not None:
                    sum_frame_age_end += result.frame_age_end
                    count_frame_age_end += 1

                if result.lane_ms is not None:
                    sum_lane_ms += result.lane_ms
                    count_lane_ms += 1

                if result.obs_ms is not None:
                    sum_obs_ms += result.obs_ms
                    count_obs_ms += 1

                if avg_count >= avg_every:
                    avg_step_dt = sum_step_dt / avg_count
                    avg_frame_age_start = (
                        sum_frame_age_start / count_frame_age_start
                        if count_frame_age_start > 0
                        else None
                    )
                    avg_frame_age_end = (
                        sum_frame_age_end / count_frame_age_end
                        if count_frame_age_end > 0
                        else None
                    )
                    avg_lane_ms = (
                        sum_lane_ms / count_lane_ms if count_lane_ms > 0 else None
                    )
                    avg_obs_ms = sum_obs_ms / count_obs_ms if count_obs_ms > 0 else None

                    msg = f"[AVG{avg_every}] step_dt={avg_step_dt:.3f}s"

                    if avg_frame_age_start is not None:
                        msg += f", frame_age_start={avg_frame_age_start:.3f}s"
                    if avg_frame_age_end is not None:
                        msg += f", frame_age_end={avg_frame_age_end:.3f}s"
                    if avg_lane_ms is not None:
                        msg += f", lane_ms={avg_lane_ms:.1f}ms"
                    if avg_obs_ms is not None:
                        msg += f", obs_ms={avg_obs_ms:.1f}ms"

                    print(msg)

                    avg_count = 0
                    sum_step_dt = 0.0
                    sum_frame_age_start = 0.0
                    sum_frame_age_end = 0.0
                    sum_lane_ms = 0.0
                    sum_obs_ms = 0.0
                    count_frame_age_start = 0
                    count_frame_age_end = 0
                    count_lane_ms = 0
                    count_obs_ms = 0

            now = time.monotonic()
            if now - last_send_time >= 1.0 / SEND_HZ:
                cmd = FinalCommand(
                    class_id=final_class,
                    angle=final_angle,
                    action=final_action,
                    status=(
                        f"obj={result.obj_id}, "
                        f"line={result.line_id}, "
                        f"angle={result.angle}, "
                        f"center_tape={result.center_tape}, "
                        f"lane_status={result.lane_status}, "
                        f"inter={result.inter_type}, "
                        f"lidar={lidar_action}, "
                        f"start={start_signal}"
                    ),
                )

                if send_timing_logger.enabled:
                    send_started = time.perf_counter()
                    arduino.send(cmd)
                    send_elapsed_ms = (time.perf_counter() - send_started) * 1000.0
                    send_timing_logger.record(send_elapsed_ms)
                else:
                    arduino.send(cmd)
                last_send_time = now

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
