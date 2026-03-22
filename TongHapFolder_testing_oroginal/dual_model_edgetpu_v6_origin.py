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
            save_debug_frames=True,
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
            else:
                lidar_action = lidar.check_action()

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
                        f"lane_status={result.lane_status}, "
                        f"inter={result.inter_type}, "
                        f"lidar={lidar_action}, "
                        f"start={start_signal}"
                    ),
                )

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
# =========================
# 3x3 zone helper
# =========================
_X1 = 1.0 / 3.0
_X2 = 2.0 / 3.0
_Y1 = 1.0 / 3.0
_Y2 = 2.0 / 3.0

_LANE_MID_X = 0.5

# straight는 상/중/하 모두 보되,
# 하단이 차량과 가깝기 때문에 가중치를 더 높게 둔다.
_ZONE_W_TOP = 0.20
_ZONE_W_MID = 0.30
_ZONE_W_BOTTOM = 0.50


def _x_zone(cx: float) -> str:
    if cx < _X1:
        return "left"
    if cx < _X2:
        return "center"
    return "right"


def _y_zone(cy: float) -> str:
    if cy < _Y1:
        return "top"
    if cy < _Y2:
        return "mid"
    return "bottom"


def _zone_weight(cy: float) -> float:
    z = _y_zone(cy)
    if z == "bottom":
        return _ZONE_W_BOTTOM
    if z == "mid":
        return _ZONE_W_MID
    return _ZONE_W_TOP


def _has_class(shapes, cls_name, x=None, y=None):
    for s in shapes:
        if s["class_name"] != cls_name:
            continue
        if x is not None and _x_zone(float(s["cx_norm"])) != x:
            continue
        if y is not None and _y_zone(float(s["cy_norm"])) != y:
            continue
        return True
    return False


def _count_class(shapes, cls_name, x=None, y=None):
    c = 0
    for s in shapes:
        if s["class_name"] != cls_name:
            continue
        if x is not None and _x_zone(float(s["cx_norm"])) != x:
            continue
        if y is not None and _y_zone(float(s["cy_norm"])) != y:
            continue
        c += 1
    return c


def compute_lane_error(shapes):
    """
    straight용 angle 원재료 계산

    요구사항 반영:
    - line만 사용
    - 상/중/하단 모두 사용 가능
    - center only는 제외
    - left only / right only는 허용
    - 하단 line에 더 큰 가중치 부여
    """
    lines = [s for s in shapes if s["class_name"] == "line"]
    if not lines:
        return 0.0, "lost"

    left_pts = []
    right_pts = []

    for s in lines:
        cx = float(s["cx_norm"])
        cy = float(s["cy_norm"])
        w = _zone_weight(cy)
        xz = _x_zone(cx)

        if xz == "left":
            left_pts.append((cx, w))
        elif xz == "right":
            right_pts.append((cx, w))
        # center only는 straight 계산에서 제외

    def weighted_mean(items):
        if not items:
            return None
        ws = sum(w for _, w in items)
        if ws <= 1e-6:
            return None
        return sum(x * w for x, w in items) / ws

    left_mean = weighted_mean(left_pts)
    right_mean = weighted_mean(right_pts)

    if left_mean is None and right_mean is None:
        return 0.0, "lost"

    if left_mean is not None and right_mean is not None:
        lane_center = (left_mean + right_mean) * 0.5
        err = ((_LANE_MID_X - lane_center) / 0.5) * 10.0
        err = float(np.clip(err, -10.0, 10.0))
        return err, "ok"

    if left_mean is not None:
        err = ((_LANE_MID_X - left_mean) / 0.5) * 10.0
        err = float(np.clip(err, -10.0, 10.0))
        return err, "left_only"

    err = ((_LANE_MID_X - right_mean) / 0.5) * 10.0
    err = float(np.clip(err, -10.0, 10.0))
    return err, "right_only"


def _raw_classify(shapes):
    """
    현재 프레임 기준 특수상황 분류.

    요구사항 반영:
    - straight는 line 기반, 상/중/하 모두 인식 가능
    - left_t / eeu 상황은 하단 1/3 위주
    - eeu + line 보조 규칙은 상단 제외, 중단/하단 사용
    - down_t는 하단 eeu wide
    - cross는 curve pair 기반 직진 상황
    """
    # straight 참고용
    has_line_left = _has_class(shapes, "line", x="left")
    has_line_right = _has_class(shapes, "line", x="right")

    # left_t
    left_bottom_curve = _has_class(shapes, "curve", x="left", y="bottom")
    mid_eeu = _has_class(shapes, "eeu", y="mid")
    bottom_eeu = _has_class(shapes, "eeu", y="bottom")
    right_line_any = _has_class(shapes, "line", x="right")

    # down_t: 하단에 eeu가 길게 위치
    eeu_bottom_left = _count_class(shapes, "eeu", x="left", y="bottom")
    eeu_bottom_center = _count_class(shapes, "eeu", x="center", y="bottom")
    eeu_bottom_right = _count_class(shapes, "eeu", x="right", y="bottom")
    eeu_bottom_wide = (
        sum(
            [
                eeu_bottom_left > 0,
                eeu_bottom_center > 0,
                eeu_bottom_right > 0,
            ]
        )
        >= 2
    )

    # cross: 너무 먼 top은 제외하고 mid/bottom 중심으로 사용
    curve_mid_left = _has_class(shapes, "curve", x="left", y="mid")
    curve_mid_right = _has_class(shapes, "curve", x="right", y="mid")
    curve_bottom_left = _has_class(shapes, "curve", x="left", y="bottom")
    curve_bottom_right = _has_class(shapes, "curve", x="right", y="bottom")
    cross_cond = (curve_mid_left and curve_mid_right) or (
        curve_bottom_left and curve_bottom_right
    )

    # 우선순위:
    # 1) 하단 curve -> left_t
    if left_bottom_curve:
        return "left_t"

    # 2) 하단 eeu wide -> down_t
    if eeu_bottom_wide:
        return "down_t"

    # 3) cross는 curve pair 기반 직진 상황
    if cross_cond and not eeu_bottom_wide:
        return "cross"

    # 4) 상단 제외, mid/bottom eeu + right line 기반 left_t
    if (mid_eeu or bottom_eeu) and right_line_any and not has_line_left:
        return "left_t"

    return None


def _resolve_cross_down(shapes):
    raw = _raw_classify(shapes)
    if raw in ("left_t", "down_t", "cross"):
        return raw
    return None


class IntersectionFSM:
    """
    현재 프레임 기반의 얇은 분류기.
    복잡한 다프레임 FSM 대신, 매 프레임 최신 결과를 즉시 반영한다.
    """

    def __init__(self, *args, **kwargs):
        self.cur = None

    def update(self, shapes):
        raw = _raw_classify(shapes)
        resolved = _resolve_cross_down(shapes)

        self.cur = resolved
        if resolved is None:
            return (False, None), raw
        return (True, resolved), raw
