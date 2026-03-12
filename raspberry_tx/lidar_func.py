import serial
import time
from typing import Optional, Tuple

HEADER = 0x54
LENGTH = 0x2C
PACKET_SIZE = 47


# =============================================================
# 안정화 필터들
# =============================================================


class StableValueFilter:
    """같은 값이 threshold번 연속 들어올 때만 채택한다."""

    def __init__(self, threshold: int = 2):
        self.threshold = threshold
        self.last_candidate = None
        self.count = 0
        self.stable_value = None

    def update(self, value):
        if value == self.last_candidate:
            self.count += 1
        else:
            self.last_candidate = value
            self.count = 1

        if self.count >= self.threshold:
            self.stable_value = value

        return self.stable_value


class AngleFilter:
    """차선 조향 오차값(angle)을 clamp + smoothing 한다."""

    def __init__(self, alpha: float = 0.7, limit: float = 30.0):
        self.alpha = alpha
        self.limit = limit
        self.prev = None

    def _clamp(self, angle: float) -> float:
        if angle > self.limit:
            return self.limit
        if angle < -self.limit:
            return -self.limit
        return angle

    def update(self, angle: Optional[float]) -> Optional[float]:
        if angle is None:
            return self.prev

        angle = self._clamp(angle)

        if self.prev is None:
            self.prev = angle
        else:
            self.prev = self.alpha * self.prev + (1.0 - self.alpha) * angle

        return self.prev


class PerceptionStabilizer:
    """
    - obj_id  : 연속 프레임 안정화
    - line_id : 연속 프레임 안정화
    - angle   : smoothing + clamp
    """

    def __init__(
        self, threshold: int = 2, alpha: float = 0.7, angle_limit: float = 30.0
    ):
        self.obj_filter = StableValueFilter(threshold=threshold)
        self.line_filter = StableValueFilter(threshold=threshold)
        self.angle_filter = AngleFilter(alpha=alpha, limit=angle_limit)

    def update(
        self,
        obj_id: Optional[int],
        line_id: Optional[int],
        angle: Optional[float],
    ) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        stable_obj = self.obj_filter.update(obj_id)
        stable_line = self.line_filter.update(line_id)
        smooth_angle = self.angle_filter.update(angle)
        return stable_obj, stable_line, smooth_angle


# =============================================================
# 최종 상태 결정기
# =============================================================
def signal_det(
    obj_id: Optional[int],
    line_id: Optional[int],
    angle: Optional[float],
    lidar_action: int,
    start_signal: bool = False,
) -> Tuple[int, Optional[float], int]:
    """
    반환:
        final_class, final_angle, final_action

    입력 의미:
        obj_id       : 객체 인식 결과
                       None, 2(물류), 3(사람), 4(자동차), 5(주차)
        line_id      : 차선 인식 결과
                       0(차선없음), 1(일반차선), 6/7/8(특수차선)
        angle        : 차선 중심 오차 기반 조향 보정값
        lidar_action : 라이다 기반 특수상황 판단 결과
                       0(없음), 1(서행), 2(좌측90도회전), 3(정상정지), 9(긴급정지)
        start_signal : 마이크/음성 기반 출발 신호(bool)

    우선순위:
        1. 사람/자동차
        2. 출발 신호
        3. 물류/주차 객체 확정
        4. 일반 차선 주행 중 라이다 선감지
        5. 특수 차선
        6. 일반 차선
        7. 차선 인식 실패
    """

    # 1) 사람 / 자동차는 class만으로 정지
    if obj_id in (3, 4):
        return obj_id, None, 0

    # 2) 출발 신호
    if start_signal:
        return 9, None, 0

    # 3) 물류 / 주차 객체 확정
    # 라이다가 판단한 특수 action을 그대로 함께 전달
    if obj_id in (2, 5):
        return obj_id, None, lidar_action

    # 4) 일반 차선 주행 중 라이다가 먼저 특수상황을 감지한 경우
    if line_id == 1 and lidar_action in (1, 2, 3, 9):
        return 1, angle, lidar_action

    # 5) 특수 차선
    if line_id in (6, 7, 8):
        return line_id, None, 0

    # 6) 일반 차선 주행
    if line_id == 1:
        return 1, angle, 0

    # 7) 차선 인식 실패
    return 0, None, 0


# =============================================================
# lidar parsing utils
# =============================================================


def u16_le(b0: int, b1: int) -> int:
    return (b1 << 8) | b0


def wrap_deg(angle: float) -> float:
    angle %= 360.0
    return angle


def cw_diff_deg(start: float, end: float) -> float:
    diff = wrap_deg(end) - wrap_deg(start)
    return diff + 360.0 if diff < 0 else diff


def angle_to_idx(angle: float) -> int:
    return int(round(wrap_deg(angle))) % 360


# =============================================================
# lidar class
# =============================================================


class LDS02:
    """
    LDS-02 데이터를 파싱하고 라이다 기반 특수 action을 판단한다.
    check_lidar() 반환값이 그대로 action 역할을 한다.
    """

    def __init__(
        self,
        port="/dev/ttyUSB0",
        baud=115200,
        timeout=0.2,
        conf_min=50,
        dist_min=100,
        dist_max=1000,
        valid_time=0.2,
    ):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        self.ranges = [None] * 360
        self.timestamps = [0.0] * 360

        self.conf_min = conf_min
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.valid_time = valid_time

        # 단계 기반 라이다 시퀀스
        # 0 -> 1(서행) -> 2(좌측 90도 회전) -> 3(정지)
        self.stage = 0

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    def reset_stage(self):
        """특수 시퀀스 종료 후 stage를 초기화한다."""
        self.stage = 0

    def _read_exact(self, n: int) -> bytes:
        data = self.ser.read(n)
        if len(data) != n:
            raise TimeoutError(f"expected {n} bytes, got {len(data)}")
        return data

    def update_once(self):
        """패킷 1개를 읽어 360도 거리 맵 캐시를 갱신한다."""
        while True:
            b = self.ser.read(1)
            if not b:
                raise TimeoutError("header timeout")
            if b[0] != HEADER:
                continue

            lb = self._read_exact(1)
            if lb[0] != LENGTH:
                continue

            rest = self._read_exact(PACKET_SIZE - 2)
            packet = bytes([HEADER, LENGTH]) + rest
            break

        start_angle = u16_le(packet[4], packet[5]) / 100.0
        data = packet[6:42]
        end_angle = u16_le(packet[42], packet[43]) / 100.0

        diff = cw_diff_deg(start_angle, end_angle)
        step = diff / 11.0
        now = time.monotonic()

        for i in range(12):
            off = i * 3
            dist = u16_le(data[off], data[off + 1])
            conf = data[off + 2]
            angle = wrap_deg(start_angle + step * i)
            idx = angle_to_idx(180 - angle)

            if conf < self.conf_min:
                continue
            if dist < self.dist_min or dist > self.dist_max:
                continue

            self.ranges[idx] = dist
            self.timestamps[idx] = now

    def is_object_in_range(
        self, min_angle: float, max_angle: float, threshold_mm: int
    ) -> bool:
        """
        각도 범위 내 최근 유효 포인트 중 임계거리 이하가 있는지 확인한다.
        각도 기준은 전방 0도, 반시계 방향 증가로 가정한다.
        """
        start = angle_to_idx(min_angle)
        end = angle_to_idx(max_angle)
        now = time.monotonic()

        i = start
        while True:
            dist = self.ranges[i]
            ts = self.timestamps[i]

            if dist is not None and (now - ts) <= self.valid_time:
                if dist <= threshold_mm:
                    return True

            if i == end:
                break
            i = (i + 1) % 360

        return False

    def get_min_distance(self, min_angle: float, max_angle: float) -> Optional[int]:
        """각도 구간 내 최근 유효 거리 중 최소값을 반환한다."""
        start = angle_to_idx(min_angle)
        end = angle_to_idx(max_angle)
        now = time.monotonic()
        result = None

        i = start
        while True:
            d = self.ranges[i]
            ts = self.timestamps[i]

            if d is not None and (now - ts) <= self.valid_time:
                if result is None or d < result:
                    result = d

            if i == end:
                break
            i = (i + 1) % 360

        return result


def check_lidar(self) -> int:
    """
    라이다 기반 특수 action 판단

    반환:
        0 : 특수상황 없음
        1 : 서행
        2 : 좌측 90도 회전
        3 : 정상 정지
        9 : 긴급 정지
    """

    for _ in range(30):
        self.update_once()

    # 전방 아주 근접하면 긴급 정지
    if self.is_object_in_range(-15, 15, 100):
        return 9

    # stage 0 -> 1 : 좌측 전방 감지 시 서행
    if self.stage == 0:
        if self.is_object_in_range(45, 90, 400):
            self.stage = 1
            return 1

    # stage 1 -> 2 : 좌측 측면 감지 시 좌측 90도 회전
    elif self.stage == 1:
        if self.is_object_in_range(80, 100, 350):
            self.stage = 2
            return 2

    # stage 2 -> 3 : 정면 근접 시 정상 정지
    elif self.stage == 2:
        if self.is_object_in_range(-15, 15, 150):
            self.stage = 3
            return 3

    return 0
