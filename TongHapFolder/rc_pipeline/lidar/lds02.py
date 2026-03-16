## ====================================================================
## 서범님 코드 발췌
## ====================================================================

import time
from typing import Optional

import serial


HEADER = 0x54
LENGTH = 0x2C
PACKET_SIZE = 47


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


class LDS02:
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

        self.stage = 0

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    def reset_stage(self):
        self.stage = 0

    def _read_exact(self, n: int) -> bytes:
        data = self.ser.read(n)
        if len(data) != n:
            raise TimeoutError(f"expected {n} bytes, got {len(data)}")
        return data

    def update_once(self):
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

    def get_min_distance(self, min_angle: float, max_angle: float):
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

    def is_object_in_range(
        self, min_angle: float, max_angle: float, threshold_mm: int
    ) -> bool:
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

    def check_action(self) -> int:
        """
        action:
          0 기본
          1 서행
          2 왼쪽 제자리회전
          3 1초 정지 후 천천히 전진
          4 정지
        """
        try:
            for _ in range(10):
                self.update_once()
        except Exception as e:
            print(f"[lidar] update Error")
            return 0
        d1 = self.is_object_in_range(45, 90, 400)
        d2 = self.is_object_in_range(90, 100, 350)
        d3 = self.is_object_in_range(-15, 0, 200)
        d4 = self.is_object_in_range(-15, 15, 200)

        m1 = self.get_min_distance(45, 90)
        m2 = self.get_min_distance(90, 100)
        m3 = self.get_min_distance(-15, 0)
        m4 = self.get_min_distance(-15, 15)

        print(f"[LIDAR] stage={self.stage} m1={m1} m2={m2} m3={m3} m4={m4}")

        print(f"[LIDAR] stage={self.stage} d1={d1} d2={d2} d3={d3} d4={d4}")

        if self.stage == 0:
            if self.is_object_in_range(45, 90, 400):
                self.stage = 1
                return 1

        elif self.stage == 1:
            if self.is_object_in_range(90, 100, 350):
                self.stage = 2
                return 2

        elif self.stage == 2:
            if self.is_object_in_range(-15, 0, 200):
                self.stage = 3
                return 3

        elif self.stage == 3:
            if self.is_object_in_range(-15, 15, 200):
                self.stage = 0
                return 4

        return 0
