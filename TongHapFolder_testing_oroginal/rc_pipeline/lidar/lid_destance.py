#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from typing import Iterator, List, Tuple

from lds02 import LDS02, angle_to_idx


SECTORS = [
    ("d1", 45, 90, 400),
    ("d2", 95, 110, 300),
    ("d3", -20, 0, 300),
    ("d4", -15, 15, 250),
]


def iter_angle_range(min_angle: float, max_angle: float) -> Iterator[int]:
    start = angle_to_idx(min_angle)
    end = angle_to_idx(max_angle)
    idx = start

    while True:
        yield idx
        if idx == end:
            break
        idx = (idx + 1) % 360


def idx_to_signed_angle(idx: int) -> int:
    return idx if idx <= 180 else idx - 360


def get_recent_points(lidar: LDS02, min_angle: float, max_angle: float) -> List[Tuple[int, int]]:
    now = time.monotonic()
    points: List[Tuple[int, int]] = []

    for idx in iter_angle_range(min_angle, max_angle):
        dist = lidar.ranges[idx]
        ts = lidar.timestamps[idx]
        if dist is None:
            continue
        if (now - ts) > lidar.valid_time:
            continue

        points.append((idx_to_signed_angle(idx), dist))

    return points


def format_points(points: List[Tuple[int, int]]) -> str:
    if not points:
        return "none"
    return " ".join(f"{angle}:{dist}" for angle, dist in points)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print LDS02 distances using the same angle windows as lds02.py"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--conf-min", type=int, default=50)
    parser.add_argument("--dist-min", type=int, default=100)
    parser.add_argument("--dist-max", type=int, default=1000)
    parser.add_argument("--valid-time", type=float, default=0.5)
    parser.add_argument(
        "--packets",
        type=int,
        default=30,
        help="how many lidar packets to read before printing one report",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="sleep between reports in seconds",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    lidar = LDS02(
        port=args.port,
        baud=args.baud,
        timeout=args.timeout,
        conf_min=args.conf_min,
        dist_min=args.dist_min,
        dist_max=args.dist_max,
        valid_time=args.valid_time,
    )

    try:
        while True:
            try:
                for _ in range(args.packets):
                    lidar.update_once()
            except Exception as exc:
                print(f"[LIDAR] read error: {exc}")
                time.sleep(args.interval)
                continue

            timestamp = time.strftime("%H:%M:%S")
            print(f"\n[{timestamp}] port={args.port} valid_time={lidar.valid_time:.2f}s")

            for name, min_angle, max_angle, threshold in SECTORS:
                min_dist = lidar.get_min_distance(min_angle, max_angle)
                detected = lidar.is_object_in_range(min_angle, max_angle, threshold)
                points = get_recent_points(lidar, min_angle, max_angle)

                print(
                    f"{name}: angles={min_angle}..{max_angle} "
                    f"threshold={threshold}mm detected={detected} min={min_dist}"
                )
                print(f"  points: {format_points(points)}")

            if args.once:
                break

            time.sleep(args.interval)
    finally:
        lidar.close()


if __name__ == "__main__":
    main()
