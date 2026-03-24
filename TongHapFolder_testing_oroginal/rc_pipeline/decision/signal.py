from typing import Optional, Tuple


MAX_DRIVE_ANGLE = 2.0


def clamp_drive_angle(angle: float) -> float:
    if angle < -MAX_DRIVE_ANGLE:
        return -MAX_DRIVE_ANGLE
    if angle > MAX_DRIVE_ANGLE:
        return MAX_DRIVE_ANGLE
    return angle


def signal_det(
    obj_id: Optional[int],
    line_id: Optional[int],
    angle: Optional[float],
    lidar_action: int,
    start_signal: bool = False,
) -> Tuple[int, Optional[float], int]:
    """
    class:
      0  lane_lost
      1  normal_lane
      2  SL
      3  person
      4  car
      5  parking
      6  left_t
      8  down_t
      9  cross
      10 KNU / box / logistics-pass family

    action:
      0 default
      1 slow
      2 rotate_left
      3 stop_then_go
      4 special_routine
    """
    if angle is None:
        angle = 0.0

    if start_signal:
        return 9, 0.0, 0

    if obj_id in (3, 4):
        return obj_id, 0.0, 0

    if obj_id in (2, 5, 10):
        return obj_id, 0.0, lidar_action

    if line_id == 1 and lidar_action != 0:
        return 1, clamp_drive_angle(angle), lidar_action

    # Ignore class 7 while a lidar action is active so it cannot interrupt action handling.
    if line_id == 7 and lidar_action != 0:
        return 0, 0.0, lidar_action

    if line_id in (6, 7, 8):
        return line_id, 0, 0

    if line_id == 9:
        return 9, 0, 0

    if line_id == 1:
        return 1, clamp_drive_angle(angle), lidar_action

    return 0, 0.0, lidar_action
