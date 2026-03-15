from typing import Optional, Tuple


def signal_det(
    obj_id: Optional[int],
    line_id: Optional[int],
    angle: Optional[float],
    lidar_action: int,
    start_signal: bool = False,
) -> Tuple[int, Optional[float], int]:
    """
    class:
      0  차선 없음
      1  일반 차선
      2  SL
      3  person
      4  car
      5  parking
      6  left_t
      7  right_t
      8  down_t
      9  cross
      10 출발 신호

    action:
      0 기본
      1 서행
      2 정지
      3 좌측 제자리 90도 회전
      4 우측 제자리 90도 회전
    """

    if obj_id in (3, 4):
        return obj_id, None, 0

    if start_signal:
        return 10, None, 0

    if obj_id in (2, 5):
        return obj_id, None, lidar_action

    if line_id == 1 and lidar_action != 0:
        return 1, angle, lidar_action

    if line_id in (6, 7, 8, 9):
        return line_id, None, 0

    if line_id == 1:
        return 1, angle, 0

    return 0, None, 0
