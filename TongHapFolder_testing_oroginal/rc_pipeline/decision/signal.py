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
      8  down_t
      9  cross
      10 KNU / box / 물류 pass 계열

    action:
      0 기본
      1 서행
      2 정지
      3 좌측 제자리 90도 회전
      4 우측 제자리 90도 회전 (현재 미사용)
    """
    # angle은 가능하면 항상 숫자로 유지
    if angle is None:
        angle = 0.0

    # 출발 신호는 별도 class 10으로 사용
    if start_signal:
        return 10, 0.0, 0

    # 사람 / 차량은 즉시 우선
    if obj_id in (3, 4):
        return obj_id, 0.0, 0

    # SL / parking / KNU(box/pass 포함)
    if obj_id in (2, 5, 10):
        return obj_id, 0.0, lidar_action

    # 일반 차선 + 라이다 동작 있으면 그대로 반영
    if line_id == 1 and lidar_action != 0:
        return 1, angle, lidar_action

    # 특수 lane 상황
    # left_t / down_t 는 임베디드에서 별도 처리
    if line_id in (6, 8):
        return line_id, angle, 0

    # cross는 직진 유지 성격
    if line_id == 9:
        return 9, angle, lidar_action

    # 일반 차선
    if line_id == 1:
        return 1, angle, lidar_action

    # 차선 없음
    return 0, 0.0, lidar_action
