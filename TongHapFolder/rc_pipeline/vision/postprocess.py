from typing import Optional


def convert_lane_result(p_le, p_ls, p_is, p_it):
    """
    내부 lane 결과를 최종 lane class 체계로 변환
      0 = 차선 없음
      1 = 일반 차선
      6 = left_t
      8 = down_t
      9 = cross
      10 = 물류 pass (현재 미사용, 필요시 확장)
    """
    inter_map = {
        "left_t": 6,
        "down_t": 8,
        "cross": 9,
        # "pass": 10,  # 필요할 때 사용
    }

    angle = float(p_le)

    if p_is and p_it in inter_map:
        return inter_map[p_it], angle

    if p_ls == "lost":
        return 0, 0.0

    return 1, angle


def convert_object_result(obs_list) -> Optional[int]:
    """
    내부 객체 결과를 최종 obj class 체계로 변환
      2  = SL
      3  = person
      4  = car
      5  = parking
      10 = 물류 pass (현재 미사용, 필요시 확장)
      없음 = None
    """
    if not obs_list:
        return None

    name = obs_list[0]["class_name"]
    obj_map = {
        "SL": 2,
        "person": 3,
        "car": 4,
        "parking": 5,
        "box": 10,
        "KNU": 10,
    }
    return obj_map.get(name)
