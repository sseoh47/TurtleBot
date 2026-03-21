from typing import Optional


def convert_lane_result(p_le, p_ls, p_is, p_it):
    """
    내부 lane 결과를 최종 lane class 체계로 변환
      0 = 차선 없음
      1 = 일반 차선
      6 = left_t
      7 = right_t
      8 = down_t
      9 = cross
    """

    inter_map = {
        "left_t": 6,
        "right_t": 7,
        "down_t": 8,
        "cross": 9,
    }

    if p_ls == "lost":
        return 0, None

    if p_is and p_it in inter_map:
        return inter_map[p_it], None

    return 1, float(p_le)


def convert_object_result(obs_list) -> Optional[int]:
    """
    내부 객체 결과를 최종 obj class 체계로 변환
      SL      -> 2
      person  -> 3
      car     -> 4
      parking -> 5
      없음    -> None
    """
    if not obs_list:
        return None

    name = obs_list[0]["class_name"]
    obj_map = {
        "SL": 2,
        "person": 3,
        "car": 4,
        "parking": 5,
    }
    return obj_map.get(name)
