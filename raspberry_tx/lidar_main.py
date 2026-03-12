# ### =============================================================
# ## 실험용 엔트리!!!!!!!!!!!!!!! 메인은 딱히 필요없음~
# ### =============================================================
# from lidar_func import LDS02
# from lidar_func import signal_det
# from lidar_func import PerceptionStabilizer

# # 메인 실행 루프: check_lidar() 상태 코드를 계속 조회한다.
# lidar = LDS02("/dev/ttyUSB0", 115200)

# stabilizer = PerceptionStabilizer()

# try:
#     while True:
#         # 예시: 모델 결과
#         raw_obj_id = 2
#         raw_line_id = 1
#         raw_angle = -12.3

#         # 1) 라이다 특수상황(action)
#         lidar_action = lidar.check_lidar()

#         # 2) 모델 결과 안정화
#         obj_id, line_id, angle = stabilizer.update(raw_obj_id, raw_line_id, raw_angle)

#         # 3) signal_det
#         final_class, final_angle, final_action = signal_det(
#             obj_id=obj_id,
#             line_id=line_id,
#             angle=angle,
#             lidar_distance=300,  # 네가 따로 계산한 최소거리 넣기
#             lidar_check=lidar_action,  # action 역할
#         )

#         print(final_class, final_angle, final_action)

# finally:
#     lidar.close()
### =============================================================
## 실험용 엔트리
# ### =============================================================
# from lidar_func import LDS02, PerceptionStabilizer, signal_det

# lidar = LDS02("COM5", 115200)   # 윈도우 예시
# stabilizer = PerceptionStabilizer(threshold=2)

# while True:
#     raw_obj_id = None
#     raw_line_id = 1
#     raw_angle = -8.5
#     start_signal = False

#     obj_id, line_id, angle = stabilizer.update(
#         raw_obj_id,
#         raw_line_id,
#         raw_angle,
#     )

#     lidar_action = lidar.check_lidar()

#     final_class, final_angle, final_action = signal_det(
#         obj_id=obj_id,
#         line_id=line_id,
#         angle=angle,
#         lidar_action=lidar_action,
#         start_signal=start_signal,
#     )

#     print(final_class, final_angle, final_action)

from lidar_func import signal_det

# 상황:
# - 객체는 아직 카메라가 못 봄
# - 차선은 일반 차선 인식 중
# - 조향 오차는 -8.5
# - 라이다는 먼저 서행 상황(action 1) 판단
# - 출발 신호 없음

final_class, final_angle, final_action = signal_det(
    obj_id=None, line_id=1, angle=-8.5, lidar_action=1, start_signal=False
)

print("final_class =", final_class)
print("final_angle =", final_angle)
print("final_action =", final_action)
from lidar_func import signal_det

# 상황:
# - 객체는 아직 카메라가 못 봄
# - 차선은 일반 차선 인식 중
# - 조향 오차는 -8.5
# - 라이다는 먼저 서행 상황(action 1) 판단
# - 출발 신호 없음

final_class, final_angle, final_action = signal_det(
    obj_id=None, line_id=1, angle=-8.5, lidar_action=1, start_signal=False
)

print("final_class =", final_class)
print("final_angle =", final_angle)
print("final_action =", final_action)
print("=========================")

print(signal_det(obj_id=3, line_id=1, angle=-5.0, lidar_action=0, start_signal=False))
print(signal_det(obj_id=2, line_id=1, angle=-5.0, lidar_action=1, start_signal=False))
print(signal_det(obj_id=None, line_id=0, angle=None, lidar_action=0, start_signal=True))

print(
    signal_det(obj_id=None, line_id=0, angle=None, lidar_action=0, start_signal=False)
)
