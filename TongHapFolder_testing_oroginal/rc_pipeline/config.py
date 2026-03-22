# rc_pipeline/config.py

LANE_MODEL_PATH = "/home/kyj/dual_origin/0322_RAS_MODEL_EDGETPU.tflite"
OBS_MODEL_PATH = "/home/kyj/dual_origin/Object_detect_edgetpu.tflite"

CAMERA_SOURCE = 0
CORAL_COUNT = 2
USE_EDGETPU = True

CAM_W = 1920
CAM_H = 1080
CAM_FPS = 30

ARDUINO_PORT = "/dev/serial0"
ARDUINO_BAUDRATE = 38400
START_BYTE = 0xAC

LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUDRATE = 115200
LIDAR_TIMEOUT = 0.2
LIDAR_CONF_MIN = 50
LIDAR_DIST_MIN = 100
LIDAR_DIST_MAX = 1000
LIDAR_VALID_TIME = 0.2

SEND_HZ = 20

# 매 프레임 핵심 상태 로그
DEBUG = True

# 10루프 평균 timing 로그
DEBUGAVG = True
