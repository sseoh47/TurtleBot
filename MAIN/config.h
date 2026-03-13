#ifndef CONFIG_H
#define CONFIG_H

// ===================== Communication =================
#define COMM_BAUDRATE 57600 // 57600

// ===================== Dynamixel =====================
#define DXL_BAUDRATE 1000000
#define DXL_PROTOCOL_VERSION 2.0

#define DXL_DIR_PIN 2 // 다이나믹셀실드 방향핀
//#define DXL_DIR_PIN 84 // OpenCR 방향핀

#define LEFT_ID  1
#define RIGHT_ID 2

// ===================== Drive Parameters ===============
#define BASE_RPM 30.0f
#define MAX_RPM 60.0f
#define MIN_RPM -60.0f

#define TURN_GAIN_RPM_PER_DEG 0.6f
#define ANGLE_LIMIT_DEG 30.0f


#define RX_LED 4

#endif