#include "motor.h"
#include "config.h"

#include <Dynamixel2Arduino.h>
#include <Arduino.h>

// #define DXL_SERIAL Serial   // 다이나믹실드
#define DXL_SERIAL Serial3

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

void initMotor()
{
    dxl.begin(DXL_BAUDRATE);
    dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

    dxl.torqueOff(LEFT_ID);
    dxl.torqueOff(RIGHT_ID);

    dxl.setOperatingMode(LEFT_ID,OP_VELOCITY);
    dxl.setOperatingMode(RIGHT_ID,OP_VELOCITY);

    dxl.torqueOn(LEFT_ID);
    dxl.torqueOn(RIGHT_ID);

    stopMotors();
}

static float clampf(float v,float min,float max)
{
    if(v<min) return min;
    if(v>max) return max;
    return v;
}

void setWheelRPM(float left,float right)
{
    left = clampf(left,MIN_RPM,MAX_RPM);
    right = clampf(right,MIN_RPM,MAX_RPM);

    dxl.setGoalVelocity(LEFT_ID,left,UNIT_RPM);
    dxl.setGoalVelocity(RIGHT_ID,right,UNIT_RPM);
}

void stopMotors()
{
    setWheelRPM(0,0);
}

void applyAngleDrive(float angleDeg,float speedScale,float bias)
{
    float angle = clampf(angleDeg, -ANGLE_LIMIT_DEG, ANGLE_LIMIT_DEG);

    float base = BASE_RPM * speedScale;
    float diff = angle * TURN_GAIN_RPM_PER_DEG + bias;

    float left  = base + diff;
    float right = base - diff;

    setWheelRPM(left, right);
}