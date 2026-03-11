#include "drive.h"

DriveMode driveMode = MODE_MANUAL;

// class 0에서 사용할 최근 유효 조향각
static float heldDriveAngle = 0.0f;

// =========================
// 초기화
// =========================
void initDrive()
{
    driveMode = MODE_MANUAL;
    heldDriveAngle = 0.0f;
}

// =========================
// 실제 모터 명령 실행
// =========================
void executeBaseAction(BaseAction act, float angle)
{
    switch (act)
    {
        case ACT_FORWARD:
            applyAngleDrive(angle, 1.0f, 0.0f);
            break;

        case ACT_LEFT:
            applyAngleDrive(angle, 1.0f, 0.0f);
            break;

        case ACT_RIGHT:
            applyAngleDrive(angle, 1.0f, 0.0f);
            break;

        case ACT_ROTATE_L:
            setWheelRPM(-30.0f, 30.0f);
            break;

        case ACT_ROTATE_R:
            setWheelRPM(30.0f, -30.0f);
            break;

        case ACT_REVERSE:
            setWheelRPM(-20.0f, -20.0f);
            break;

        case ACT_STOP:
            stopMotors();
            break;

        case ACT_SLOW:
            applyAngleDrive(angle, 0.5f, 0.0f);
            break;
    }
}

// =========================
// class 0 : 차선 없음
// 직전 조향 유지
// =========================
void handleLineLost()
{
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE)
        return;

    executeBaseAction(ACT_FORWARD, heldDriveAngle);
}

// =========================
// class 1 : 차선 인식
// angle 반영 직진
// =========================
void handleLineFollow(float angle)
{
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE)
        return;

    heldDriveAngle = angle;
    executeBaseAction(ACT_FORWARD, angle);
}

// =========================
// class 7 : 직진
// =========================
void handleStraight(float angle)
{
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE)
        return;

    heldDriveAngle = angle;
    executeBaseAction(ACT_FORWARD, angle);
}

// =========================
// class 6,8 : 좌회전
// =========================
void handleLeftTurn(float angle)
{
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE)
        return;

    executeBaseAction(ACT_LEFT, angle);
}

// =========================
// class 3,4 : 긴급 정지
// =========================
void emergencyStop()
{
    stopMotors();
    cancelRoutine();
    driveMode = MODE_EMERGENCY;
}

void handleEmergencyStop()
{
    emergencyStop();
}

// =========================
// class 9 : 출발 신호
// emergency 상태일 때만 복귀
// =========================
void handleResume()
{
    if (driveMode == MODE_EMERGENCY)
    {
        driveMode = MODE_MANUAL;
        executeBaseAction(ACT_FORWARD, heldDriveAngle);
    }
}

// =========================
// 기본 처리
// =========================
void handleDefault()
{
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE)
        return;

    executeBaseAction(ACT_FORWARD, heldDriveAngle);
}