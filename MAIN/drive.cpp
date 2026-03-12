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
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE || driveMode == MODE_LOGISTICIn)
        return;

    executeBaseAction(ACT_FORWARD, heldDriveAngle);
}

// =========================
// class 1 : 차선 인식
// angle 반영 직진
// =========================
void handleLineFollow(float angle)
{
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE || driveMode == MODE_LOGISTICIn)
        return;

    heldDriveAngle = angle;
    executeBaseAction(ACT_FORWARD, angle);
}

// =========================
// class 6,8:좌회전 / 7:직진 통합
// =========================
bool timedActionActive = false;     // 시간 행동 진행 여부
unsigned long actionStart = 0;      // 시작 시각
unsigned long actionDuration = 0;   // 유지 시간(ms)

BaseAction currentAction;           // ACT_LEFT / ACT_FORWARD ...
float currentAngle = 0;             // 조향각

void handleTimedAction(BaseAction act, float angle, unsigned long duration)
{
    // 비상/루틴 중이면 실행 안 함
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE || driveMode == MODE_LOGISTICIn)
        return;

    // 처음 진입 시 시작 세팅
    if (!timedActionActive)
    {
        timedActionActive = true;
        actionStart = millis();
        actionDuration = duration;
        currentAction = act;
        currentAngle = angle;
    }

    // 행동 유지
    executeBaseAction(currentAction, currentAngle);

    // 시간 종료 체크
    if (millis() - actionStart >= actionDuration)
    {
        timedActionActive = false;
        // executeBaseAction(ACT_FORWARD, 0);   // 기본 직진 복귀
    }
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
    startRoutine(logisticsRoutineOut, logisticsRoutineOutLength);
    driveMode = MODE_MANUAL;
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