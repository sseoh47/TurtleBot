#include "drive.h"

DriveMode driveMode = MODE_MANUAL;

// class 0에서 사용할 최근 유효 조향각
static float heldDriveAngle = 0.0f;
static bool class7StraightActive = false;
static unsigned long class7StraightStart = 0;
static const unsigned long class7StraightDuration = 1000;

// =========================
// 초기화
// =========================
void initDrive()
{
    driveMode = MODE_MANUAL;
    heldDriveAngle = 0.0f;
    class7StraightActive = false;
    class7StraightStart = 0;
}

// =========================
// 조향 정규화
// =========================
float normalizeLineAngle(float rawAngle)
{
    // min, max를 기준으로 정규화 하는 부분 : -100 ~ 100 -> -10 ~ 10
    rawAngle /= 3.4f; // 10 <- 이부분은 라즈베리에서 최대값 최소값 고려해서 변경해야함.

    // 제한 거는 부분
    if (rawAngle < -30.0f || rawAngle > 30.0f)
        return 0.0f;
    if (rawAngle < -10.0f)
        return -10.0f;
    if (rawAngle > 10.0f)
        return 10.0f;
    return rawAngle;
}


// =========================
// 실제 모터 명령 실행
// =========================
void executeBaseAction(BaseAction act, float angle, float speedOffset)
{
    switch (act)
    {
        case ACT_FORWARD:
            applyAngleDrive(angle, 1.0f, 0.0f, speedOffset);
            break;

        case ACT_LEFT:
            applyAngleDrive(angle, 1.0f, 0.0f, speedOffset);
            break;

        case ACT_RIGHT:
            applyAngleDrive(angle, 1.0f, 0.0f, speedOffset);
            break;

        case ACT_ROTATE_L:
            setWheelRPM(-15.0f, 15.0f);
            break;

        case ACT_ROTATE_R:
            setWheelRPM(15.0f, -15.0f);
            break;

        case ACT_REVERSE:
            setWheelRPM(-20.0f, -20.0f);
            break;

        case ACT_STOP:
            stopMotors();
            break;

        case ACT_SLOW:
            applyAngleDrive(angle, 0.5f, 0.0f, speedOffset);
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

    executeBaseAction(ACT_SLOW, heldDriveAngle);
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
    executeBaseAction(ACT_SLOW, angle, 3.0f);
}

// =========================
// class 6,8:좌회전 / 7:직진 통합
// =========================
bool timedActionActive = false;     // 시간 행동 진행 여부
bool timedActionWait = false;

unsigned long actionStart = 0;      // 시작 시각
unsigned long actionDuration = 0;   // 유지 시간(ms)
unsigned long waitDuration=3100;

BaseAction currentAction;           // ACT_LEFT / ACT_FORWARD ...
float currentAngle = 0;             // 조향각

void handleTimedAction(BaseAction act, float angle, unsigned long duration)
{
    // 비상/루틴 중이면 실행 안 함
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE || driveMode == MODE_LOGISTICIn)
        return;

    // 처음 진입 시 시작 세팅
    if (!timedActionActive && !timedActionWait)
    {
        timedActionWait = true;
        actionStart = millis();
        actionDuration = duration;
        currentAction = act;
        currentAngle = angle;
        return;
    }

    // 1단계: 대기
    if (timedActionWait)
    {
        // 대기 중에는 기존 차선 주행 유지
        executeBaseAction(ACT_FORWARD, 0);

        if (millis() - actionStart >= waitDuration)
        {
            timedActionWait = false;
            timedActionActive = true;
            actionStart = millis();   // 이제 실제 행동 시작 시각으로 다시 저장
        }
        return;
    }

    // 2단계: 실제 행동 수행
    if (timedActionActive)
    {
        executeBaseAction(currentAction, currentAngle);

        if (millis() - actionStart >= actionDuration)
        {
            timedActionActive = false;
        }
    }
}

bool isClass7StraightActive()
{
    if (class7StraightActive && millis() - class7StraightStart >= class7StraightDuration)
    {
        class7StraightActive = false;
    }

    return class7StraightActive;
}

void cancelClass7Straight()
{
    class7StraightActive = false;
    class7StraightStart = 0;
}

void handleClass7Straight()
{
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE || driveMode == MODE_LOGISTICIn)
        return;

    if (!isClass7StraightActive())
    {
        class7StraightActive = true;
        class7StraightStart = millis();
    }

    executeBaseAction(ACT_FORWARD, 0.0f);

    if (!isClass7StraightActive())
    {
        cancelClass7Straight();
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
}

// =========================
// 기본 처리
// =========================
void handleDefault()
{
    if (driveMode == MODE_EMERGENCY || driveMode == MODE_ROUTINE || driveMode == MODE_LOGISTICIn)
        return;

    executeBaseAction(ACT_FORWARD, heldDriveAngle);
}
