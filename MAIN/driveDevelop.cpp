#include "drive.h"

// ================= 루틴 상태 =================
static RoutineState routine =
{
    false,
    0,
    0,
    0,
    nullptr
};

// ================= action 2 회전 상태 =================
static bool rotateActionActive = false;
static unsigned long rotateActionStart = 0;
static const unsigned long ROTATE_90_MS = 2000;   // 실차 튜닝 필요

// ================= 물류 루틴 =================
TimedAction logisticsRoutine[] =
{
    {ACT_FORWARD, 0.0f, 1500},
    {ACT_LEFT,    0.0f, 1200},
    {ACT_STOP,    0.0f,    0}
};

const int logisticsRoutineLength =
    sizeof(logisticsRoutine) / sizeof(TimedAction);

// ================= 주차 루틴 =================
TimedAction parkingRoutine[] =
{
    {ACT_ROTATE,  0.0f, 1500},
    {ACT_FORWARD, 0.0f, 1000},
    {ACT_REVERSE, 0.0f, 1200},
    {ACT_STOP,    0.0f,    0}
};

const int parkingRoutineLength =
    sizeof(parkingRoutine) / sizeof(TimedAction);

// ================= 루틴 실행 여부 =================
bool isRoutineActive()
{
    return routine.active;
}

// ================= 루틴 시작 =================
void startRoutine(TimedAction* r, int length)
{
    if (driveMode != MODE_MANUAL)
        return;

    routine.active = true;
    routine.index = 0;
    routine.start = millis();
    routine.routine = r;
    routine.length = length;

    driveMode = MODE_ROUTINE;
}

// ================= 루틴 취소 =================
void cancelRoutine()
{
    routine.active = false;
    routine.index = 0;
    routine.start = 0;
    routine.length = 0;
    routine.routine = nullptr;

    rotateActionActive = false;

    if (driveMode == MODE_ROUTINE)
        driveMode = MODE_MANUAL;
}

// ================= 루틴 갱신 =================
void updateRoutine()
{
    if (!routine.active || routine.routine == nullptr)
        return;

    if (routine.index >= routine.length)
    {
        stopMotors();
        cancelRoutine();
        return;
    }

    unsigned long now = millis();
    TimedAction &act = routine.routine[routine.index];

    executeBaseAction(act.action, act.angle);

    // duration 0이면 종료 단계
    if (act.duration == 0)
    {
        cancelRoutine();
        return;
    }

    if (now - routine.start >= act.duration)
    {
        routine.index++;
        routine.start = now;
    }
}

// ================= class 2,5 특수 처리 =================
void handleSpecialTarget(int classId, float angle, int action)
{
    if (driveMode == MODE_EMERGENCY)
        return;

    // 루틴 중이면 일반 특수 처리 무시
    if (driveMode == MODE_ROUTINE)
        return;

    switch (action)
    {
        case 1:
            // action1 : 서행
            executeBaseAction(ACT_SLOW, angle);
            break;

        case 2:
            // action2 : 좌측 90도 회전(시간 기반)
            if (!rotateActionActive)
            {
                rotateActionActive = true;
                rotateActionStart = millis();
            }

            if (millis() - rotateActionStart < ROTATE_90_MS)
            {
                executeBaseAction(ACT_ROTATE, 0.0f);
            }
            else
            {
                rotateActionActive = false;
                executeBaseAction(ACT_STOP, 0.0f);
            }
            break;

        case 3:
            // action3 : 정면 근접 정지
            executeBaseAction(ACT_STOP, 0.0f);
            break;

        case 4:
            // action4 : 하드코딩된 루틴 수행
            if (classId == 2)
                startRoutine(logisticsRoutine, logisticsRoutineLength);
            else if (classId == 5)
                startRoutine(parkingRoutine, parkingRoutineLength);
            break;

        case 9:
            // 현재 비워둠
            break;

        default:
            // 특수 action 없으면 기본 접근 주행
            executeBaseAction(ACT_FORWARD, angle);
            break;
    }
}