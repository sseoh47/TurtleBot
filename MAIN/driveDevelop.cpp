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
static bool rotateActionDone = false;
static unsigned long rotateActionStart = 0;
static const unsigned long ROTATE_90_MS = 2000;   // 실차 튜닝 필요

// ================= 물류 루틴 IN =================
TimedAction logisticsRoutineIn[] =
{
    {ACT_ROTATE_R, 0.0f, 2000, 0.0f},
    {ACT_FORWARD, 0.0f, 2000, -10.0f},
    {ACT_ROTATE_R, 0.0f, 2000, 0.0f},
    {ACT_REVERSE, 0.0f, 2000, -10.0f},
    {ACT_STOP,    0.0f,    0}
};

const int logisticsRoutineInLength =
    sizeof(logisticsRoutineIn) / sizeof(TimedAction);

// ================= 물류 루틴 OUT =================
TimedAction logisticsRoutineOut[] =
{
    {ACT_FORWARD, 0.0f, 3000, -10.0f},
    {ACT_ROTATE_L, 0.0f, 2000, 0.0f},
    {ACT_STOP,    0.0f,    0}
};

const int logisticsRoutineOutLength =
    sizeof(logisticsRoutineOut) / sizeof(TimedAction);

// ================= 물류X PASS 루틴 =================
TimedAction logisticsRoutinePASS[] =
{
    {ACT_REVERSE, 0.0f, 1500, -10.0f},
    {ACT_ROTATE_R, 0.0f, 2000, 0.0f},
    {ACT_STOP,    0.0f,    0}
};

const int logisticsRoutinePASSLength =
    sizeof(logisticsRoutinePASS) / sizeof(TimedAction);

// ================= 주차 루틴 =================
TimedAction parkingRoutine[] =
{
    {ACT_ROTATE_R, 0.0f, 2000, 0.0f},
    {ACT_FORWARD, 0.0f, 2000, -10.0f},
    {ACT_ROTATE_R, 0.0f, 2000, 0.0f},
    {ACT_REVERSE, 0.0f, 2000, -10.0f},
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
    if ((driveMode != MODE_MANUAL ) && (driveMode != MODE_LOGISTICIn))
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

    //driveMode = MODE_MANUAL;
}

// ================= 루틴 갱신 =================
void updateRoutine()
{
    if (!routine.active || routine.routine == nullptr)
        return;

    if (routine.index >= routine.length)
    {
        TimedAction* finished = routine.routine;

        stopMotors();
        cancelRoutine();

        if (finished == logisticsRoutineIn)
            driveMode = MODE_LOGISTICIn;   // IN 끝났으니 대기 모드
        else
            driveMode = MODE_MANUAL;       // OUT / PASS / PARKING 끝났으니 일반 모드

        return;

        // stopMotors();
        // cancelRoutine();
        // return;
    }

    unsigned long now = millis();
    TimedAction &act = routine.routine[routine.index];

    executeBaseAction(act.action, act.angle, act.speedOffset);

    // duration 0이면 종료 단계
    if (act.duration == 0)
    {
        TimedAction* finished = routine.routine;

        stopMotors();
        cancelRoutine();

        if (finished == logisticsRoutineIn)
            driveMode = MODE_LOGISTICIn;
        else
            driveMode = MODE_MANUAL;

        return;
        
        // cancelRoutine();
        // return;
    }

    if (now - routine.start >= act.duration)
    {
        routine.index++;
        routine.start = now;
    }
}

// ================= class 1,2,5,10 특수 처리 =================
void handleSpecialTarget(int classId, float angle, int action)
{
    if (driveMode == MODE_EMERGENCY)
        return;

    // 루틴 중이면 일반 특수 처리 무시
    if (driveMode == MODE_ROUTINE || driveMode == MODE_LOGISTICIn )
        return;

    if (action != 2)
    {
        rotateActionActive = false;
        rotateActionDone = false;
    }

    switch (action)
    {
        case 1:
            // action1 : 서행
            executeBaseAction(ACT_SLOW, angle);
            break;

        case 2:
            // action2 : 좌측 90도 회전(시간 기반)
            // 정지 + 90도 + 서행직진(angle=0)
            if (!rotateActionActive && !rotateActionDone)
            {
                rotateActionActive = true;
                rotateActionStart = millis();
                executeBaseAction(ACT_STOP, 0.0f);
            }
            if (rotateActionActive)
            {
                if (millis() - rotateActionStart < ROTATE_90_MS) // ROTATE_90_MS==2000
                {
                    executeBaseAction(ACT_ROTATE_L, 0.0f);
                }
                else if (millis() - rotateActionStart < ROTATE_90_MS+500) // ROTATE_90_MS==2500 -> 약 500ms동안 정지
                {
                    executeBaseAction(ACT_STOP, 0.0f);
                }
                else
                {
                    rotateActionActive = false;
                    rotateActionDone = true;
                }
            }
            else if (rotateActionDone)
            {
                rotateActionActive = false;
                executeBaseAction(ACT_SLOW, 0);
            }
            break;

        case 3:
        case 9:
            // action3 : 정면 근접 정지
            executeBaseAction(ACT_STOP, 0.0f);
            if (classId == 2)
            {
                //driveMode = MODE_LOGISTICIn;
                startRoutine(logisticsRoutineIn, logisticsRoutineInLength);
            }
            else if (classId == 5)
                startRoutine(parkingRoutine, parkingRoutineLength);
            else if (classId == 10)
                startRoutine(logisticsRoutinePASS, logisticsRoutinePASSLength);
            break;

        default:
            break;
    }
}