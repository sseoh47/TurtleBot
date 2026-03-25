#include "drive.h"
//테스트
static unsigned long stopStart = 0;
static const unsigned long STOP_DURATION = 1000; // 3000ms 정지

// ================= setup 시작 루틴  =================
void updateStartupRoutine()
{
    unsigned long now = millis();

    switch (startupStep)
    {
        case 0:
            executeBaseAction(ACT_FORWARD, 0);
            if (now - startupStepStart >= 1500)
            {
                startupStep = 1;
                startupStepStart = now;
            }
            break;

        case 1:
            executeBaseAction(ACT_LEFT, -30);
            if (now - startupStepStart >= 2000)
            {
                startupStep = 2;
                startupStepStart = now;
            }
            break;

        case 2:
            executeBaseAction(ACT_FORWARD, 0);
            if (now - startupStepStart >= 2500)
            {
                startupStep = 3;
                startupStepStart = now;
            }
            break;

        case 3:
            executeBaseAction(ACT_LEFT, -30);
            if (now - startupStepStart >= 2000)
            {
                executeBaseAction(ACT_STOP, 0);
                startupRoutineActive = false;
            }
            break;

        // default:
        //     executeBaseAction(ACT_STOP, 0);
        //     startupRoutineActive = false;
        //     break;
    }
}


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
// static bool rotateActionActive = false;
// static bool rotateActionDone = false;
// static unsigned long rotateActionStart = 0;
static bool slowActionActive = false;
// static unsigned long slowActionStart = 0;
// static const unsigned long ROTATE_90_MS = 2000;   // 실차 튜닝 필요

// ================= 물류 루틴 IN =================
TimedAction logisticsRoutineIn[] =
{
    {ACT_ROTATE_R, 0.0f, 2450, 0.0f},
    {ACT_FORWARD, 0.0f, 2500, -10.0f},
    {ACT_ROTATE_R, 0.0f, 2450, 0.0f},
    {ACT_REVERSE, 0.0f, 3000, -10.0f},
    {ACT_STOP,    0.0f,    0}
};

const int logisticsRoutineInLength =
    sizeof(logisticsRoutineIn) / sizeof(TimedAction);

// ================= 물류 루틴 OUT =================
TimedAction logisticsRoutineOut[] =
{
    {ACT_FORWARD, 0.0f, 2500, -10.0f},
    {ACT_ROTATE_L, 0.0f, 2450, 0.0f},
    {ACT_STOP,    0.0f,    0}
};

const int logisticsRoutineOutLength =
    sizeof(logisticsRoutineOut) / sizeof(TimedAction);

// ================= 물류X PASS 루틴 =================
TimedAction logisticsRoutinePASS[] =
{
    {ACT_REVERSE, 0.0f, 500, -10.0f},
    {ACT_ROTATE_R, 0.0f, 2450, 0.0f},
    {ACT_STOP,    0.0f,    0}
};

const int logisticsRoutinePASSLength =
    sizeof(logisticsRoutinePASS) / sizeof(TimedAction);

// ================= 주차 루틴 =================
TimedAction parkingRoutine[] =
{
    {ACT_ROTATE_L, 0.0f, 2450, 0.0f},
    {ACT_FORWARD, 0.0f, 3500, -10.0f},
    {ACT_ROTATE_L, 0.0f, 2450, 0.0f},
    {ACT_REVERSE, 0.0f, 4500, -10.0f},
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

    slowActionActive = false;

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

        if (finished == parkingRoutine)
        {   
            while (true)
            {
                stopMotors();
            }
        }
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

        if (finished == parkingRoutine)
        {
            while (true)
            {
                stopMotors();
            }
        }

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
    // 루틴 중이면 일반 특수 처리 무시
    if (driveMode == MODE_ROUTINE || driveMode == MODE_LOGISTICIn || driveMode == MODE_EMERGENCY)
        return;

    if (action != 4)
    {
        slowActionActive = false;
    }

    switch (action)
    {
        case 1:
            // action1 : 서행
            executeBaseAction(ACT_SLOW, angle);
            break;

        case 2:
            executeBaseAction(ACT_ROTATE_L, 0.0f);
            break;
            
        case 3:
            executeBaseAction(ACT_SLOW, angle);
            break;


        case 4:
            if (!slowActionActive)
            {
                slowActionActive = true;
                stopStart = millis();
            }

            if (millis() - stopStart < STOP_DURATION)
            {
                executeBaseAction(ACT_STOP, 0.0f);
                return;
            }

            slowActionActive = false;
            // action4 : 정면 근접 정지
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
            else
                startRoutine(logisticsRoutinePASS, logisticsRoutinePASSLength);
                //executeBaseAction(ACT_REVERSE, 0.0f);
            break;

        default:
            break;
    }
}
