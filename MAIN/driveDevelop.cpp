#include "drive.h"

// 주행 시스템 전체의 현재 상태 저장하는 전역 변수
DriveMode driveMode = MODE_MANUAL;

// 현재 실행 중인 루틴의 상태 저장 변수
static RoutineState routine =
{
    false,  // 현재 루틴 실행 중
    0,      // 현재 단계 인덱스
    0,      // 시작 시간
    0,      // 루틴 길이
    nullptr // 연결된 루틴 배열
};

// ================= 물류 루틴 =================
TimedAction logisticsRoutine[] =
{
    {ACT_FORWARD,0,2000},
    {ACT_REVERSE,0,1000},
    {ACT_STOP,0,0}
};

const int logisticsRoutineLength =
    sizeof(logisticsRoutine) / sizeof(TimedAction);

// ================= 주차 루틴 =================
TimedAction parkingRoutine[] =
{
    {ACT_ROTATE,0,2000},
    {ACT_FORWARD,0,1000},
    {ACT_ROTATE,0,2000},
    {ACT_REVERSE,0,2000},
    {ACT_STOP,0,0}
};

const int parkingRoutineLength =
    sizeof(parkingRoutine) / sizeof(TimedAction);

// ================= 루틴 시작 =================
void startRoutine(TimedAction* r, int length)
{
    // 현재 루틴 실행 중/긴급 정지면 새로운 루틴 시작X
    if(driveMode != MODE_MANUAL)
        return;

    routine.active = true;      //루틴 실행 중
    routine.index = 0;          //0부터 시작
    routine.start = millis();   //현재 단계 시작 시간 저장
    routine.routine = r;        //현재 실행할 루틴 배열
    routine.length = length;    //이 루틴의 전체 길이 저장

    driveMode = MODE_ROUTINE;   //특수 주행 모드로 변경
}

// ================= 루틴 진행 =================
void updateRoutine()
{
    // 특수 주행이 비활성/연결된 루틴 배열X -> 동작X
    if(!routine.active || routine.routine == nullptr)
        return;

    // 배열 범위 보호
    if(routine.index >= routine.length)
    {
        stopMotors();               // 모터 정지
        routine.active = false;     // 특수 주행 루틴 종료
        driveMode = MODE_MANUAL;    // 일반 주행 상태로 복귀
        return;
    }

    unsigned long now = millis();

    TimedAction &act = routine.routine[routine.index];  // 현재 특수 주행 루틴의 단계

    executeBaseAction(act.action, act.angle);           // 현재 단계 행동 시작

    // 종료 단계 검사
    // 특수 주행 루틴 종료 후, 일반 주행 상태 모드로
    if(act.duration == 0)
    {
        routine.active = false;
        driveMode = MODE_MANUAL;
        return;
    }

    // 현재 단계 경과 시간 넘기면 다음 단계로
    if(now - routine.start >= act.duration)
    {
        routine.index++;
        routine.start = now;
    }
}

// ================= 긴급 정지 =================
void emergencyStop()
{
    stopMotors();
    routine.active = false;
    driveMode = MODE_EMERGENCY;
}

// ================= 루틴 취소 =================
// 현재 루틴만 취소
// 일반 주행 상태로 복귀
void cancelRoutine()
{
    routine.active = false;
    driveMode = MODE_MANUAL;
}