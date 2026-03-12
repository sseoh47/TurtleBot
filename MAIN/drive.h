#ifndef DRIVE_H
#define DRIVE_H

#include <Arduino.h>
#include "motor.h"

// ================= 기본 주행 종류 =================
enum BaseAction
{
    ACT_FORWARD,
    ACT_LEFT,
    ACT_RIGHT,
    ACT_ROTATE_L,
    ACT_ROTATE_R,
    ACT_REVERSE,
    ACT_STOP,
    ACT_SLOW
};

// ================= 주행 모드 =================
enum DriveMode
{
    MODE_MANUAL,
    MODE_ROUTINE,
    MODE_EMERGENCY,
    MODE_LOGISTICIn
};

// ================= 루틴 동작 구조 =================
struct TimedAction
{
    BaseAction action;
    float angle;
    unsigned long duration;
    float speedOffset;
};

// ================= 루틴 상태 =================
struct RoutineState
{
    bool active;
    int index;
    unsigned long start;
    int length;
    TimedAction* routine;
};

// ================= 전역 =================
extern DriveMode driveMode;

extern TimedAction logisticsRoutineIn[];
extern TimedAction logisticsRoutineOut[];
extern TimedAction logisticsRoutinePASS[];
extern TimedAction parkingRoutine[];

extern const int logisticsRoutineInLength;
extern const int logisticsRoutineOutLength;
extern const int logisticsRoutinePASSLength;
extern const int parkingRoutineLength;

extern bool timedActionActive;
extern unsigned long actionStart;
extern unsigned long actionDuration;

extern BaseAction currentAction;
extern float currentAngle;

// ================= 초기화 =================
void initDrive();

// ================= 일반 주행 처리 =================
void handleLineLost();
void handleLineFollow(float angle);
// void handleStraight(float angle);
// void handleLeftTurn(float angle);
void handleTimedAction(BaseAction act, float angle, unsigned long duration);

void handleEmergencyStop();
void handleResume();
void handleDefault();

// ================= class 2,5 특수 처리 =================
void handleSpecialTarget(int classId, float angle, int action);

// ================= 공통 실행 =================
void executeBaseAction(BaseAction act, float angle, float speedOffset = 0.0f);

// ================= 루틴 관련 =================
void startRoutine(TimedAction* r, int length);
void updateRoutine();
void cancelRoutine();
bool isRoutineActive();

// ================= 비상 정지 =================
void emergencyStop();

#endif