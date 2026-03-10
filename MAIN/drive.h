#ifndef DRIVE_H
#define DRIVE_H

#include <Arduino.h>
#include "motor.h"

// ================= 기본 주행 종류 =================
enum BaseAction
{
    ACT_FORWARD,    // 직진
    ACT_LEFT,       // 좌회전
    ACT_RIGHT,      // 우회전
    ACT_ROTATE,     // 제자리 회전
    ACT_REVERSE,    // 후진
    ACT_STOP,       // 정지
    ACT_SLOW        // 감속 주행
};

// ================= 주행 모드 =================
enum DriveMode
{
    MODE_MANUAL,    // 일반 주행 상태(라파에서 받은 angle, action으로 바로 움직인다)
    MODE_ROUTINE,   // 특수 주행 상태(물류/주차)
    MODE_EMERGENCY  // 긴급 정지 상태(사람감지 등)
};

// ================= 루틴 동작 구조 =================
// 특수 주행 루틴의 한 단계 구조
struct TimedAction
{
    BaseAction action;          // 어떤 주행인지(직진, 좌회전,...)
    float angle;                // angle
    unsigned long duration;     // 유지 시간(ms)
};

// ================= 루틴 상태 =================
// 현재 루틴이 어디까지 진행되었는지 저장
struct RoutineState
{
    bool active;            // 현재 루틴 실행 중인지
    int index;              // 현재 몇 번째 단계인지
    unsigned long start;    // 현재 단계 시작 시간
    int length;             // 현재 루틴 전체 배열 길이
    TimedAction* routine;   // 현재 실행 중인 루틴 배열을 가리키는 포인터
};

// ================= 전역 =================
extern DriveMode driveMode;                 // 현재 주행 모드

extern TimedAction logisticsRoutine[];      // 물류 주차 루틴
extern TimedAction parkingRoutine[];        // 도착 주차 루틴

// 배열 인덱스 범위 명시 -> 오동작 등 방지용
extern const int logisticsRoutineLength;    // 루틴 배열 길이
extern const int parkingRoutineLength;      // 루틴 배열 길이

// ================= 함수 =================
void updateDrive(float angle, int action);              // 주행 시스템의 메인 진입(loop()에서 호출)

void manualDrive(float angle, int action);              // 일반 주행 처리 함수
void executeBaseAction(BaseAction act, float angle);    // 주행을 실제 모터 명령으로

void startRoutine(TimedAction* r, int length);          // 특수 주행 루틴 시작 함수
void updateRoutine();                                   // 특수 주행 루틴 실행 함수

void emergencyStop();       // 긴급 정지 함수
void cancelRoutine();       // 현재 특수주행 끝내고 일반 주행으로 돌아간다.

#endif