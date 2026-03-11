#include "communication.h"
#include "drive.h"
#include "motor.h"
//테스트용 주서억~
int class_ = 0;
float angle = 0.0f;
int action = 0;

//통신끊김시 정지를 위한 변수
unsigned long lastRxTime = 0;
const unsigned long RX_TIMEOUT = 300; // ms

void setup()
{
    initCommunication();
    initMotor();
    initDrive();
}

void loop()
{
    //값 수신시 처리하는 코드
    if (readCommand(class_, angle, action))
    {
        lastRxTime = millis();
    }

    // 통신 끊김 방어
    if (millis() - lastRxTime > RX_TIMEOUT)
    {
        stopMotors();
        return;
    }

    // 입력값 필터링
    if (class_ < 0 || class_ > 9)
        class_ = 0;

    if (angle > 45.0f) angle = 45.0f;
    if (angle < -45.0f) angle = -45.0f;

    if (!(action == 0 || action == 1 || action == 2 || action == 3 || action == 4 || action == 9))
        action = 0;

    // emergency 상태인데 사람/자동차가 사라졌으면 자동 복귀
    if (driveMode == MODE_EMERGENCY && class_ != 3 && class_ != 4)
    {
        driveMode = MODE_MANUAL;
    }

    switch (class_)
    {
        case 0:
            // 차선 인식 안됨 -> 직전 조향 유지
            handleLineLost();
            break;

        case 1:
            // 차선 인식 -> angle 반영 직진 보정
            handleLineFollow(angle);
            break;

        case 2:
        case 5:
            // 물류 / 주차 표기 -> action 기반 특수 처리
            handleSpecialTarget(class_, angle, action);
            break;

        case 3:
        case 4:
            // 사람 / 자동차 -> 즉시 정지
            handleEmergencyStop();
            break;

        case 6:
            // 임시: 좌회전
            handleLeftTurn(angle);
            break;

        case 7:
            // 임시: 직진
            handleStraight(angle);
            break;

        case 8:
            // 임시: 좌회전
            handleLeftTurn(angle);
            break;

        case 9:
            // 출발 신호
            handleResume();
            break;

        default:
            handleDefault();
            break;
    }

    updateRoutine();
}