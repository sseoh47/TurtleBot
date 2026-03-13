#include "communication.h"
#include "drive.h"
#include "motor.h"

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

    // // 통신 끊김 방어
    // if (millis() - lastRxTime > RX_TIMEOUT)
    // {
    //     stopMotors();
    //     return;
    // }

    // // 입력값 필터링
    // if (class_ < 0 || class_ > 10)
    //     class_ = 0;

    // if (!(action == 0 || action == 1 || action == 2 || action == 3 || action == 9))
    //     action = 0;


    // // -------------------------
    // // EMERGENCY 우선 처리
    // // -------------------------
    // if (class_ == 3 || class_ == 4)
    // {
    //     handleEmergencyStop();
    //     return;
    // }

    // emergency 상태인데 사람/자동차가 사라졌으면 자동 복귀
    if (driveMode == MODE_EMERGENCY && class_ != 3 && class_ != 4)
    {
        driveMode = MODE_MANUAL;
    }

    // // -------------------------
    // // timedAction 진행 중
    // // -------------------------
    // if (timedActionActive)
    // {
    //     handleTimedAction(currentAction, currentAngle, actionDuration);
    //     return;
    // }

    switch (class_)
    {
        // case 0:
        //     // 차선 인식 안됨 -> 직전 조향 유지
        //     handleLineLost();
        //     break;

        case 1:
            // 차선 인식 -> angle 반영 직진 보정
            // if (action) handleSpecialTarget(class_, angle, action);
            // else handleLineFollow(angle);
            handleLineFollow(angle);
            break;

        // case 2:
        // case 5:
        // case 10:
        //     // 물류 / 물류x / 주차 표기 -> action 기반 특수 처리
        //     handleSpecialTarget(class_, angle, action);
        //     break;

        // case 6:
        // case 8:
        //     // 임시: 좌회전(2초)
        //     handleTimedAction(ACT_LEFT, -30, 2000);
        //     // handleLeftTurn(-30);
        //     break;

        // case 7:
        //     // 임시: 직진(3초)
        //     handleTimedAction(ACT_FORWARD, angle, 3000);
        //     // handleStraight(angle);
        //     break;

        // case 9:
        //     if (driveMode == MODE_LOGISTICIn)
        //     {
        //         startRoutine(logisticsRoutineOut, logisticsRoutineOutLength);
        //     }
        //     break;

        default:
            handleLineFollow(30);
            // handleDefault();
            break;
    }

    // updateRoutine();
}