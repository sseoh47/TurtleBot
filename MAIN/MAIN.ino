#include "communication.h"
#include "drive.h"
#include "motor.h"

int class_ = 0;
float angle = 0.0f;
int action = 0;

void setup()
{
    initCommunication();
    initMotor();
    initDrive();
}

void loop()
{
    // 새 명령 수신 시 값 갱신, 실패하면 이전 값 유지
    readCommand(class_, angle, action);

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