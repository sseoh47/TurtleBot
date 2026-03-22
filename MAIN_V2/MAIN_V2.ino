#include "communication.h"
#include "drive.h"
#include "motor.h"
// 테스트 파일
int class_ = 0;
float angle = 0.0f;
int action = 0;

unsigned long lastRxTime = 0;
const unsigned long RX_TIMEOUT = 300;

bool startupRoutineActive = false;
bool startupTriggered = false;
uint8_t startupStep = 0;
unsigned long startupStepStart = 0;


void setup()
{
    initCommunication();
    initMotor();
    initDrive();

    startupRoutineActive = false;
    startupTriggered = false;
    startupStep = 0;
    startupStepStart = 0;
}

void loop()
{
    if (readCommand(class_, angle, action))
    {
        lastRxTime = millis();
    }

    // 시작하면 한번 실행될 루틴
    if (!startupTriggered && class_ == 9)
    {
        startupRoutineActive = true;
        startupTriggered = true;
        startupStep = 0;
        startupStepStart = millis();
    }

    else if (!startupTriggered)
    {
        executeBaseAction(ACT_STOP, 0);
        return;
    }

    if (startupRoutineActive)
    {
        updateStartupRoutine();
        return;
    }

    if (class_ < 0 || class_ > 10)
        class_ = 0;

    if (!(action == 0 || action == 1 || action == 2 || action == 3 || action == 4))
        action = 0;

    if (class_ == 3 || class_ == 4)
    {
        //handleEmergencyStop();
        executeBaseAction(ACT_STOP, 0);
        return;
    }

    if (driveMode == MODE_EMERGENCY && class_ != 3 && class_ != 4)
    {
        driveMode = MODE_MANUAL;
    }

    if (timedActionWait || timedActionActive)
    {
        handleTimedAction(currentAction, currentAngle, actionDuration);
        return;
    }

    switch (class_)
    {
        case 0:
            if (action) handleSpecialTarget(class_, angle, action);
            else handleLineLost();
            break;

        case 1:
            if (action) handleSpecialTarget(class_, angle, action);
            else handleLineFollow(normalizeLineAngle(angle));
            break;

        case 2:
        case 5:
        case 10:
            handleSpecialTarget(class_, angle, action);
            break;

        case 6:
        case 8:
            handleTimedAction(ACT_LEFT, -18, 3250);
            break;

        case 7:
            handleTimedAction(ACT_FORWARD, 0, 3250);
            break;

        case 9:
            if (driveMode == MODE_LOGISTICIn)
            {
                startRoutine(logisticsRoutineOut, logisticsRoutineOutLength);
            }
            break;

        default:
            handleDefault();
            break;
    }

    updateRoutine();
}
