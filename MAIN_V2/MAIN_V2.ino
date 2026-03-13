#include "communication.h"
#include "drive.h"
#include "motor.h"

int class_ = 0;
float angle = 0.0f;
int action = 0;

unsigned long lastRxTime = 0;
const unsigned long RX_TIMEOUT = 300;

void setup()
{
    initCommunication();
    initMotor();
    initDrive();
}

void loop()
{
    if (readCommand(class_, angle, action))
    {
        lastRxTime = millis();
    }

    if (millis() - lastRxTime > RX_TIMEOUT)
    {
        stopMotors();
        return;
    }

    if (class_ < 0 || class_ > 10)
        class_ = 0;

    if (!(action == 0 || action == 1 || action == 2 || action == 3 || action == 9))
        action = 0;

    if (class_ == 3 || class_ == 4)
    {
        handleEmergencyStop();
        return;
    }

    if (driveMode == MODE_EMERGENCY && class_ != 3 && class_ != 4)
    {
        driveMode = MODE_MANUAL;
    }

    if (timedActionActive)
    {
        handleTimedAction(currentAction, currentAngle, actionDuration);
        return;
    }

    switch (class_)
    {
        case 0:
            handleLineLost();
            break;

        case 1:
            if (action)
                handleSpecialTarget(class_, angle, action);
            else
                handleLineFollow(angle);
            break;

        case 2:
        case 5:
        case 10:
            handleSpecialTarget(class_, angle, action);
            break;

        case 6:
        case 8:
            handleTimedAction(ACT_LEFT, -30, 2000);
            break;

        case 7:
            handleTimedAction(ACT_FORWARD, angle, 3000);
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
