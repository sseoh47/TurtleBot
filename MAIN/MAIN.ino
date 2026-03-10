#include "communication.h"
#include "drive.h"
#include "motor.h"

int class_;
float angle;
int action;

bool driveEnabled = false;

void setup()
{
    initCommunication();
    initMotor();
}

void loop()
{
    if(readCommand(class_, angle, action))
    {
        Serial.print("class: ");
        Serial.println(class_);

        Serial.print("action: ");
        Serial.println(action);

        switch(class_)
        {
            case 9: // START
                driveEnabled = true;
                cancelRoutine();
                action = ACT_FORWARD;
            break;

            case 10: // ARRIVE
                driveEnabled = false;
                emergencyStop();
            break;

            case 3: // 사람 감지 → 긴급정지
                driveEnabled = false;
                emergencyStop();
            break;

            case 2: // 물류 루틴
                if(driveEnabled)
                    startRoutine(logisticsRoutine, logisticsRoutineLength);
            break;

            case 6: // 주차 루틴
                if(driveEnabled)
                    startRoutine(parkingRoutine, parkingRoutineLength);
            break;

            case 5: // 직진
                action = ACT_FORWARD;
            break;

            case 7: // 좌회전
                action = ACT_LEFT;
            break;

            case 8: // 우회전
                action = ACT_RIGHT;
            break;

            case 4: // 저속 주행
                action = ACT_SLOW;
            break;

            default:
            break;
        }
    }

    if(driveEnabled)
        updateDrive(angle, action);

    updateRoutine();
}