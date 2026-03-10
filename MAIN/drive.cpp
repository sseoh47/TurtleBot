#include "drive.h"

// 일반 주행 중, 실제 수행 중인 주행 저장(새 주행 들어왔을때, 기존과 비교)
static BaseAction currentAction = ACT_STOP;
// 현재 행동이 시작된 시간(ms) 저장
static unsigned long actionStart = 0;
// 현재 행동을 몇 ms동안 할지 저장
static unsigned long actionDuration = 0;

// BaseAction을 실제 모터 명령으로 바꿔 실행하는 함수
// act   : 어떤 행동을 할지
// angle : 차선 중앙 오차 기반 조향값
void executeBaseAction(BaseAction act, float angle)
{
    switch(act)
    {
        case ACT_FORWARD:   // 직진
            applyAngleDrive(angle,1.0,0);
            break;

        case ACT_LEFT:      // 좌회전
            applyAngleDrive(angle - 20,1.0,0);
            break;

        case ACT_RIGHT:     // 우회전
            applyAngleDrive(angle + 20,1.0,0);
            break;

        case ACT_ROTATE:    // 제자리 회전
            setWheelRPM(30,-30);
            break;

        case ACT_REVERSE:   //후진
            setWheelRPM(-20,-20);
            break;

        case ACT_STOP:      // 정지
            stopMotors();
            break;

        case ACT_SLOW:      // 감속
            applyAngleDrive(angle,0.5,0);
            break;
    }
}

// 일반 주행(MODE_MANUAL)에서 action과 angle을 이용해 현재 행동을 결정하고 실행하는 함수
// angle  : 차선 중앙 오차값(조향용)
// action : 라즈베리파이나 상위 제어부가 넘긴 기본 행동 값
void manualDrive(float angle, int action)
{
    unsigned long now = millis();

    // 새로운 주행과 기존 주행이 다르면
    // 새 주행이 시작됨!
    if(action != currentAction)
    {
        // int로 들어온 주행값을 BaseAction enum으로 변환, 저장
        currentAction = (BaseAction)action;
        actionStart = now;

        // 행동 종류별 유지 시간 설정
        switch(action)
        {
            case ACT_LEFT:
            case ACT_RIGHT:
                actionDuration = 2000;
                break;

            default:
                actionDuration = 0;
        }
    }

    // 현재 주행 시간이 제한이 있으며 시간이 지난 경우
    if(actionDuration > 0 && now - actionStart > actionDuration)
    {
        // 직진 상태로 복귀
        currentAction = ACT_FORWARD;
        // 제한 시간 0으로 초기화
        actionDuration = 0;
    }

    // 현재 결정된 행동을 실제 모터에 적용해 주행
    executeBaseAction(currentAction, angle);
}

// 주행 시스템의 메인 진입 함수
// 보통 loop()에서 계속 호출되며,
// 현재 상태(MANUAL / ROUTINE / EMERGENCY)에 따라 적절한 동작을 수행

// 우선순위:
// 1. EMERGENCY
// 2. STOP 명령
// 3. ROUTINE
// 4. MANUAL
void updateDrive(float angle, int action)
{
    // 이미 긴급정지 상태이면
    // 어떤 명령이 들어와도 계속 정지 상태를 유지한다.
    if(driveMode == MODE_EMERGENCY)
    {
        stopMotors();
        return;
    }

    // 현재 들어온 action이 STOP이면
    // 즉시 긴급 정지 처리로 전환한다.
    // routine 실행 중이든 manual 중이든 모두 무시하고 정지를 우선
    if(action == ACT_STOP)
    {
        emergencyStop();
        return;
    }

     // 현재 루틴 실행 중이면
    // 일반 주행(manualDrive)은 수행하지 않고
    // 루틴 단계 실행(updateRoutine)만 진행한다.
    if(driveMode == MODE_ROUTINE)
    {
        updateRoutine();
        return;
    }

    // angle + action을 이용해 manualDrive를 수행한다.
    manualDrive(angle, action);
}