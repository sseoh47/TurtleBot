#include "communication.h"

#include <Arduino.h>
#include <string.h>

#if defined(ARDUINO_AVR_UNO) || defined(ARDUINO_AVR_MEGA2560)
#include <SoftwareSerial.h>
SoftwareSerial classSoftSerial(7,8); //RX TX
#define CLASS_SERIAL classSoftSerial
#else
#define CLASS_SERIAL Serial
#endif

char rxBuf[64]; // 문자열 임시 저장 버퍼(시리얼 통신은 한 글자씩 들어오기에 모아둘 공간)
uint8_t rxIdx = 0; // 현재 버퍼에 몇 글자 들어왔는지 가리키는 인덱스

void initCommunication()
{
    CLASS_SERIAL.begin(COMM_BAUDRATE);
}

// 시리얼로 들어오는 한 줄 문자열을 읽어서 class_, angle, action으로　분리
// 매개변수 : &(참조), 함수 내에서 바꾼 값이 밖에도 반영
bool readCommand(int &class_, float &angle, int &action)
{
    while(CLASS_SERIAL.available() > 0) // 현재 시리얼 버퍼에 읽지 않은 문자가 몇 개 있는지
    {
        char c = CLASS_SERIAL.read(); // 버퍼에서 한 글자씩 읽어온다

        if(c == '\r') continue; // \r:캐리지 리턴 무시(일부 환경에서 줄 끝이 \r\n)

        if(c == '\n')
        {
            if(rxIdx == 0) return false; // 빈 줄이면 무시

            rxBuf[rxIdx] = '\0'; // 문자열 끝 표시(C는 문자열 끝에 \0있어야함. 문자열 함수들이 끝이라고 인식)

            // 콤마 기준 자르기
            char *p1 = strtok(rxBuf,",");
            char *p2 = strtok(NULL,",");
            char *p3 = strtok(NULL,",");

            if(p1 == NULL) return false; // 첫번째 값 없을 경우 실패

            class_ = atoi(p1);

            if(p2 != NULL)
                angle = atof(p2);
            else
                angle = 0;

            if(p3 != NULL)
                action = atoi(p3);
            else
                action = 0;

            rxIdx = 0; // 버퍼 인덱스 초기화(한 줄 명령 처리 끝났기에 버퍼 위치 0으로)
            return true; // 통신 성공 반환
        }

        // 일반 문자면 버퍼에 저장(줄바꿈 전까지 저장됨)
        if(rxIdx < sizeof(rxBuf)-1)
            rxBuf[rxIdx++] = c;
        // 버퍼 넘치면 reset(오버플로우 방지)
        else
            rxIdx = 0;
    }

    return false;
}