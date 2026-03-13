# RC Car Keyboard Test (Raspberry Pi)

라즈베리파이에서 키보드 입력을 이용해 RC카(MCU)에 직접 주행 명령을 보내는 테스트 프로그램이다.  
MCU와의 통신은 **Serial(UART)** 을 사용하며, 명령 형식은 아래와 같다.

class,angle,action\n

예시:
1,-10.5,0

---

## 프로젝트 목적

이 테스트 프로그램은 다음을 빠르게 검증하기 위해 사용한다.

- MCU와 시리얼 통신 정상 여부
- 모터 제어 동작 확인
- class / angle / action 프로토콜 확인
- 특수 동작(action) 테스트
- 자율주행 로직 없이 수동 테스트

---

## 프로젝트 구조

rc_keyboard_test/
│
├─ manual_drive_test.py (메인 실행 파일)
├─ requirements.txt (필요 패키지)
└─ README.md (현재 문서)

---

## 설치 방법

1. 폴더 생성

mkdir rc_keyboard_test
cd rc_keyboard_test

2. 패키지 설치

pip install -r requirements.txt

---

## 시리얼 포트 설정

manual_drive_test.py 상단의 포트를 환경에 맞게 수정해야 한다.

예시

SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 57600

라즈베리파이에서 연결된 포트 확인

ls /dev/ttyUSB*
ls /dev/ttyACM*

일반적으로 다음 중 하나이다.

- /dev/ttyUSB0
- /dev/ttyACM0

---

## 실행 방법

python3 manual_drive_test.py

---

## 키보드 조작

[기본 주행]

↑ (UP) : 직진  
← (LEFT) : 좌 조향  
→ (RIGHT) : 우 조향

SPACE : 긴급 정지  
R : 재출발

---

## class 테스트

Z : class 0 (차선 인식 실패)

X : class 6 (좌회전 차선)

C : class 7 (직진 차선)

V : class 8 (특수 차선)

---

## 물류 / 주차 action 테스트

1 : cargo + action1 (서행)

2 : cargo + action2 (좌측 회전)

3 : cargo + action3 (정지)

4 : cargo + action4 (특수 루틴)

5 : park + action1

6 : park + action2

7 : park + action3

8 : park + action4

---

## 객체 테스트

P : 사람 감지 (class 3)

O : 자동차 감지 (class 4)

---

## 프로그램 종료

Q : 프로그램 종료

---

## MCU 통신 프로토콜

라즈베리파이 → MCU로 전송되는 데이터 형식

class,angle,action\n

예시

1,-8.5,0

의미

class
0 : 차선 없음
1 : 일반 주행
2 : 물류
3 : 사람
4 : 자동차
5 : 주차
6~8 : 특수 차선
9 : 출발 신호

angle
차선 중심 오차 기반 조향 값 (degree)

action
0 : 일반 주행
1 : 서행
2 : 좌측 90도 회전
3 : 정지
9 : 긴급 정지

---

## 주의 사항

1. MCU와 baudrate가 반드시 동일해야 한다.

예:
57600

2. 프로그램 종료 시 자동으로 정지 명령을 전송한다.

3. 실제 주행 테스트 시 반드시 RC카를 바닥에서 띄운 상태로
   초기 테스트를 진행할 것.

---

## 개발 환경

Raspberry Pi  
Python 3.x  
pyserial
