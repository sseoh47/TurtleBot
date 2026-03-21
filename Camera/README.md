# TurtleBot Camera Node

라즈베리 파이 5 + Camera Module 3에서 실행하는 비전 송신 코드입니다.

기능:

- 흰색 차선을 검출하면 `class=1`과 중앙 추종용 `angle` 전송
- `ㅓ`, `ㅜ`, `ㄱ` 형태로 판단되면 `class=8` 전송
- 십자가 형태로 판단되면 `class=7` 전송
- 흰색이 거의 없으면 `class=0` 전송

전송 포맷:

- `0xAC + int16 class + float angle + int16 action + xor`
- 아두이노 `MAIN_V2/communication.cpp` 형식과 동일

설치:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-serial python3-picamera2
```

실행:

```bash
cd ~/Desktop/KDT_AI_v2/TurtleBot/Camera
python3 vision_sender.py --serial-port /dev/ttyUSB0
```

기본 차선 주행 전용 v2:

```bash
python3 vision_sender_v2.py --serial-port /dev/ttyUSB0
```

헤드리스 실행:

```bash
python3 vision_sender.py --serial-port /dev/ttyUSB0 --headless
```

튜닝 포인트:

- `--angle-gain`: 조향 각도 민감도
- `--min-lane-pixels`: 흰색이 어느 정도 있어야 차선으로 인정할지
- 코드 안의 HSV white threshold와 branch threshold는 실제 트랙 조명에 맞춰 조절 필요

판단 방식:

- 차선 마스크에서 가장 큰 주행 경로 성분을 선택
- ROI 가장자리의 흰색 연결 여부로 `left/right/top/bottom` branch 존재를 판정
- 조합이 `ㅓ`, `ㅜ`, `ㄱ`이면 좌회전(`class=8`)
- 네 방향이 모두 열리면 십자가(`class=7`)
- 그 외에는 차선 추종(`class=1`)

주의:

- 실제 트랙 모양과 카메라 장착 위치에 따라 threshold는 거의 반드시 한 번은 현장 튜닝해야 합니다.
- 아두이노 쪽 baudrate와 포트(`/dev/ttyUSB0`, `/dev/ttyAMA0` 등)를 실제 연결 상태에 맞춰 수정하세요.

`vision_sender_v2.py` 설명:

- 교차로 분류 없이 항상 `class=1`만 전송
- 흰색 차선 중심만 계산해서 `angle` 전송
- 초기 테스트, 차선 추종 튜닝, PID 전 단계 확인용으로 더 단순하게 사용 가능
