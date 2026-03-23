#include "communication.h"

#include <Arduino.h>
#include <stdint.h>
#include <string.h>

#if defined(ARDUINO_AVR_UNO) || defined(ARDUINO_AVR_MEGA2560)
#include <SoftwareSerial.h>
SoftwareSerial classSoftSerial(7, 8); // RX TX
#define CLASS_SERIAL classSoftSerial
#else
#define CLASS_SERIAL Serial
#endif

namespace
{
    struct CommandPayload
    {
        int16_t classId;
        float angle;
        int16_t action;
    };

    static_assert(sizeof(CommandPayload) == COMMAND_PAYLOAD_SIZE, "Unexpected packet size");

    uint8_t rxBuf[COMMAND_PACKET_SIZE];
    size_t rxIdx = 0;
    bool receivingPacket = false;

    uint8_t computeXor(const uint8_t *data, size_t len)
    {
        uint8_t crc = 0;
        for (size_t i = 0; i < len; ++i)
            crc ^= data[i];
        return crc;
    }
}

void initCommunication()
{
    CLASS_SERIAL.begin(COMM_BAUDRATE);
}

bool readCommand(int &class_, float &angle, int &action)
{
    while (CLASS_SERIAL.available() > 0)
    {
        uint8_t byteIn = static_cast<uint8_t>(CLASS_SERIAL.read());

        if (!receivingPacket)
        {
            if (byteIn != COMMAND_START_BYTE)
                continue;

            rxBuf[0] = byteIn;
            rxIdx = 1;
            receivingPacket = true;
            continue;
        }

        rxBuf[rxIdx++] = byteIn;

        if (rxIdx < COMMAND_PACKET_SIZE)
            continue;

        receivingPacket = false;
        rxIdx = 0;

        const uint8_t receivedCrc = rxBuf[COMMAND_PACKET_SIZE - 1];
        const uint8_t computedCrc = computeXor(&rxBuf[1], COMMAND_PAYLOAD_SIZE);

        if (receivedCrc != computedCrc)
            continue;

        CommandPayload payload;
        memcpy(&payload, &rxBuf[1], sizeof(payload));

        class_ = static_cast<int>(payload.classId);
        angle = payload.angle;
        action = static_cast<int>(payload.action);
        return true;
    }

    return false;
}