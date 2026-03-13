#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <Arduino.h>
#include "config.h"

constexpr uint8_t COMMAND_START_BYTE = 0xAC;
constexpr size_t COMMAND_PAYLOAD_SIZE = sizeof(int16_t) + sizeof(float) + sizeof(int16_t);
constexpr size_t COMMAND_PACKET_SIZE = 1 + COMMAND_PAYLOAD_SIZE + 1;

void initCommunication();
bool readCommand(int &class_, float &angle, int &action);

#endif
