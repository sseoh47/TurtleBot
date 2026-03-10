#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <Arduino.h>
#include "config.h"

void initCommunication();
bool readCommand(int &class_, float &angle, int &action);

#endif