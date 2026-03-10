#ifndef MOTOR_H
#define MOTOR_H

void initMotor();

void applyAngleDrive(float angleDeg,float speedScale,float bias);
void setWheelRPM(float left, float right);
void stopMotors();

#endif