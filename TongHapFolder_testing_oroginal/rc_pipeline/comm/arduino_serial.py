import serial
import struct

from config import START_BYTE
from domain.types import FinalCommand


class ArduinoSerial:
    def __init__(self, port: str, baudrate: int):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=0.1)

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    def build_packet(self, cmd: FinalCommand) -> bytes:
        angle = 0.0 if cmd.angle is None else float(cmd.angle)
        payload = struct.pack("<hfh", int(cmd.class_id), angle, int(cmd.action))

        crc = 0
        for value in payload:
            crc ^= value

        return bytes([START_BYTE]) + payload + bytes([crc])

    def send(self, cmd: FinalCommand) -> bytes:
        packet = self.build_packet(cmd)
        self.ser.write(packet)
        self.ser.flush()
        return packet
