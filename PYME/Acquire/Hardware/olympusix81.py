import serial
import time
import threading

class halogenlamp:
    def __init__(self, name, portname='COM17', **kwargs):
        self.ser_port = serial.Serial(portname, 19200, parity='E',
                                      timeout=2, writeTimeout=2)
        self.lock = threading.Lock()
        self.name= name
        self.powerControlable = False
        self.isOn=True
        self.TurnOn()

    def query(self, command, lines_expected=1):
        with self.lock:
            self.ser_port.reset_input_buffer()
            self.ser_port.write(command)
            reply = [self.ser_port.readline() for line in range(lines_expected)]
            self.ser_port.reset_input_buffer()
        return reply

    def IsOn(self):
        return self.isOn
        
    def TurnOn(self):
        self.query(b'1LOG IN\r\n')
        self.query(b'1LMPSW ON\r\n')
        self.ser_port.flush()
        self.isOn = True

    def TurnOff(self):
        self.query(b'1LMPSW OFF\r\n')
        self.query(b'1LOG OUT\r\n')
        self.ser_port.flush()
        self.isOn = False

    def GetName(self):
        return self.name