import serial

class FW102B:
    def __init__(self, ser_port='COM3'):
        self.ser = serial.Serial(ser_port, baudrate=115200, timeout=1)

    def getPos(self):
        self.ser.write('pos?\r')
        reply = self.ser.readline()
        return int(reply.split('\r')[-2].split(' ')[0])

    def setPos(self, pos):
        self.ser.write('pos=%d\r' % pos)
        reply = self.ser.readline()
