import serial;
import time

class piezo_e816:    
    def __init__(self, portname='COM4', maxtravel = 100.00, c_grad=1, c_off = 0):
        self.max_travel = maxtravel
        self.ser_port = serial.Serial(portname, 115200, rtscts=1, timeout=2, writeTimeout=2)
        self.ser_port.write('SVO A1\n')
        self.c_grad = c_grad
        self.c_off = c_off
        
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        if (fPos >= 0):
            if (fPos <= self.max_travel):
                # KORREKTURFAKTOR FUER P-621
                fPos = self.c_grad*fPos + self.c_off
                self.ser_port.write('MOV A%3.4f\n' % fPos)
            else:
                self.ser_port.write('MOV A%3.4f\n' % self.max_travel)
        else:
            self.ser_port.write('MOV A%3.4f\n' % 0.0)
        self.ser_port.flush()

    def GetPos(self, iChannel=1):
        self.ser_port.flush()
        time.sleep(0.001)
        self.ser_port.write('POS? A\n')
        self.ser_port.flushOutput()
        time.sleep(0.001)
        res = self.ser_port.readline()
        return float(res)

    def GetControlReady(self):
        return True

    def GetChannelObject(self):
        return 1

    def GetChannelPhase(self):
        return 1

    def GetMin(self,iChan=1):
        return 0

    def GetMax(self, iChan=1):
        return self.max_travel
