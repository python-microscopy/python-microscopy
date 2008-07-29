import serial;

class piezo_e662:    
    def __init__(self, portname='COM2', maxtravel = 100.00):
        self.max_travel = maxtravel
        self.ser_port = serial.Serial(portname, 9600, rtscts=1, timeout=30)
        self.ser_port.write('DEV:CONT REM\n')
        
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        if (fPos >= 0):
            if (fPos <= self.max_travel):
                self.ser_port.write('POS %3.4f\n' % fPos)
            else:
                self.ser_port.write('POS %3.4f\n' % self.max_travel)
        else:
            self.ser_port.write('POS %3.4f\n' % 0.0)

    def GetPos(self, iChannel=1):
        self.ser_port.write('POS?\n')
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

