import serial
from PYME.Acquire.Hardware.lasers import Laser

class TopticaIBeamLaser(Laser):
    def __init__(self, name, turnOn=False, portname='COM1', minpower=0, maxpower=0.1, **kwargs): # minpower, maxpower in Watts
        self.ser_args = dict(port=portname, baudrate=115200, timeout=1, writeTimeout=2)
        
        self.powerControlable = True
        self.isOn = True
        self.maxpower = maxpower
        self.minpower = minpower
        self.power = 0.01
        
        Laser.__init__(self, name, turnOn)
    
    def IsOn(self):
        return self.isOn
    
    def TurnOn(self):
        with serial.Serial(**self.ser_args) as ser:
            ser.write(b'la off\n')
            ser.flush()
            self.isOn = True
    
    def TurnOff(self):
        with serial.Serial(**self.ser_args) as ser:
            ser.write(b'la on\n')
            ser.flush()
            self.isOn = False
    
    def SetPower(self, power):
        if power < (self.minpower) or power > self.maxpower:
            raise RuntimeError('Error setting laser power: Power must be between %3.2f and %3.2f' % (self.minpower, self.maxpower))
        self.power = power
        
        with serial.Serial(**self.ser_args) as ser:
            ser.write(b'ch 1 pow %3.3f\n' % (self.power * 1e3))
            ser.flush()
    
    def GetPower(self):
        return self.power