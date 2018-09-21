#!/usr/bin/python

import serial

from PYME.Acquire.Hardware.lasers import Laser

class MatchboxLaser(Laser):
    def __init__(self, name, turnOn=False, portname='COM1', minpower=0.001, maxpower=0.1, maxpowerDAC = 2200, **kwargs): # minpower, maxpower in Watts
        self.ser_args = dict(port=portname, baudrate=9600, timeout=1, writeTimeout=2)

        self.powerControlable = True
        self.isOn=True
        self.maxpower = maxpower
        self.minpower = minpower
        self.power =  minpower/maxpower
        self.maxpowerDAC = maxpowerDAC

        Laser.__init__(self, name, turnOn)

    def IsOn(self):
        return self.isOn

    def TurnOn(self):
        with serial.Serial(**self.ser_args) as ser:
            ser.write('f 1\n')
            ser.flush()
            self.isOn = True

    def TurnOff(self):
        with serial.Serial(**self.ser_args) as ser:
            ser.write('f 0\n')
            ser.flush()
            self.isOn = False

    def SetPower(self, power):
        if power < (self.minpower/self.maxpower) or power > 1:
            power = self.minpower/self.maxpower
            raise RuntimeError('Error setting laser power: Power must be between %3.2f and 1' % power)
        self.power = power

        with serial.Serial(**self.ser_args) as ser:
            ser.write('p %3.0f\n' % (self.power*self.maxpowerDAC))
            ser.flush()

    #    if self.isOn:
    #        self.TurnOn() #turning on actually sets power

    def getOutputPowerDAC(self):
        with serial.Serial(**self.ser_args) as ser:
            ser.write('r p\n')
            #ser.flush()
    
            res = ser.readline()
        
        return ord(res[0])*256+ord(res[1])

    def GetPower(self):
        return self.power