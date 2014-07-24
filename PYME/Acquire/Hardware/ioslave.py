# -*- coding: utf-8 -*-
"""
Created on Sun Oct 06 15:59:23 2013

@author: David Baddeley
"""

import serial
import time

class IOSlave(object):
    def __init__(self, port='COM12'):
        self.ser_port = serial.Serial(port, 115200, rtscts=0, timeout=.1, writeTimeout=2)
        
    def SetDigital(self, chan, value):
        self.ser_port.write('SD%d %d\n' % (chan, value))
        res = self.ser_port.readline()
       
    def SetAnalog(self, chan, value):
        self.ser_port.write('SA%d %f\n' % (chan, value))
        res = self.ser_port.readline()
    
    def SetFlash(self, chan, value):
        self.ser_port.write('SF%d %d\n' % (chan, value))
        res = self.ser_port.readline()
    
    def GetAnalog(self, chan):
        self.ser_port.write('QA%d\n' % chan)
        return float(self.ser_port.readline())
        
    def GetTemperature(self, chan):
        self.ser_port.write('QT%d\n' % chan)
        return float(self.ser_port.readline())
        


        
from PYME.Acquire.Hardware.lasers import Laser

class AOMLaser(Laser):
    ENABLE_PIN=4
    AMPLITUDE_PIN=5
    FULL_SCALE_VOLTS = 5.0
    def __init__(self, name,turnOn=False, ios = None, maxpower=1):
        if ios == None:
            self.ios = IOSlave()
        else:
            self.ios = ios
            
        self.powerControlable = True
        self.isOn=True
        self.maxpower = maxpower

        self.SetPower(1)

        Laser.__init__(self, name, turnOn)

    def IsOn(self):
        return self.isOn

    def TurnOn(self):
        self.ios.SetDigital(self.ENABLE_PIN, 1)
        self.isOn = True

    def TurnOff(self):
        self.ios.SetDigital(self.ENABLE_PIN, 0)
        self.isOn = False

    def SetPower(self, power):
        if power < 0 or power > 1:
            raise RuntimeError('Error setting laser power: Power must be between 0 and 1')
        self.ios.SetAnalog(self.AMPLITUDE_PIN, power/self.FULL_SCALE_VOLTS)
        self.power = power

        if self.isOn:
            self.TurnOn() #turning on actually sets power

    def GetPower(self):
        return self.power
        
if __name__ == '__main__':
    #run as a temperature logger
    ios = IOSlave()

    time.sleep(2)    
    
    f = open('c:\\Data\\temp.log', 'a')
    
    print( 'starting loop')
    while True:
        #print 't'
        temp = ios.GetTemperature(0)
        print(( '%f\t %f' % (time.time(), temp)))
        f.write('%f\t %f\n' % (time.time(), temp))
        f.flush()
        time.sleep(1)
