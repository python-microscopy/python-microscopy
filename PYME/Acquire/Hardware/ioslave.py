# -*- coding: utf-8 -*-
"""
Created on Sun Oct 06 15:59:23 2013

@author: David Baddeley
"""

import serial
import time

class IOSlave(object):
    def __init__(self, port='COM12'):
        self.ser_port = serial.Serial(port, 9600, rtscts=0, timeout=2, writeTimeout=2)
        
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
    
    

if __name__ == '__main__':
    #run as a temperature logger
    ios = IOSlave()

    time.sleep(2)    
    
    f = open('c:\\Data\\temp.log', 'a')
    
    print 'starting loop'
    while True:
        #print 't'
        temp = ios.GetTemperature(0)
        print '%f\t %f' % (time.time(), temp)
        f.write('%f\t %f\n' % (time.time(), temp))
        f.flush()
        time.sleep(1)