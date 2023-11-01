# -*- coding: utf-8 -*-
"""
Created on Sun Oct 06 15:59:23 2013

@author: David Baddeley
"""

import serial
import time
import threading

#class IOSlave(object):
#    def __init__(self, port='COM12'):
#        self.ser_port = serial.Serial(port, 115200, rtscts=0, timeout=.1, writeTimeout=2)
#        
#    def SetDigital(self, chan, value):
#        self.ser_port.write('SD%d %d\n' % (chan, value))
#        res = self.ser_port.readline()
#       
#    def SetAnalog(self, chan, value):
#        self.ser_port.write('SA%d %f\n' % (chan, value))
#        res = self.ser_port.readline()
#    
#    def SetFlash(self, chan, value):
#        self.ser_port.write('SF%d %d\n' % (chan, value))
#        res = self.ser_port.readline()
#    
#    def GetAnalog(self, chan):
#        self.ser_port.write('QA%d\n' % chan)
#        return float(self.ser_port.readline())
#        
#    def GetTemperature(self, chan):
#        self.ser_port.write('QT%d\n' % chan)
#        return float(self.ser_port.readline())
#        
#class IOSlave2(object):
#    def __init__(self, port='COM12'):
#        #self.portname = port
#        self.ser_port = serial.Serial(None, 115200, rtscts=0, timeout=.1, writeTimeout=2)
#        self.ser_port.setPort(port)
#        
#    def SetDigital(self, chan, value):
#        self.ser_port.open()
#        try:
#            self.ser_port.write('SD%d %d\n' % (chan, value))
#            res = self.ser_port.readline()
#        finally:
#            self.ser_port.close()
#       
#    def SetAnalog(self, chan, value):
#        self.ser_port.open()
#        try:
#            self.ser_port.write('SA%d %f\n' % (chan, value))
#            res = self.ser_port.readline()
#        finally:
#            self.ser_port.close()
#    
#    def SetFlash(self, chan, value):
#        self.ser_port.open()
#        try:
#            self.ser_port.write('SF%d %d\n' % (chan, value))
#            res = self.ser_port.readline()
#        finally:
#            self.ser_port.close()
#    
#    def GetAnalog(self, chan):
#        self.ser_port.open()
#        try:
#            self.ser_port.write('QA%d\n' % chan)
#            res = float(self.ser_port.readline())
#        finally:
#            self.ser_port.close()
#        return res
#        
#    def GetTemperature(self, chan):
#        self.ser_port.open()
#        try:
#            self.ser_port.write('QT%d\n' % chan)
#            res =  float(self.ser_port.readline())
#        finally:
#            self.ser_port.close()
#        return res

class IOSlave(object):
    def __init__(self, port='COM12'):
        self.ser_args = dict(port=port, baudrate=115200, rtscts=False, timeout=.1, writeTimeout=2)
        self.lock = threading.Lock()
        
    def SetDigital(self, chan, value):
        with self.lock:
            with serial.Serial(**self.ser_args) as ser:
                ser.write(b'SD%d %d\n' % (chan, value))
                res = ser.readline()
       
    def SetAnalog(self, chan, value):
        with self.lock:
            with serial.Serial(**self.ser_args) as ser:
                ser.write(b'SA%d %f\n' % (chan, value))
                res = ser.readline()
    
    def SetFlash(self, chan, value):
        with self.lock:
            with serial.Serial(**self.ser_args) as ser:
                ser.write(b'SF%d %d\n' % (chan, value))
                res = ser.readline()
    
    def GetAnalog(self, chan):
        with self.lock:
            with serial.Serial(**self.ser_args) as ser:
                ser.write(b'QA%d\n' % chan)
                res = float((ser.readline()).decode().split('\\')[0])
        
        return res
        
    def GetTemperature(self, chan):
        with self.lock:
            with serial.Serial(**self.ser_args) as ser:
                ser.write(b'QT%d\n' % chan)
                res = float(ser.readline())
        
        return res

class IOSlaveAlwaysOpen(object):
    """
    Implementation of IOSlave without context managers on the serial commands. Note that the standard IOSlave is
    preferable, but this is useful for ports which take a long time to open so commands can be dropped when opening the
    port and immediately writing.

    Notes
    -----
    serial.Serial('COMX') can return before commands will be received if written, which is why this class exists.
    """
    def __init__(self, port='COM12'):
        self.lock = threading.Lock()
        self.ser = serial.Serial(port=port, baudrate=115200, rtscts=False, timeout=.1, writeTimeout=2)

    def SetDigital(self, chan, value):
        with self.lock:
            self.ser.write(b'SD%d %d\n' % (chan, value))
            self.ser.readline()

    def SetAnalog(self, chan, value):
        with self.lock:
            self.ser.write(b'SA%d %f\n' % (chan, value))
            self.ser.readline()

    def SetFlash(self, chan, value):
        with self.lock:
            self.ser.write(b'SF%d %d\n' % (chan, value))
            self.ser.readline()

    def SetServo(self, chan, value):
        with self.lock:
            self.ser.write(b'SS%d %d\n' % (chan, value))
            self.ser.readline()

    def GetAnalog(self, chan):
        with self.lock:
            self.ser.write(b'QA%d\n' % chan)
            res = float((self.ser.readline()).decode().split('\\')[0])

        return res

    def GetTemperature(self, chan):
        with self.lock:
            self.ser.write(b'QT%d\n' % chan)
            res = float(self.ser.readline())

        return res

from PYME.Acquire.Hardware.lasers import Laser

class AOMLaser(Laser):
    ENABLE_PIN=4
    AMPLITUDE_PIN=5
    FULL_SCALE_VOLTS = 5.0
    def __init__(self, name,turnOn=False, ios = None, maxpower=1, **kwargs):
        if ios is None:
            self.ios = IOSlave()
        else:
            self.ios = ios
            
        self.powerControlable = True
        self.isOn=True
        self.maxpower = maxpower

        self.SetPower(1)

        Laser.__init__(self, name, turnOn, **kwargs)

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
        self.ios.SetAnalog(self.AMPLITUDE_PIN, power*self.FULL_SCALE_VOLTS)
        self.power = power

        if self.isOn:
            self.TurnOn() #turning on actually sets power

    def GetPower(self):
        return self.power


class DigitalShutter(Laser):
    def __init__(self, name, turnOn=False, ios=None, pin=13,**kwargs):
        if ios is None:
            self.ios = IOSlave()
        else:
            self.ios = ios

        self._enable_pin = pin

        self.powerControlable = False
        self.isOn = True

        Laser.__init__(self, name, turnOn, **kwargs)

    def IsOn(self):
        return self.isOn

    def TurnOn(self):
        self.ios.SetDigital(self._enable_pin, 1)
        self.isOn = True

    def TurnOff(self):
        self.ios.SetDigital(self._enable_pin, 0)
        self.isOn = False


class FiberShaker(IOSlaveAlwaysOpen):
    def __init__(self, com_port, channel, on_value):
        IOSlaveAlwaysOpen.__init__(self, com_port)
        self.is_on = False
        self.channel = channel
        self.on_value = on_value
        self._counter = 0

    def TurnOn(self):
        self.is_on = True
        self.SetAnalog(self.channel, self.on_value)

    def TurnOff(self):
        self.is_on = False
        self.SetAnalog(self.channel, 0)

    def Notify(self, enable):
        """

        Parameters
        ----------
        enable: bool
            Request from device to increment or decrement counter. True means the device wants the FiberShaker on, while
            False means it is no longer needed.

        """
        if enable:
            self._counter += 1
        else:
            self._counter -= 1
            self._counter = max(self._counter, 0)
        if self._counter == 0 and self.is_on:
            self.TurnOff()
        elif self._counter > 0 and not self.is_on:
            self.TurnOn()

class ServoFiberShaker(FiberShaker):
    """
    Fiber shaker using a brushless DC motor, requiring 50 Hz PWM (e.g. arduino servo library) to control
    """
    def TurnOn(self):
        self.is_on = True
        self.SetServo(self.channel, self.on_value)

    def TurnOff(self):
        self.is_on = False
        self.SetServo(self.channel, 0)


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
