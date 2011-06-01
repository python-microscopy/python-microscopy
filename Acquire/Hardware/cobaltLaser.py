#!/usr/bin/python

##################
# piezo_e816.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import serial;
import time

from lasers import Laser

class CobaltLaser(Laser):
    def __init__(self, name,turnOn=False, portname='COM1'):
        self.ser_port = serial.Serial(portname, 115200, timeout=2, writeTimeout=2)
        self.powerControlable = True
        self.isOn=True

        self.power = self._getOutputPower()

        Laser.__init__(self, name, turnOn)

    def IsOn(self):
        return self.isOn

    def TurnOn(self):
        self.ser_port.write('p %3.2f\r\n' % self.power)
        self.ser_port.flush()
        self.isOn = True

    def TurnOff(self):
        self.ser_port.write('p 0\r\n')
        self.ser_port.flush()
        self.isOn = False

    def SetPower(self, power):
        if power < 0 or power > .1:
            raise RuntimeError('Error setting laser power: Power must be between 0 and .1')
        self.power = power

        if self.isOn:
            self.TurnOn() #turning on actually sets power

    def _getOutputPower(self):
        self.ser_port.write('p?\r')
        self.ser_port.flush()

        return float(self.ser_port.readline())

    def GetPower(self):
        return self.power