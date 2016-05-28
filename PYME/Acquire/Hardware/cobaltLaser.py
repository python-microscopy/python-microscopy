#!/usr/bin/python

##################
# piezo_e816.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import serial
#import time

from PYME.Acquire.Hardware.lasers import Laser

class CobaltLaser(Laser):
    def __init__(self, name,turnOn=False, portname='COM1', maxpower=0.1, **kwargs):
        self.ser_args = dict(port=portname, baudrate=115200, timeout=2, writeTimeout=2)
        #ser = serial.Serial(portname, 115200, 
        #                              timeout=2, writeTimeout=2)
        self.powerControlable = True
        self.isOn=True
        self.maxpower = maxpower

        self.power =  0.01#self._getOutputPower()
        
        self._TurnOn()

        Laser.__init__(self, name, turnOn, **kwargs)

    def IsOn(self):
        return self.isOn
        
    def _TurnOn(self):
        with serial.Serial(**self.ser_args) as ser:
            ser.write('@cobas 0\r\n')
            ser.write('l1\r\n')
            ser.flush()
            self.isOn = True

    def TurnOn(self):
        with serial.Serial(**self.ser_args) as ser:
            ser.write('p %3.2f\r\n' % (self.power*self.maxpower))
            ser.flush()
            self.isOn = True

    def TurnOff(self):
        with serial.Serial(**self.ser_args) as ser:
            ser.write('p 0\r\n')
            ser.flush()
            self.isOn = False

    def SetPower(self, power):
        if power < 0 or power > 1:
            raise RuntimeError('Error setting laser power: Power must be between 0 and 1')
        self.power = power

        if self.isOn:
            self.TurnOn() #turning on actually sets power

    def _getOutputPower(self):
        with serial.Serial(**self.ser_args) as ser:
            ser.write('p?\r')
            ser.flush()
    
            res = float(ser.readline())
        
        return res

    def GetPower(self):
        return self.power