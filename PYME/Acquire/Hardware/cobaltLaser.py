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
    def __init__(self, name,turnOn=False, portname='COM1', maxpower=0.1):
        self.ser_port = serial.Serial(portname, 115200, 
                                      timeout=2, writeTimeout=2)
        self.powerControlable = True
        self.isOn=True
        self.maxpower = maxpower

        self.power =  0.01#self._getOutputPower()
        
        self._TurnOn()

        Laser.__init__(self, name, turnOn)

    def IsOn(self):
        return self.isOn
        
    def _TurnOn(self):
        self.ser_port.write('@cobas 0\r\n')
        self.ser_port.write('l1\r\n')
        self.ser_port.flush()
        self.isOn = True

    def TurnOn(self):
        self.ser_port.write('p %3.2f\r\n' % (self.power*self.maxpower))
        self.ser_port.flush()
        self.isOn = True

    def TurnOff(self):
        self.ser_port.write('p 0\r\n')
        self.ser_port.flush()
        self.isOn = False

    def SetPower(self, power):
        if power < 0 or power > 1:
            raise RuntimeError('Error setting laser power: Power must be between 0 and 1')
        self.power = power

        if self.isOn:
            self.TurnOn() #turning on actually sets power

    def _getOutputPower(self):
        self.ser_port.write('p?\r')
        self.ser_port.flush()

        return float(self.ser_port.readline())

    def GetPower(self):
        return self.power