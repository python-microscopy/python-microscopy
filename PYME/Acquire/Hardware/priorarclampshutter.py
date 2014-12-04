#!/usr/bin/python

##################
# priorarclampshutter.py
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
import time

class Priorarclampshutter:
    def __init__(self, name, portname='COM6'):
        self.ser_port = serial.Serial(portname, 9600, 
                                      timeout=2, writeTimeout=2)
        self.name= name
        self.powerControlable = False
        self.isOn=True
        self.TurnOn()

    def IsOn(self):
        return self.isOn
        
    def TurnOn(self):
        self.ser_port.write('8,1,0\r')
        self.ser_port.flush()
        self.isOn = True

    def TurnOff(self):
        self.ser_port.write('8,1,1\r')
        self.ser_port.flush()
        self.isOn = False

    def GetName(self):
        return self.name

   