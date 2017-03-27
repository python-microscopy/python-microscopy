#!/usr/bin/python

##################
# piezo_e662.py
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
from .base_piezo import PiezoBase

class piezo_e662(PiezoBase):
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

