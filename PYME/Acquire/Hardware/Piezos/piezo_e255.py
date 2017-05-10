#!/usr/bin/python

###############
# piezo_e255.py
#
# Copyright David Baddeley, 2012
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
################
import serial
from .base_piezo import PiezoBase

class piezo_e255(PiezoBase):
    #Script fuer den alten PI-Controller E255.60,
    #der nur im Volt-Modus angesteuert erden kann.
    # hier wird erst mal Ch2 verwendet
    def __init__(self, portname='COM1', maxtravel = 100):
        self.max_travel = maxtravel     
        self.ser_port = serial.Serial(portname, 9600, rtscts=0, timeout=60)
        self.ser_port.write('2DO17\r\n')
        self.ser_port.write('2DG16705\r\n')
        self.ser_port.write('2SO0\r\n')
        #self.dummy = 0;
        self.lastPos = 0
                                
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        if (fPos >= 0):
            if (fPos <= self.max_travel):
                self.ser_port.write('2SO%05.0f\r\n' % (100.0*fPos))
                self.lastPos = fPos
            else:
                self.ser_port.write('2SO%05.0f\r\n' % (100.0*self.max_travel))
                self.lastPos = self.max_travel
        else:
            self.ser_port.write('2SO%05.0f\r\n' % 0.0)
            self.lastPos = 0

    def GetPos(self, iChannel=1):
        return self.lastPos
    
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

