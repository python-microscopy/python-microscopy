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
import time

from PYME.Acquire.Hardware.lasers import Laser

class PriorLumen(Laser):
    """Pretend we're a laser so that we can re-use the laser GUI controls"""
    def __init__(self, name,turnOn=False, portname='COM1', **kwargs):
        self.portname = portname
        self.ser_port = serial.Serial(None, 
                                      timeout=.01, writeTimeout=.01, xonxoff=0, rtscts=0)
        self.ser_port.port = portname
        self.powerControlable = False
        self.isOn=False

        Laser.__init__(self, name, turnOn, **kwargs)
        
    def _read(self):
        c = ''
        m = ''
        i = 0
        while not c == '\r' and i < 10:
            c = self.ser_port.read()
            m += c
            i += 1

    def IsOn(self):
        return self.isOn
        #this doesn't seem to work ...
        # try:
        #     self.ser_port.open()
        #     self.ser_port.write('8 1\r')
        #     self.ser_port.flush()
        #
        #     time.sleep(.1)
        #
        #     m = self._read()
        #     print( m)
        #
        #
        #     self.ser_port.close()
        #
        #     return int(m) == 0
        #
        # except Exception:
        #     return False

        #return self.isOn

    def TurnOn(self):
        try:
            #print 'o'            
            self.ser_port.open()
            #print 'w'
            self.ser_port.write('8 1 0\r')
            self.ser_port.flush()
            m = self._read()
                
            self.isOn = True
            #print 'c'
            self.ser_port.close()
        finally:
            pass

    def TurnOff(self):
        try:
            #print 'o'
            self.ser_port.open()
            #print 'w'
            self.ser_port.write('8 1 1\r')
            self.ser_port.flush()
            self.isOn = False
            
            m = self._read()
            #print 'c'    
            self.ser_port.close()
        finally:
            pass

