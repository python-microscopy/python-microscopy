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

from lasers import Laser

class PriorLumen(Laser):
    '''Pretend we're a laser so that we can re-use the laser GUI controls'''
    def __init__(self, name,turnOn=False, portname='COM1'):
        self.ser_port = serial.Serial(portname, 
                                      timeout=1, writeTimeout=1, xonxoff=0, rtscts=0)
        self.powerControlable = False
        self.isOn=True

        Laser.__init__(self, name, turnOn)

    def IsOn(self):
        self.ser_port.write('8 1\r')
        self.ser_port.flush()
        
        c = ''
        m = ''
        while not c == '\r':
            c = self.ser_port.read()
            m += c
            
        return int(m) == 0

        #return self.isOn

    def TurnOn(self):
        self.ser_port.write('8 1 0\r')
        self.ser_port.flush()
        c = ''
        m = ''
        while not c == '\r':
            c = self.ser_port.read()
            m += c
            
        self.isOn = True

    def TurnOff(self):
        self.ser_port.write('8 1 1\r')
        self.ser_port.flush()
        self.isOn = False
        
        c = ''
        m = ''
        while not c == '\r':
            c = self.ser_port.read()
            m += c

