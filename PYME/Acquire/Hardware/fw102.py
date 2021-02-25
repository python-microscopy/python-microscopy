#!/usr/bin/python

##################
# fw102.py
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

class FW102B:
    def __init__(self, ser_port='COM3'):
        self.ser = serial.Serial(ser_port, baudrate=115200, timeout=1)

    def getPos(self):
        self.ser.write('pos?\r'.encode())
        reply = self.ser.readline()
        print(reply)
        return int(reply.split(b'\r')[-2].split(b' ')[0])

    def setPos(self, pos):
        self.ser.write(('pos=%d\r' % pos).encode())
        reply = self.ser.readline().decode()
