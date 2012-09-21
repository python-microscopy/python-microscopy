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

import serial;
import time

class piezo_e816:    
    def __init__(self, portname='COM1', maxtravel = 12.00, Osen=None, hasTrigger=False):
        self.max_travel = maxtravel
        self.waveData = None
        self.numWavePoints = 0
        

        self.ser_port = serial.Serial(portname, 115200, rtscts=1, timeout=2, writeTimeout=2)
        if not Osen == None:
            #self.ser_port.write('SPA A8 %3.4f\n' % Osen)
            self.osen = Osen
        else:
            self.osen = 0
            
        self.MAXWAVEPOINTS = 64
        if self.GetFirmwareVersion() > 3.2:
            self.MAXWAVEPOINTS = 256

        self.ser_port.write('WTO A0\n')
        self.ser_port.write('SVO A1\n')
        
        self.lastPos = self.GetPos()

        self.driftCompensation = False
        self.hasTrigger = hasTrigger

    def ReInit(self):
        self.ser_port.write('WTO A0\n')
        self.ser_port.write('SVO A1\n')
        self.lastPos = self.GetPos() 
        
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        if (fPos >= 0):
            if (fPos <= self.max_travel):
                self.ser_port.write('MOV A%3.4f\n' % fPos)
                self.lastPos = fPos
            else:
                self.ser_port.write('MOV A%3.4f\n' % self.max_travel)
                self.lastPos = self.max_travel
        else:
            self.ser_port.write('MOV A%3.4f\n' % 0.0)
            self.lastPos = 0.0

    def GetPos(self, iChannel=0):
        self.ser_port.flush()
        time.sleep(0.05)
        self.ser_port.write('POS? A\n')
        self.ser_port.flushOutput()
        time.sleep(0.05)
        res = self.ser_port.readline()
        return float(res) + self.osen

    def SetDriftCompensation(self, dc = True):
        if dc:
            self.ser_port.write('DCO A1\n')
            self.ser_port.flushOutput()
            self.driftCompensation = True
        else:
            self.ser_port.write('DCO A0\n')
            self.ser_port.flushOutput()
            self.driftCompensation = False

    def PopulateWaveTable(self,iChannel, data):
        '''Load wavetable data to piezo controller - data should be a sequence/array of positions
        (in um)and should have at most 64 items'''
        if len(data) > self.MAXWAVEPOINTS:
            #wave table too big
            raise RuntimeError('piezo_e816 - wave table must not have more than %d entries' % self.MAXWAVEPOINTS)

        self.waveData = data
        self.numWavePoints = len(data)

        self.ser_port.flush()
        time.sleep(0.05)

        for i, v in zip(range(self.numWavePoints), data):
            self.ser_port.write('SWT A%d %3.4f\n' % (i, v))
            self.ser_port.flushOutput()
            time.sleep(0.01)
            res = self.ser_port.readline()
            #print res

    def GetWaveTable(self):
        '''Read wavetable back from contoller'''

        data = []

        self.ser_port.flush()
        time.sleep(0.05)

        for i in range(64):
            self.ser_port.write('SWT? A%d\n' %i)
            self.ser_port.flushOutput()
            time.sleep(0.05)
            res = self.ser_port.readline()
            data.append(float(res))

        return data

    def StartWaveOutput(self,iChannel=0, dwellTime=None):
        '''Start wavetable output. dwellTime should be the desired dwell time in
        microseconds, or None (default). If dwellTime is None then the external
        wavetable trigger input (pin 9 on io connector) will be used.'''

        if self.numWavePoints < 1:
            raise RuntimeError('piezo_e816 - no wave table defined')

        if dwellTime == None:
            #triggered
            self.ser_port.write('WTO A%d\n' % self.numWavePoints)
        else:
            self.ser_port.write('WTO A%d %3.4f\n' % (self.numWavePoints, dwellTime))
            
        self.ser_port.flushOutput()

    def StopWaveOutput(self, iChannel=0):
        self.ser_port.write('WTO A0\n')
        self.ser_port.flushOutput()

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
        
    def GetFirmwareVersion(self):
        import re
        self.ser_port.write('*IDN?\n')
        self.ser_port.flush()
        
        verstring = self.ser_port.readline()
        return float(re.findall(r'V(\d\.\d\d)', verstring)[0])
        
        
        
