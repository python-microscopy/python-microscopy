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

class piezo_e816:    
    def __init__(self, portname='COM1', maxtravel = 12.00, Osen=None):
        self.max_travel = maxtravel
        self.waveData = None
        self.numWavePoints = 0

        self.ser_port = serial.Serial(portname, 115200, rtscts=1, timeout=2, writeTimeout=2)
        if not Osen == None:
            #self.ser_port.write('SPA A8 %3.4f\n' % Osen)
            self.osen = Osen
        else:
            self.osen = 0
        self.ser_port.write('SVO A1\n')
        self.ser_port.write('WTO A0\n')
        self.lastPos = self.GetPos()

        self.driftCompensation = False

    def ReInit(self):
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
        if len(data) > 64:
            #wave table too big
            raise RuntimeError('piezo_e816 - wave table must not have more than 64 entries')

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
