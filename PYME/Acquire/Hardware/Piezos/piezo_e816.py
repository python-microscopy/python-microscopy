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

import logging
import threading

from .base_piezo import PiezoBase

class piezo_e816(PiezoBase):
    def __init__(self, portname='COM1', maxtravel = 12.00, Osen=None, hasTrigger=False):
        self.max_travel = maxtravel
        self.waveData = None
        self.numWavePoints = 0

        self.lock = threading.Lock()
        

        self.ser_port = serial.Serial(portname, 115200, rtscts=1, timeout=2, writeTimeout=2)
        if not Osen is None:
            #self.ser_port.write('SPA A8 %3.4f\n' % Osen)
            self.osen = Osen
        else:
            self.osen = 0
            
        self.MAXWAVEPOINTS = 64
        if self.GetFirmwareVersion() > 3.2:
            self.MAXWAVEPOINTS = 256

        self.ser_port.write(b'WTO A0\n')
        self.ser_port.write(b'SVO A1\n')
        
        self.lastPos = self.GetPos()

        self.driftCompensation = False
        self.hasTrigger = hasTrigger

    def ReInit(self):
        self.ser_port.write(b'WTO A0\n')
        self.ser_port.write(b'SVO A1\n')
        self.lastPos = self.GetPos() 
        
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        with self.lock:
            if (fPos >= 0):
                if (fPos <= self.max_travel):
                    self.ser_port.write(b'MOV A%3.4f\n' % fPos)
                    self.lastPos = fPos
                else:
                    self.ser_port.write(b'MOV A%3.4f\n' % self.max_travel)
                    self.lastPos = self.max_travel
            else:
                self.ser_port.write(b'MOV A%3.4f\n' % 0.0)
                self.lastPos = 0.0

    def GetPos(self, iChannel=0):
        with self.lock:
            self.ser_port.flushOutput()
            time.sleep(0.05)
            self.ser_port.write(b'POS? A\n')
            self.ser_port.flushOutput()
            time.sleep(0.05)
            res = self.ser_port.readline()
            return float(res) + self.osen

    def SetDriftCompensation(self, dc = True):
        with self.lock:
            if dc:
                self.ser_port.write(b'DCO A1\n')
                self.ser_port.flushOutput()
                self.driftCompensation = True
            else:
                self.ser_port.write(b'DCO A0\n')
                self.ser_port.flushOutput()
                self.driftCompensation = False

    def PopulateWaveTable(self,iChannel, data):
        """Load wavetable data to piezo controller - data should be a sequence/array of positions
        (in um)and should have at most 64 items"""
        if len(data) > self.MAXWAVEPOINTS:
            #wave table too big
            raise RuntimeError('piezo_e816 - wave table must not have more than %d entries' % self.MAXWAVEPOINTS)

        self.waveData = data
        self.numWavePoints = len(data)

        self.ser_port.flush()
        time.sleep(0.05)

        for i, v in zip(range(self.numWavePoints), data):
            self.ser_port.write(b'SWT A%d %3.4f\n' % (i, v))
            self.ser_port.flushOutput()
            time.sleep(0.01)
            res = self.ser_port.readline()
            #print res

    def GetWaveTable(self):
        """Read wavetable back from contoller"""

        data = []

        self.ser_port.flush()
        time.sleep(0.05)

        for i in range(64):
            self.ser_port.write(b'SWT? A%d\n' %i)
            self.ser_port.flushOutput()
            time.sleep(0.05)
            res = self.ser_port.readline()
            data.append(float(res))

        return data

    
    # Ruisheng: "StartWaveOutput" is temporarily removed due to a "callNum" error. This forces getBestScanner() to fall back to zScanner. See zScanner.py
    # def StartWaveOutput(self,iChannel=0, dwellTime=None):
    #     """Start wavetable output. dwellTime should be the desired dwell time in
    #     microseconds, or None (default). If dwellTime is None then the external
    #     wavetable trigger input (pin 9 on io connector) will be used."""

    #     if self.numWavePoints < 1:
    #         raise RuntimeError('piezo_e816 - no wave table defined')

    #     if dwellTime is None:
    #         #triggered
    #         self.ser_port.write('WTO A%d\n' % self.numWavePoints)
    #     else:
    #         self.ser_port.write('WTO A%d %3.4f\n' % (self.numWavePoints, dwellTime))
            
    #     self.ser_port.flushOutput()

    def StopWaveOutput(self, iChannel=0):
        self.ser_port.write(b'WTO A0\n')
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
        with self.lock:
            import re
            self.ser_port.write(b'*IDN?\n')
            self.ser_port.flush()

            verstring = self.ser_port.readline().decode()
            return float(re.findall(r'V(\d\.\d\d)', verstring)[0])

    def GetTargetPos(self,iChannel=0):
        return self.lastPos

import numpy as np
class piezo_e816T(PiezoBase):
    def __init__(self, portname='COM1', maxtravel=12.00, Osen=None, hasTrigger=False):
        self.max_travel = maxtravel
        #self.waveData = None
        #self.numWavePoints = 0
        self.units = 'um'

        self.lock = threading.Lock()

        self.ser_port = serial.Serial(portname, 115200, rtscts=1, timeout=2, writeTimeout=2)
        if not Osen is None:
            # self.ser_port.write('SPA A8 %3.4f\n' % Osen)
            self.osen = Osen
        else:
            self.osen = 0

        #self.MAXWAVEPOINTS = 64
        #if self.GetFirmwareVersion() > 3.2:
        #    self.MAXWAVEPOINTS = 256

        self.ser_port.write(b'WTO A0\n')
        self.ser_port.write(b'SVO A1\n')

        self.servo = True
        self.errCode = 0
        self.onTarget = False

        #self.lastPos = self.GetPos()

        self.driftCompensation = False
        self.hasTrigger = hasTrigger

        self.loopActive = True
        self.stopMove = False
        self.position = np.array([0.])
        # self.velocity = np.array([self.maxvelocity, self.maxvelocity])

        # self.targetPosition = np.array([200.])
        self.targetPosition = np.array([(maxtravel / 2.0)])
        # self.targetVelocity = self.velocity.copy()

        self.lastTargetPosition = self.position.copy()


        self.tloop = threading.Thread(target=self._Loop)
        self.tloop.daemon=True
        self.tloop.start()


    def _Loop(self):
        while self.loopActive:
            self.lock.acquire()
            try:
                self.ser_port.flushInput()
                self.ser_port.flushOutput()

                # check position
                time.sleep(0.005)
                self.ser_port.write(b'POS? A\n')
                self.ser_port.flushOutput()
                time.sleep(0.005)
                res1 = self.ser_port.readline()
                # res2 = self.ser_port.readline()
                # print res1, res2
                self.position[0] = float(res1)+ self.osen
                # self.position[1] = float(res2.split('=')[1])

                self.ser_port.write(b'ERR?\n')
                self.ser_port.flushOutput()
                self.errCode = int(self.ser_port.readline())

                if not self.errCode == 0:
                    logging.info(('Stage Error: %d' % self.errCode))

                # print self.targetPosition, self.stopMove
                if not np.all(self.targetPosition == self.lastTargetPosition):
                    # update our target position
                    pos = np.clip(self.targetPosition, 0, self.max_travel)

                    self.ser_port.write(b'MOV A %3.9f\n' % (pos[0],))
                    self.lastTargetPosition = pos.copy()
                    # print('p')
                    logging.debug('Moving piezo to target: %f' % (pos[0],))

                if np.allclose(self.position, self.targetPosition, atol=.002):
                    self.onTarget = True

                # check to see if we're on target
                #self.ser_port.write('ONT? A\n')
                #self.ser_port.flushOutput()
                #time.sleep(0.005)
                #res1 = self.ser_port.readline()

                #if res1 == '':
                #    time.sleep(.5)
                #    res1 = self.ser_port.readline()

                #try:
                #    self.onTarget = int(res1) == 1
                #except ValueError:
                #    self.onTarget = False
                #    logging.exception('Value error on response from ONT')

                # time.sleep(.1)

            except serial.SerialTimeoutException:
                print('Serial Timeout')
                pass
            except IndexError:
                print('IndexException')
            finally:
                self.stopMove = False
                self.lock.release()

        # close port on loop exit
        self.ser_port.close()
        logging.info("Piezo serial port closed")


    def close(self):
        logging.info("Shutting down piezo")
        with self.lock:
            self.loopActive = False
            # time.sleep(.01)
            # self.ser_port.close()

    def ReInit(self):
        with self.lock:
            self.ser_port.write(b'WTO A0\n')
            self.ser_port.write(b'SVO A1\n')
            time.sleep(1)
            self.lastPos = self.GetPos()

    def OnTarget(self):
        return self.onTarget

    def GetPos(self, iChannel=0):
        # self.ser_port.flush()
        # time.sleep(0.005)
        # self.ser_port.write('POS? %d\n' % iChannel)
        # self.ser_port.flushOutput()
        # time.sleep(0.005)
        # res = self.ser_port.readline()
        pos = self.GetPosXY()

        return pos[iChannel - 1]

    def GetPosXY(self):
        return self.position

    def GetTargetPos(self, iChannel=0):
        return self.targetPosition[iChannel]

    def MoveTo(self, iChannel, fPos, bTimeOut=True, vel=None):
        chan = iChannel - 1
        # if vel == None:
        #    vel = self.maxvelocity
        # self.targetVelocity[chan] = vel
        self.targetPosition[chan] = min(max(fPos, 0), self.max_travel)
        self.onTarget = False

    def MoveRel(self, iChannel, incr, bTimeOut=True):
        self.targetPosition[iChannel] += incr
        self.onTarget = False

    def MoveInDir(self, dx, dy, th=.0000):
        # self.targetVelocity[0] = abs(dx)*self.maxvelocity
        # self.targetVelocity[1] = abs(dy)*self.maxvelocity

        if dx > th:
            self.targetPosition[0] = min(max(np.round(self.position[0] + 1), 0), self.max_travel)
        elif dx < -th:
            self.targetPosition[0] = min(max(np.round(self.position[0] - 1), 0), self.max_travel)
        else:
            self.targetPosition[0] = self.position[0]

        self.onTarget = False

    def SetDriftCompensation(self, dc=True):
        with self.lock:
            if dc:
                self.ser_port.write(b'DCO A1\n')
                self.ser_port.flushOutput()
                self.driftCompensation = True
            else:
                self.ser_port.write(b'DCO A0\n')
                self.ser_port.flushOutput()
                self.driftCompensation = False

    # def PopulateWaveTable(self, iChannel, data):
    #     """Load wavetable data to piezo controller - data should be a sequence/array of positions
    #     (in um)and should have at most 64 items"""
    #     if len(data) > self.MAXWAVEPOINTS:
    #         # wave table too big
    #         raise RuntimeError('piezo_e816 - wave table must not have more than %d entries' % self.MAXWAVEPOINTS)
    #
    #     self.waveData = data
    #     self.numWavePoints = len(data)
    #
    #     self.ser_port.flush()
    #     time.sleep(0.05)
    #
    #     for i, v in zip(range(self.numWavePoints), data):
    #         self.ser_port.write('SWT A%d %3.4f\n' % (i, v))
    #         self.ser_port.flushOutput()
    #         time.sleep(0.01)
    #         res = self.ser_port.readline()
    #         # print res

    # def GetWaveTable(self):
    #     """Read wavetable back from contoller"""
    #
    #     data = []
    #
    #     self.ser_port.flush()
    #     time.sleep(0.05)
    #
    #     for i in range(64):
    #         self.ser_port.write('SWT? A%d\n' % i)
    #         self.ser_port.flushOutput()
    #         time.sleep(0.05)
    #         res = self.ser_port.readline()
    #         data.append(float(res))
    #
    #     return data

    # def StartWaveOutput(self, iChannel=0, dwellTime=None):
    #     """Start wavetable output. dwellTime should be the desired dwell time in
    #     microseconds, or None (default). If dwellTime is None then the external
    #     wavetable trigger input (pin 9 on io connector) will be used."""
    #
    #     if self.numWavePoints < 1:
    #         raise RuntimeError('piezo_e816 - no wave table defined')
    #
    #     if dwellTime is None:
    #         # triggered
    #         self.ser_port.write('WTO A%d\n' % self.numWavePoints)
    #     else:
    #         self.ser_port.write('WTO A%d %3.4f\n' % (self.numWavePoints, dwellTime))
    #
    #     self.ser_port.flushOutput()

    # def StopWaveOutput(self, iChannel=0):
    #     self.ser_port.write('WTO A0\n')
    #     self.ser_port.flushOutput()

    def GetControlReady(self):
        return True

    def GetChannelObject(self):
        return 1

    def GetChannelPhase(self):
        return 1

    def GetMin(self, iChan=1):
        return 0

    def GetMax(self, iChan=1):
        return self.max_travel

    def GetFirmwareVersion(self):
        import re
        with self.lock:
            self.ser_port.write(b'*IDN?\n')
            self.ser_port.flush()

            verstring = self.ser_port.readline().decode()
            return float(re.findall(r'V(\d\.\d\d)', verstring)[0])

