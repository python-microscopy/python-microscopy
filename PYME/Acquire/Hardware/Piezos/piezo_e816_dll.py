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

# To install the E816 driver, download 
# http://update.pi-portal.ws/down.php?link=http://syst13.synserver.de/PIUpdateFinder/PI_C-990_CD1_V_1_0_0_6.zip&name=PI_C-990_CD1_V_1_0_0_6.zip&id=59
# and run PISoftwareSuite.exe. Then, navigate to PI_C-990_CD1_V_1_0_0_6\Development\C++\API\noGUI
# and copy the correct version of E816_DLL(_x64).dll to C:\Windows\System32

#import serial
import time
import numpy as np
import logging
import threading
from PYME.Acquire.eventLog import logEvent
from .base_piezo import PiezoBase
from PYME.Acquire.Hardware.GCS import gcs

logger = logging.getLogger(__name__)

def get_connected_devices():
    n, devs= gcs.EnumerateUSB('E-816')

    return devs.split(b'\n')[:n]

class piezo_e816(PiezoBase):
    def __init__(self, identifier=None, maxtravel = 12.00, Osen=None, hasTrigger=False):
        self.max_travel = maxtravel
        self.waveData = None
        self.numWavePoints = 0

        self.lock = threading.Lock()

        if identifier is None:
            devices = get_connected_devices()
            identifier = devices[0]

        self.id = gcs.ConnectUSB(identifier)
        

        if not Osen is None:
            #self.ser_port.write('SPA A8 %3.4f\n' % Osen)
            self.osen = Osen
        else:
            self.osen = 0
            
        self.MAXWAVEPOINTS = 64
        if self.GetFirmwareVersion() > 3.2:
            self.MAXWAVEPOINTS = 256

        #self.ser_port.write('WTO A0\n')
        #self.ser_port.write('SVO A1\n')
        gcs.SVO(self.id,b'A', [1])
        
        self.lastPos = self.GetPos()

        self.driftCompensation = False
        self.hasTrigger = hasTrigger

    def ReInit(self):
        #self.ser_port.write('WTO A0\n')
        #self.ser_port.write('SVO A1\n')
        gcs.SVO(self.id, b'A', [1])
        self.lastPos = self.GetPos() 
        
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        with self.lock:
            if (fPos >= 0):
                if (fPos <= self.max_travel):
                    #self.ser_port.write('MOV A%3.4f\n' % fPos)
                    gcs.MOV(self.id, b'A', [fPos])
                    self.lastPos = fPos
                else:
                    #self.ser_port.write('MOV A%3.4f\n' % self.max_travel)
                    gcs.MOV(self.id, b'A', [self.max_travel])
                    self.lastPos = self.max_travel
            else:
                #self.ser_port.write('MOV A%3.4f\n' % 0.0)
                gcs.MOV(self.id, b'A', [0.0])
                self.lastPos = 0.0

    def GetPos(self, iChannel=0):
        with self.lock:
            #self.ser_port.flushOutput()
            time.sleep(0.05)
            #self.ser_port.write('POS? A\n')
            #self.ser_port.flushOutput()
            #time.sleep(0.05)
            #res = self.ser_port.readline()
            return float(gcs.qPOS(self.id,b'A')[iChannel]) + self.osen

    def SetDriftCompensation(self, dc = True):
        with self.lock:
            if dc:
                #self.ser_port.write('DCO A1\n')
                #self.ser_port.flushOutput()
                gcs.DCO(b'A', [1])
                self.driftCompensation = True
            else:
                #self.ser_port.write('DCO A0\n')
                #self.ser_port.flushOutput()
                gcs.DCO(b'A', [0])
                self.driftCompensation = False

    def PopulateWaveTable(self,iChannel, data):
        """Load wavetable data to piezo controller - data should be a sequence/array of positions
        (in um)and should have at most 64 items"""
        if len(data) > self.MAXWAVEPOINTS:
            #wave table too big
            raise RuntimeError('piezo_e816 - wave table must not have more than %d entries' % self.MAXWAVEPOINTS)

        self.waveData = data
        self.numWavePoints = len(data)

        #self.ser_port.flush()
        #time.sleep(0.05)

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
            #self.ser_port.write('*IDN?\n')
            #self.ser_port.flush()

            #verstring = self.ser_port.readline()
            verstring = gcs.qIDN(self.id).decode()
            return float(re.findall(r'V(\d\.\d\d)', verstring)[0])

    def GetTargetPos(self,iChannel=0):
        return self.lastPos

class piezo_e816T(PiezoBase):
    def __init__(self, identifier=None, maxtravel=12.00, Osen=None, hasTrigger=False, target_tol=.002,
                 update_rate=0.005):
        """

        Parameters
        ----------
        identifier
        maxtravel
        Osen
        hasTrigger
        target_tol: float
            OnTarget tolerance, units of [um]. If position and target position are within target_tol, OnTarget() returns
            True
        update_rate: float
            Seconds for the polling thread to pause between loops.
        """
        self.max_travel = maxtravel
        #self.waveData = None
        #self.numWavePoints = 0
        self.units = 'um'
        self._target_tol = target_tol
        self._update_rate = update_rate

        self.lock = threading.Lock()

        if identifier is None:
            devices = get_connected_devices()
            identifier = devices[0]
        self._identifier = identifier
        self.id = gcs.ConnectUSB(self._identifier)
        
        if not Osen is None:
            # self.ser_port.write('SPA A8 %3.4f\n' % Osen)
            self.osen = Osen
        else:
            self.osen = 0

        #self.MAXWAVEPOINTS = 64
        #if self.GetFirmwareVersion() > 3.2:
        #    self.MAXWAVEPOINTS = 256

        #self.ser_port.write('WTO A0\n')
        #self.ser_port.write('SVO A1\n')
        gcs.SVO(self.id, b'A', [1])

        self.servo = True
        self.errCode = 0
        self.onTarget = False

        #self.lastPos = self.GetPos()

        self.driftCompensation = False
        self.hasTrigger = hasTrigger

        self.position = np.array([0.])
        # self.velocity = np.array([self.maxvelocity, self.maxvelocity])

        self.targetPosition = np.array([maxtravel / 2.0])
        # self.targetVelocity = self.velocity.copy()

        self.lastTargetPosition = self.position.copy()
        self._start_loop()
    
    def _start_loop(self):
        self.loopActive = True
        self.tloop = threading.Thread(target=self._Loop)
        self.tloop.daemon=True
        self.tloop.start()


    def _Loop(self):
        while self.loopActive:
            self.lock.acquire()
            try:
                # check position
                time.sleep(self._update_rate)

                self.position[0] = float(gcs.qPOS(self.id, b'A')[0])+ self.osen

                self.errCode = int(gcs.qERR(self.id))

                if not self.errCode == 0:  # I have yet to see this work
                    logging.info(('Stage Error: %d' % self.errCode))

                # print self.targetPosition, self.stopMove
                if not np.all(self.targetPosition == self.lastTargetPosition):
                    # update our target position
                    pos = np.clip(self.targetPosition, 0, self.max_travel)

                    gcs.MOV(self.id, b'A', pos[:1])
                    self.lastTargetPosition = pos.copy()
                    # print('p')
                    # logging.debug('Moving piezo to target: %f' % (pos[0],))

                if np.allclose(self.position, self.targetPosition, atol=self._target_tol):
                    if not self.onTarget:
                        logEvent('PiezoOnTarget', '%.3f' % self.position[0], time.time())
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
            except RuntimeError as e:
                # gcs.fcnWrap.HandleError throws Runtimes for everything
                logger.error(str(e))
                try:
                    self.errCode = int(gcs.qERR(self.id))
                    logger.error('error code: %s' % str(self.errCode))
                except:
                    logger.error('no error code retrieved')
                if '-1' in str(e):
                    logger.debug('reinitializing GCS connection, 10 s pause')
                    gcs.CloseConnection(self.id)
                    time.sleep(10.0)  # this takes at least more than 1 s
                    try:
                        self.id = gcs.ConnectUSB(self._identifier)
                        logger.debug('restablished connection to piezo')
                    except RuntimeError as e:
                        logger.error('trying to get new device ID')
                        devices = get_connected_devices()
                        self._identifier = devices[0]
                        logger.debug('new device ID acquired')
                        self.id = gcs.ConnectUSB(self._identifier)
                    time.sleep(1.0)
                    logger.debug('turning on servo')
                    gcs.SVO(self.id, b'A', [1])
                    time.sleep(1.0)
            finally:
                self.lock.release()

        # close port on loop exit
        gcs.CloseConnection(self.id)
        logging.info("Piezo USB connection closed")


    def close(self):
        logging.info("Shutting down piezo")
        with self.lock:
            self.loopActive = False

        time.sleep(0.5)
            # time.sleep(.01)
            # self.ser_port.close()

    def ReInit(self):
        """
        Reinitialize a closed connection to the pifoc. Note the pifoc
        connection must have already been closed, whether by using `close`
        or the polling loop failing.
        """
        with self.lock:
            logging.info('restarting e816')
            self.loopActive = False
            time.sleep(1.0)
            self.id = gcs.ConnectUSB(self._identifier)
            gcs.SVO(self.id, b'A', [1])
            time.sleep(1.0)
            self.lastPos = self.GetPos()
        
        logging.info('reinitialized, starting loop')
        self._start_loop()

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
                gcs.DCO(self.id, b'A', 1)
                self.driftCompensation = True
            else:
                gcs.DCO(self.id, b'A', 0)
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

