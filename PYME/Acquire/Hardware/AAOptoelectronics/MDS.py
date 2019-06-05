#!/usr/bin/python

###############
# MPBCWLaser.py
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
import time
import threading
import re

try:
    import Queue
except ImportError:
    import queue as Queue

class AAOptoMDS(object):
    def __init__(self, comPort = 'COM6', turnOn=False, name='AAOptoMDS', nChans=8, **kwargs):
        self.com = serial.Serial(comPort,  timeout=1)
        self.name = name
        self.nChans = nChans

        self.freq = [0] * nChans
        self.power = [0] * nChans
        self.isOn = [False] * nChans

        # Grab the initial properties
        self.GetInitStatus()
        time.sleep(1)

        self.doPoll = True

        self.commandQueue = Queue.Queue()
        self.replyQueue = Queue.Queue()

        self.threadPoll = threading.Thread(target=self._poll)
        self.threadPoll.start()

        time.sleep(1)

        [self.TurnOff(ch) for ch in range(self.nChans)]

    def _purge(self):
        try:
            while True:
                self.replyQueue.get_nowait()
        except:
            pass

    def _query(self, command):
        """
        Get value from RF driver via serial port.
        """
        self._purge()
        cmd = '?%s\r\n' % command
        self.commandQueue.put(cmd)

        try:
            line = self.replyQueue.get(timeout=3)
            return line
        except:
            pass

    def _readline(self, ser, nLines=1):
        ret = ''
        for n in range(nLines):
            time.sleep(.1)
            ret += ser.readline()
        return ret

    def _poll(self):
        while self.doPoll:
            with self.com as ser:
                try:
                    cmd = self.commandQueue.get(False)
                    # print cmd
                    ser.write(cmd)
                    ser.flushOutput()

                except Queue.Empty:
                    pass

                # wait a little for reply
                ret = self._readline(ser)

                if not ret == '':
                    self.replyQueue.put(ret)

            time.sleep(.05)

    def IsOn(self, ch):
        return self.isOn[ch]

    def TurnOn(self, ch):
        self._query('L'+str(ch+1)+'O1')
        self.isOn[ch] = True

    def TurnOff(self, ch):
        self._query('L'+str(ch+1)+'O0')
        self.isOn[ch] = False

    def SetPower(self, power, ch):
        self._query('L'+str(ch+1)+'D'+str(power))
        self.power[ch] = power

    def GetPower(self, ch):
        return self.power[ch]

    def SetFreq(self, freq, ch):
        self._query('L'+str(ch+1)+'F'+str(freq))
        self.freq[ch] = freq

    def GetFreq(self, ch):
        return self.freq[ch]

    def GetInitStatus(self):
        """
        Initial properties. Must be called before polling starts.
        """
        self._purge()
        cmd = '?S\r'
        with self.com as ser:
            ser.write(cmd)
            ser.flushOutput()
            val = self._readline(ser, self.nChans+1)
        ret = filter(None, val.split('\n'))
        for ch in range(self.nChans):
            retrm = ret[ch].replace('\x00', '').replace('= ', '=').strip()
            s = re.compile(r'l(?P<ch>\d+) F=(?P<freq>\d+.\d+) P=(?P<power>\d+.\d+) (?P<onoff>\w+)')
            vals = s.search(retrm)
            self.freq[ch] = float(vals.group('freq'))
            self.power[ch] = float(vals.group('power'))
            self.isOn[ch] = vals.group('onoff') == 'ON'

    def SaveSettings(self):
        self._query('E')

    def Close(self):
        print('Shutting down %s' % self.name)
        [self.TurnOff(ch) for ch in range(self.nChans)]
        time.sleep(.1)
        self.doPoll = False

    def __del__(self):
        self.Close()
