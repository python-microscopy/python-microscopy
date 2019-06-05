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

try:
    import Queue
except ImportError:
    import queue as Queue

from PYME.Acquire.Hardware.lasers import Laser


class MPBCWLaser(Laser):
    def __init__(self, comPort = 'COM3', turnOn=False, name='MPBCW', init_power=1000, **kwargs):
        self.com = serial.Serial(comPort, timeout=.1)
        self.powerControlable = True

        self.doPoll = True

        self.commandQueue = Queue.Queue()
        self.replyQueue = Queue.Queue()

        self.threadPoll = threading.Thread(target=self._poll)
        self.threadPoll.start()

        time.sleep(1)

        self.power = 0
        self.SetPower(init_power)
        self.MAX_POWER = float(self._query('getpowersetptlim 0').split()[1])
        self.isOn = False

        Laser.__init__(self, name, turnOn, **kwargs)

    def _purge(self):
        try:
            while True:
                self.replyQueue.get_nowait()
        except:
            pass

    def _query(self, command):
        """
        Get value from laser via serial port.
        """
        self._purge()
        cmd = '?%s\r\n' % command
        self.commandQueue.put(cmd)

        line = self.replyQueue.get(timeout=3)

        return line

    def _readline(self, ser):
        return ser.readline()

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
                time.sleep(.1)
                ret = self._readline(ser)

                if not ret == '':
                    self.replyQueue.put(ret)

            time.sleep(.05)

    def IsOn(self):
        return self.isOn

    def TurnOn(self):
        self._query('setldenable 1')
        self.isOn = True

    def TurnOff(self):
        self._query('setldenable 0')
        self.isOn = False

    def SetPower(self, power):
        self._query('setpower 2 ' + str(power))
        self.power = power

    def GetPower(self):
        return self.power

    def GetRealPower(self):
        return float(self._query('power 0').split('\r')[0])

    def Close(self):
        print('Shutting down %s' % self.name)
        # try:
        self.TurnOff()
        time.sleep(.1)
        # finally:
        self.doPoll = False
        # self.ser_port.close()

    def __del__(self):
        self.Close()
