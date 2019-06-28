#!/usr/bin/python

###############
# CoherentOBISLaser.py
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


class CoherentOBISLaser(Laser):
    def __init__(self, comPort = 'COM8', turnOn=False, name='OBIS', init_power=0.005, **kwargs):
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
        self.MIN_POWER = float(self._query(b'SOUR:POW:LIM:LOW?'))
        self.MAX_POWER = float(self._query(b'SOUR:POW:LIM:HIGH?'))
        self.isOn = False

        self._query(b'SYST:COMM:HAND OFF')

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
        cmd = b'?%s\r\n' % command
        self.commandQueue.put(cmd)

        try:
            line = self.replyQueue.get(timeout=3)
        except:
            line = None

        return line

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

                if not ret == b'':
                    self.replyQueue.put(ret)

            time.sleep(.05)

    def _readline(self, ser):
        return self.com.readline()

    def IsOn(self):
        # Would be nice to check, but there is a performance hit (this gets tracked as a scope state to update the GUI)
        return self.isOn

    def TurnOn(self):
        self._query(b'SOUR:AM:STAT ON')
        self.isOn = True

    def TurnOff(self):
        self._query(b'SOUR:AM:STAT OFF')
        # FIXME - would be nice to check this worked
        self.isOn = False

    def SetPower(self, power):
        self._query(('SOUR:POW:LEV:IMM:AMPL ' + str(power)).encode())
        self.power = power

    def GetPower(self):
        return self.power

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
