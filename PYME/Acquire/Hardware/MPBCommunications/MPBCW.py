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
    def __init__(self, comPort = 'COM3', name='MPBCW', turn_on=False, init_power=200, **kwargs):
        self.com = serial.Serial(comPort, 9600, timeout=.1)
        self.powerControlable = True

        self.command_queue = Queue.Queue()
        self.reply_queue = Queue.Queue()

        self.polling_thread = None
        self.is_on = False

        time.sleep(1)

        self.power = 0


        Laser.__init__(self, name, turn_on, **kwargs)
        if self.IsOn():
            self.SetPower(init_power)
            self.MAX_POWER = float(self._query(b'getpowersetptlim 0').split()[1])

    def _purge(self):
        try:
            while True:
                self.reply_queue.get_nowait()
        except:
            pass

    def _query(self, command):
        """
        Get value from laser via serial port.
        """
        self._purge()
        cmd = b'%b\r\n' % command
        self.command_queue.put(cmd)

        line = self.reply_queue.get(timeout=3)

        return line

    def _readline(self, ser):
        return ser.readline()

    def _poll(self):
        while self.is_on:
            with self.com as ser:
                try:
                    cmd = self.command_queue.get(False)
                    ser.write(cmd)
                    ser.flushOutput()

                except Queue.Empty:
                    pass

                # wait a little for reply
                time.sleep(.1)
                ret = self._readline(ser)

                if ret != b'':
                    self.reply_queue.put(ret)

            time.sleep(.05)

    def IsOn(self):
        """
        Returns
        -------
        is_on: bool
            Initialization status of the laser

        Notes
        -----
        Would be nice to check explicitly that everything is working, e.g. (self.is_on and
        self.polling_thread is not None and self.polling_thread.is_alive() and self.com.is_open and self.check_on()),
        but this function is called to update the microscope state, so a serial command is too expensive.

        """
        return self.is_on

    def check_on(self):
        response = self._query(b'getldenable')
        return bool(response.split(b'\n')[0])

    def TurnOn(self):
        # make sure serial is open
        try:
            self.com.open()
        except serial.SerialException:
            pass

        self.is_on = True
        # make sure polling thread is alive
        if self.polling_thread is None or not self.polling_thread.is_alive():
            self.polling_thread = threading.Thread(target=self._poll)
            self.polling_thread.start()

        # turn on the laser
        self._query(b'setldenable 1')

    def TurnOff(self):
        self._query(b'setldenable 0')
        # stop polling
        self.is_on = False
        # kill polling thread
        self.polling_thread.join()

    def SetPower(self, power):
        self._query(('setpower 2 ' + str(power)).encode())
        self.power = power

    def GetPower(self):
        return self.power

    def GetRealPower(self):
        return float(self._query(b'power 0').split(b'\r')[0])

    def Close(self):
        print('Shutting down %s' % self.name)
        self.TurnOff()
        time.sleep(.1)

    def __del__(self):
        self.Close()
