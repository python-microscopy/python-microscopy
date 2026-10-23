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
    powerControlable = True
    AVAILABLE_MODES = ['CWP', 'CWC', 'DIGITAL', 'ANALOG', 'MIXED']

    def __init__(self, serial_port='COM8', turn_on=False, name='OBIS', init_power=5, reply_ok=False, **kwargs):
        """

        Parameters
        ----------
        serial_port: str
            serial port
        turn_on: bool
            Whether or not to turn on the laser on instantiating the class
        name: str
            Name of the laser
        init_power: float
            In units of mW
        reply_ok: bool
            Whether this laser is configured to reply 'OK' after each message or not.
            Note this has been observed when communicating throughthe OBIS power supply
            rather than directly with the head unit.
        kwargs
        """
        self.serial_port = serial.Serial(serial_port, timeout=.1)
        self.lock = threading.Lock()
        self._reply_OK = int(reply_ok)



        time.sleep(1)

        self.power = 0
        self.SetPower(init_power)
        self.MIN_POWER = 1e3 * float(self.query(b'SOUR:POW:LIM:LOW?\r\n', lines_expected=1)[0])
        self.MAX_POWER = 1e3 * float(self.query(b'SOUR:POW:LIM:HIGH?\r\n', lines_expected=1)[0])
        self.is_on = False
        self._mode = self.GetMode()

        # self.query(b'SYST:COMM:HAND OFF\r\n', lines_expected=0)

        Laser.__init__(self, name, turn_on, **kwargs)



    def query(self, command, lines_expected=1):
        """
      Send serial command and return a set number of reply lines from the device before clearing the device outputs

      Parameters
      ----------
      command: bytes
          Command to send to the device. Must be complete, e.g. b'command\r\n'
      lines_expected: int
          Number of interesting lines to be returned from the device. Any remaining output from the device will be
          cleared. Note that having lines_expected larger than the actual number of reply lines for a given command
          will not crash, but will take self.timeout seconds for each extra line requested.

      Returns
      -------
      reply: list
          list of lines retrieved from the device. Blank lines are possible

      Notes
      -----
      serial.Serial.readlines method was not used because our device requires a wait until each line is read before
      it writes the next line.
      """
        with self.lock:
            self.serial_port.reset_input_buffer()
            self.serial_port.write(command)
            reply = [self.serial_port.readline() for line in range(lines_expected + self._reply_OK)]
            self.serial_port.reset_input_buffer()
        return reply




    def IsOn(self):
        # Would be nice to check, but there is a performance hit (this gets tracked as a scope state to update the GUI)
        return self.is_on

    def TurnOn(self):
        self.query(b'SOUR:AM:STAT ON\r\n',lines_expected=0)
        self.is_on = True

    def TurnOff(self):
        self.query(b'SOUR:AM:STAT OFF\r\n', lines_expected=0)
        # FIXME - would be nice to check this worked
        self.is_on = False

    def SetPower(self, power):
        """

        Parameters
        ----------
        power: float
            Power in units of mW, note that laser API takes units of W

        Returns
        -------

        """
        self.query(b'SOUR:POW:LEV:IMM:AMPL %f\r\n' % (power / 1e3), lines_expected=0)
        self.power = power

    def GetPower(self):
        return self.power

    def Close(self):
        print('Shutting down %s' % self.name)
        self.TurnOff()
        time.sleep(.1)
    
    def GetMode(self):
        """
        Returns
        -------
        mode: str
            Current modulation mode as reported by the laser, one of 'CWP', 'CWC',
            'DIGITAL', 'ANALOG' or 'MIXED'
        """
        return self.query(b'SOUR:AM:SOUR?\r\n', lines_expected=1)[0].decode().strip()

    def SetAnalog(self):
        """
        Switch to external analog modulation. If not already in analog mode, the laser
        is turned off first, as mode changes are unreliable while emitting. Output power
        is then set by the analog input voltage rather than SetPower.
        """
        if self.GetMode() == 'ANALOG':
            return
        self.TurnOff()
        self.query(b'SOUR:AM:EXT ANAL\r\n', lines_expected=0)
        self._mode = 'ANALOG'

    def SetDigital(self):
        """
        Switch to external digital modulation. If not already in digital mode, the laser
        is turned off first, as mode changes are unreliable while emitting. Output is then
        switched between off and the SetPower level by the digital input.
        """
        if self.GetMode() == 'DIGITAL':
            return
        self.TurnOff()
        self.query(b'SOUR:AM:EXT DIG\r\n', lines_expected=0)
        self._mode = 'DIGITAL'

    def SetMixed(self):
        """
        Switch to external mixed (digital and analog) modulation. If not already in mixed
        mode, the laser is turned off first, as mode changes are unreliable while emitting.
        """
        if self.GetMode() == 'MIXED':
            return
        self.TurnOff()
        self.query(b'SOUR:AM:EXT MIX\r\n', lines_expected=0)
        self._mode = 'MIXED'

    def SetCW(self):
        """
        Switch to CW constant power mode. If not already in CW mode, the laser is
        turned off first, as mode changes are unreliable while emitting.
        """
        if self.GetMode() == 'CWP':
            return
        self.TurnOff()
        self.query(b'SOUR:AM:INT CWP\r\n', lines_expected=0)
        self._mode = 'CWP'

    def SetCWCurrent(self):
        """
        Switch to CW constant current mode. If not already in constant current mode, the
        laser is turned off first, as mode changes are unreliable while emitting.
        """
        if self.GetMode() == 'CWC':
            return
        self.TurnOff()
        self.query(b'SOUR:AM:INT CWC\r\n', lines_expected=0)
        self._mode = 'CWC'
    
    def SetMode(self, mode):
        """
        Parameters
        ----------
        mode: str
            One of 'CWP', 'CWC', 'DIGITAL', 'ANALOG' or 'MIXED'
        """
        if mode == 'CWP':
            self.SetCW()
        elif mode == 'CWC':
            self.SetCWCurrent()
        elif mode == 'DIGITAL':
            self.SetDigital()
        elif mode == 'ANALOG':
            self.SetAnalog()
        elif mode == 'MIXED':
            self.SetMixed()
        else:
            raise ValueError('Unknown mode %s' % mode)

    def registerStateHandlers(self, scopeState):
        Laser.registerStateHandlers(self, scopeState)
        scopeState.registerHandler('Lasers.%s.Mode' % self.name, lambda: self._mode, self.SetMode,
                                   permitted_values=self.AVAILABLE_MODES)

    def __del__(self):
        self.Close()
