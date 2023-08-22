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
from PYME.Acquire.Hardware.lasers import Laser

# Notes:
# Interlock defeat must be plugged into the analog interface
# Oddly, the CONTROL switch on the back panel must be set to
# LOCAL rather than REMOTE to use USB commands to do anything
# more than query states.\
# Laser uses a virtual COM port, if device manager does not 
# show laser as virtual COM port, download virtual COM port
# driver from FTDI, laser should show up as USB Serial Converter
# in device manager, check Load VC box in properties, Laser
# should now show up as virtual COM port


class CoherentSapphireLaser(Laser):
    power_controllable = True
    powerControlable = power_controllable
    units='mW'
    def __init__(self, serial_port='COM7', turn_on=False, name='Sapphire', init_power=5, **kwargs):
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
        kwargs
        """
        self.serial_port = serial.Serial(serial_port, baudrate=19200, timeout=.1)
        self.lock = threading.Lock()
        with self.lock:
            self.serial_port.reset_input_buffer()
            self.serial_port.write(b'?E\r\n')
            echo_ret = self.serial_port.readlines()
            self.serial_port.reset_input_buffer()
        if any([b'Sapphire' in s for s in echo_ret]):
            self._sapphire_prompt = 1
        else:
            self._sapphire_prompt = 0
        
        if any([b'?E' in s for s in echo_ret]):
            self._echo = 1
        else:
            self._echo = 0
        
        self.SetPower(init_power)
        self.MIN_POWER = 1e3 * float(self.query(b'?MINLP\r\n')[0])
        self.MAX_POWER = 1e3 * float(self.query(b'?MAXLP\r\n')[0])
        self.is_on = False

        # self.query(b'SYST:COMM:HAND OFF\r\n', lines_expected=0)

        Laser.__init__(self, name, turn_on, **kwargs)



    def query(self, command):
        """
      Send serial command and return a set number of reply lines from the device before clearing the device outputs

      Parameters
      ----------
      command: bytes
          Command to send to the device. Must be complete, e.g. b'command\r\n'

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
            reply = self.serial_port.readlines()
            self.serial_port.reset_input_buffer()
        if self._echo:
            reply.pop(0)
        if self._sapphire_prompt:
            reply.pop(-1)
        return reply


    def IsOn(self):
        # Would be nice to check, but there is a performance hit (this gets tracked as a scope state to update the GUI)
        return self.is_on

    def TurnOn(self):
        self.query(b'L=1\r\n')
        self.is_on = True

    def TurnOff(self):
        self.query(b'L=0\r\n')
        # FIXME - would be nice to check this worked
        self.is_on = False

    def check_set_power(self):
        power = float(self.query(b'?SP\r\n')[0])
        return power

    def check_power_output(self):
        power = float(self.query(b'?P\r\n')[0])
        return power

    def SetPower(self, power):
        # set power
        self.query(b'P=%f\r\n' % power)
        # log actual set power
        self.power = self.check_set_power()

    def GetMAX(self):

        return float(self.query(b'?MAXLP\r\n')[0])

    def GetMIN(self):

        return float(self.query(b'?MINLP\r\n')[0])

    def GetPower(self):
        return self.power

    def Close(self):
        print('Shutting down %s' % self.name)
        self.TurnOff()
        time.sleep(.1)

    def __del__(self):
        self.Close()
