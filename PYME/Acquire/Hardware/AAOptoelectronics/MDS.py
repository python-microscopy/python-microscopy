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
# import serial
# import time
# import threading
import re
from PYME.Acquire.Hardware.aotf import AOTF
import serial
import threading
import logging
logger = logging.getLogger(__name__)


class AAOptoMDS(AOTF):
    """
    Our device doesn't tolerate many port openings/closings and requires its buffer to be cleared regularly. Could be
    because we connect to it through an external USB hub. As a result we keep the serial ports open rather than using
    context managers.
    """
    def __init__(self, calibrations, com_port='COM6', name='AAOptoMDS', n_chans=8, serial_timeout=1, baud_rate=57600):
        """
        Parameters
        ----------
        calibrations: dict
            see PYME.Acquire.Hardware.aotf.AOTF base class. Note that the key's (corresponding to each channel) should
            be zero-indexed, this class will handle converting to one-indexed channels to match the MDS API.
        com_port: str
            Name of the com port to connect to, e.g. 'COM14'.
        name: str
            Name of the device
        n_chans: int
            Number of channels, typically 4 or 8.
        serial_timeout: float
            Timeout to be used in all serial reads
        """
        AOTF.__init__(self, name, calibrations, n_chans)

        # initialize serial
        self.timeout = serial_timeout
        # initialize and configure the serial port without opening it
        self.com_port = serial.Serial(com_port, baud_rate, timeout=serial_timeout)
        self.lock = threading.Lock()
        self.is_on = True
        # set to internal control mode, grab a couple extra lines to give the unit time to write before clearing buffer
        logger.debug('Setting AOTF to internal control mode')
        self.query(b'I0\r\n', lines_expected=3)
        # Grab the initial properties
        logger.debug('Getting MDS status')
        self.GetStatus()
        logger.debug('Disabling any active AOTF channels')
        [(self.Disable(channel), self.SetPower(0, channel)) if self.channel_enabled[channel] else None for channel in range(self.n_chans)]

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
            self.com_port.reset_input_buffer()
            self.com_port.write(command)
            reply = [self.com_port.readline() for line in range(lines_expected)]
            self.com_port.reset_input_buffer()
        return reply

    def GetStatus(self):
        """
        Initial properties. Must be called before polling starts.
        """
        raw_reply = self.query(b'S?\r\n', lines_expected=36)
        reply = []
        for line in raw_reply:
            if line.startswith(b'\rl'):
                reply.append(line)
        for channel in range(self.n_chans):
            retrm = reply[channel].replace(b'\x00', b'').replace(b'= ', b'=').replace(b'-', b'').strip()
            s = re.compile(br'l(?P<channel>\d+) F=(?P<freq>\d+.\d+) P=(?P<power>\d+.\d+) (?P<onoff>\w+)')
            vals = s.search(retrm)
            self.freq[channel] = float(vals.group('freq'))
            self.power[channel] = float(vals.group('power'))
            self.channel_enabled[channel] = vals.group('onoff') == b'ON'

    def Enable(self, channel):
        if not self.is_on:
            self.TurnOn()
        # turn on the channel
        self.query(b'L%dO1\r\n' % (channel + 1))
        self.channel_enabled[channel] = True

    def Disable(self, channel):
        self.query(b'L%dO0\r\n' % (channel + 1))
        self.channel_enabled[channel] = False

    def SetPower(self, power, channel):
        command = b'L%dD%2.2f' % (channel+1, power)

        self.query(command + b'\r\n')
        self.power[channel] = power

    def GetPower(self, channel):
        return self.power[channel]

    def SetPowerAndEnable(self, power, channel):
        command = b'L%dD%2.2f' % (channel+1, power)
        if power > 0:
            # make sure we're running
            if not self.is_on:
                self.TurnOn()
            # enable the channel
            command += b'O1'
            self.channel_enabled[channel] = True

        self.query(command + b'\r\n')
        self.power[channel] = power

    def SetFreq(self, freq, channel):
        self.query(b'L%dF%3.2f\r\n' % (channel + 1, freq))
        self.freq[channel] = freq

    def GetFreq(self, channel):
        return self.freq[channel]

    def Close(self):
        logger.debug('Shutting down %s' % self.name)
        # shut down active channels
        for channel in range(self.n_chans):
            if self.channel_enabled[channel]:
                try:
                    self.Disable(channel)
                except:
                    pass

        # close serial port
        self.is_on = False
        try:
            self.com_port.close()
        except Exception as e:
            logger.error(str(e))
