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
from PYME.Acquire.Hardware.serial_device import SerialDevice
import logging
logger = logging.getLogger(__name__)

class AAOptoMDS(SerialDevice, AOTF):
    def __init__(self, calibrations, com_port='COM6', name='AAOptoMDS', n_chans=8, serial_timeout=1):
        logger.debug('AOTF init')
        AOTF.__init__(self, name, calibrations, n_chans)
        logger.debug('Serial device init')
        SerialDevice.__init__(self, com_port, name, serial_timeout)

        # Grab the initial properties
        logger.debug('Getting MDS status')
        self.GetStatus()
        logger.debug('Disabling any active AOTF channels')
        [self.Disable(channel) if self.channel_enabled[channel] else None for channel in range(self.n_chans)]

    def GetStatus(self):
        """
        Initial properties. Must be called before polling starts.
        """
        raw_reply = self.query(b'S?\r\n', lines_expected=5)
        reply = []
        for line in raw_reply:
            if line.startswith(b'\rl'):
                reply.append(line)
        for channel in range(self.n_chans):
            retrm = reply[channel].replace(b'\x00', b'').replace(b'= ', b'=').strip()
            s = re.compile(br'l(?P<channel>\d+) F=(?P<freq>\d+.\d+) P=(?P<power>\d+.\d+) (?P<onoff>\w+)')
            vals = s.search(retrm)
            self.freq[channel] = float(vals.group('freq'))
            self.power[channel] = float(vals.group('power'))
            self.channel_enabled[channel] = vals.group('onoff') == 'ON'

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
        if power > 0:
            # make sure we're running
            if not self.is_on:
                self.TurnOn()
            # enable the channel
            command += b'O1'
            self.channel_enabled[channel] = True

        self.query(command + b'\r\n')
        self.power[channel] = power

    def GetPower(self, channel):
        return self.power[channel]

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

        # stop polling
        SerialDevice.close(self)
