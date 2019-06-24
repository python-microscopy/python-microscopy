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
from PYME.Acquire.Hardware.aotf import AOTF
import logging
logger = logging.getLogger(__name__)

try:
    import Queue
except ImportError:
    import queue as Queue

class AAOptoMDS(AOTF):
    def __init__(self, calibrations, com_port='COM6', name='AAOptoMDS', n_chans=8):
        AOTF.__init__(self, name, calibrations)
        self.com = serial.Serial(com_port,  timeout=1)
        self.n_chans = n_chans

        self.freq = [0] * n_chans
        self.power = [0] * n_chans
        self.is_on = True
        self.channel_enabled = [False] * n_chans

        self.command_queue = Queue.Queue()
        self.reply_queue = Queue.Queue()

        self.polling_thread = None
        self.TurnOn()

        # Grab the initial properties
        self.GetStatus()

        [self.TurnOff(channel) if self.channel_enabled[channel] else None for channel in range(self.n_chans)]

    def _purge(self):
        try:
            while True:
                self.reply_queue.get_nowait()
        except:
            pass

    def _query(self, command):
        """
        Get value from RF driver via serial port.
        """
        self._purge()
        cmd = b'%b\r\n' % command.encode()
        self.command_queue.put(cmd)

        try:
            line = self.reply_queue.get()
            return line
        except:
            pass

    def _readline(self, ser, nLines=1):
        ret = b''
        for n in range(nLines):
            time.sleep(.1)
            ret += ser.readline()
        return ret

    def _poll(self):
        while self.is_on:
                try:
                    cmd = self.command_queue.get(False)
                    # print cmd
                    self.com.write(cmd)
                    self.com.flushOutput()

                except Queue.Empty:
                    pass

                # wait a little for reply
                ret = self._readline(self.com)

                if not ret == b'':
                    self.reply_queue.put(ret)

        time.sleep(.05)

    def IsOn(self, channel=None):
        if channel is not None:
            return self.is_on
        else:
            return self.channel_enabled[channel] and self.is_on

    def IsEnabled(self, channel):
        return self.channel_enabled[channel]

    def TurnOn(self):
        """

        Parameters
        ----------
        channel

        Returns
        -------

        Notes
        -----
        Polling must be resumed (i.e. is_on True and polling thread active) before any serial commands are issued.

        """
        self.is_on = True
        # make sure com port is open
        if not self.com.is_open:  # be a bit gentler than
            self.com.open()
        # make sure we are polling
        if not self.polling_thread or not self.polling_thread.is_alive():
            self.polling_thread = threading.Thread(target=self._poll)
            self.polling_thread.start()

    def Enable(self, channel):
        if not self.is_on:
            self.TurnOn()
        # turn on the channel
        self._query('L' + str(channel + 1) + 'O1')
        self.channel_enabled[channel] = True

    def Disable(self, channel):
        self._query('L' + str(channel + 1) + 'O0')
        self.channel_enabled[channel] = False

    def SetPower(self, power, channel):
        command = 'L'+str(channel+1)+'D%2.2f' % power
        if power > 0:
            # make sure we're running
            if not self.is_on:
                self.TurnOn()
            # enable the channel
            command += 'O1'
            self.channel_enabled[channel] = True

        self._query(command)
        self.power[channel] = power

    def GetPower(self, channel):
        return self.power[channel]

    def SetFreq(self, freq, channel):
        self._query('L'+str(channel+1)+'F'+str(freq))
        self.freq[channel] = freq

    def GetFreq(self, channel):
        return self.freq[channel]

    def GetStatus(self):
        """
        Initial properties. Must be called before polling starts.
        """
        self._purge()
        self.com.flush()
        self.com.write(b'S?\r\n')
        self.com.flushOutput()
        # get the four lines
        reply = []
        while len(reply) < self.n_chans:  # avoid the pitfall of taking an empty line, b'\n'
            line = self.reply_queue.get()
            if line.startswith(b'\rl'):
                reply.append(line)
        self._purge()
        # ret = [filter(None, reply.split(b'\n')))
        for channel in range(self.n_chans):
            retrm = reply[channel].replace(b'\x00', b'').replace(b'= ', b'=').strip()
            s = re.compile(br'l(?P<channel>\d+) F=(?P<freq>\d+.\d+) P=(?P<power>\d+.\d+) (?P<onoff>\w+)')
            vals = s.search(retrm)
            self.freq[channel] = float(vals.group('freq'))
            self.power[channel] = float(vals.group('power'))
            self.channel_enabled[channel] = vals.group('onoff') == 'ON'

    def SaveSettings(self):
        self._query('E')

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
        self.is_on = False
        try:
            self.polling_thread.join()
        except:
            pass

        # disconnect from com port
        self.com.close()