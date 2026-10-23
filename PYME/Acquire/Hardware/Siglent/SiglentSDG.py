#!/usr/bin/python

###############
# SiglentSDG.py
#
# Driver for Siglent SDG series arbitrary waveform generators (tested on SDG2042X)
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
import threading
import re
import numpy as np

import pyvisa


class SiglentSDG(object):
    def __init__(self, resource, channel=1, name='SDG', high_z=True, min_voltage=None, max_voltage=None):
        """

        Parameters
        ----------
        resource: str
            VISA resource string, e.g. 'USB0::0xF4EC::0x1102::SDG2XFBCA00334::INSTR' or
            'TCPIP0::192.168.10.2::INSTR'. To find it, run ``pyvisa.ResourceManager().list_resources()`` with the generator connected.
        channel: int
            Output channel to control (1 or 2)
        name: str
            Name of the generator
        high_z: bool
            Set the output load to high impedance, so that the set voltages match the voltages
            delivered into a high impedance input (e.g. a laser analog modulation input).
            If False, the generator assumes a 50 Ohm load.
        min_voltage: float
            Lowest output voltage allowed, in V. Commands which would go below this raise an error before anything
            is sent to the generator. None for no limit.
        max_voltage: float
            Highest output voltage allowed, in V. e.g. 5 to protect a 0-5 V laser modulation input. None for no limit.
        """
        self.name = name
        self.channel = int(channel)
        self.lock = threading.Lock()
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage

        rm = pyvisa.ResourceManager()

        self.instr = rm.open_resource(resource)
        self.instr.write_termination = '\n'
        self.instr.read_termination = '\n'
        self.instr.timeout = 2000  # ms

        self.idn = self.query('*IDN?')

        if high_z:
            self.write('C%d:OUTP LOAD,HZ' % self.channel)
        else:
            self.write('C%d:OUTP LOAD,50' % self.channel)

        self._read_settings()
        self.is_on = False
        self.OutputOff()

    def write(self, command):
        """
        Send a command to the device.

        Parameters
        ----------
        command: str
            Command to send, without termination, e.g. 'C1:OUTP ON'
        """
        with self.lock:
            self.instr.write(command)

    def query(self, command):
        """
        Send a query and return the reply.

        Parameters
        ----------
        command: str
            Query to send, without termination, e.g. 'C1:BSWV?'

        Returns
        -------
        reply: str
            Reply from the device with termination stripped
        """
        with self.lock:
            return self.instr.query(command).strip()

    def GetName(self):
        return self.name

    def SetWaveform(self, waveform):
        """

        Parameters
        ----------
        waveform: str
            One of 'SINE', 'SQUARE', 'RAMP', 'PULSE', 'NOISE', 'ARB', 'DC'
        """
        self.write('C%d:BSWV WVTP,%s' % (self.channel, waveform.upper()))
        self._read_settings()
        
    def _check_voltages(self, high, low):
        """
        Raise an error if an output between low and high volts would be outside the allowed range.
        """
        if self.max_voltage is not None and high > self.max_voltage:
            raise ValueError('%s: %g V is above the maximum allowed output of %g V' % (self.name, high, self.max_voltage))
        if self.min_voltage is not None and low < self.min_voltage:
            raise ValueError('%s: %g V is below the minimum allowed output of %g V' % (self.name, low, self.min_voltage))

    def SetFrequency(self, frequency):
        """

        Parameters
        ----------
        frequency: float
            Frequency in Hz
        """
        self.write('C%d:BSWV FRQ,%f' % (self.channel, frequency))
        self._read_settings()

    def SetHighLow(self, high, low):
        """

        Parameters
        ----------
        high: float
            High level in V
        low: float
            Low level in V
        """
        self._check_voltages(high, low)
        self.write('C%d:BSWV HLEV,%f,LLEV,%f' % (self.channel, high, low))
        self._read_settings()

    def SetHigh(self, high):
        self.SetHighLow(high, self._low)

    def SetLow(self, low):
        self.SetHighLow(self._high, low)

    def SetDutyCycle(self, duty):
        """

        Parameters
        ----------
        duty: float
            Duty cycle in percent (square and pulse waveforms only)
        """
        self.write('C%d:BSWV DUTY,%f' % (self.channel, duty))
        self._read_settings()

    def SetSymmetry(self, symmetry):
        """

        Parameters
        ----------
        symmetry: float
            Symmetry in percent (ramp waveform only)
        """
        self.write('C%d:BSWV SYM,%f' % (self.channel, symmetry))
        self._read_settings()

    def SetOffset(self, offset):
        """

        Parameters
        ----------
        offset: float
            Offset in V. For the DC waveform this is the output voltage.
        """
        if self._waveform == 'DC':
            self._check_voltages(offset, offset)
        else:
            # changing the offset shifts the whole waveform up or down
            shift = offset - self._offset
            self._check_voltages(self._high + shift, self._low + shift)
        self.write('C%d:BSWV OFST,%f' % (self.channel, offset))
        self._read_settings()

    def SetSquare(self, frequency, high, low, duty=50):
        """
        Convenience function to set up a square wave in one call.

        Parameters
        ----------
        frequency: float
            Frequency in Hz
        high: float
            High level in V
        low: float
            Low level in V
        duty: float
            Duty cycle in percent
        """
        self.SetWaveform('SQUARE')
        self.SetFrequency(frequency)
        self.SetHighLow(high, low)
        self.SetDutyCycle(duty)

    def SetArbitrary(self, voltages, sample_rate, name='pyme_arb'):
        """
        Upload a custom waveform and play it point by point (TrueArb mode).

        Parameters
        ----------
        voltages: array-like
            Output voltage for each point, in V. 8 to 8e6 points. The waveform repeats continuously.
        sample_rate: float
            Points per second (1e-6 to 75e6). One cycle lasts len(voltages) / sample_rate seconds.
        name: str
            Name the waveform is stored under on the generator (overwritten if it already exists).
        """
        voltages = np.asarray(voltages, dtype=float)
        if not (8 <= len(voltages) <= 8e6):
            raise ValueError('Waveform must have between 8 and 8e6 points, got %d' % len(voltages))
        if not (1e-6 <= sample_rate <= 75e6):
            raise ValueError('Sample rate must be between 1e-6 and 75e6 Sa/s')

        high = voltages.max()
        low = voltages.min()
        self._check_voltages(high, low)
        if high == low:
            raise ValueError('Waveform is constant - use the DC waveform instead')

        # the generator scales its 16-bit range (-32767..32767) to OFST +- AMPL/2
        amplitude = high - low
        offset = (high + low) / 2.
        data = np.round((voltages - offset) / (amplitude / 2.) * 32767).astype('<i2').tobytes()

        command = ('C%d:WVDT WVNM,%s,FREQ,%f,AMPL,%f,OFST,%f,PHASE,0,WAVEDATA,'
                   % (self.channel, name, sample_rate / len(voltages), amplitude, offset))

        with self.lock:
            old_timeout = self.instr.timeout
            self.instr.timeout = 60000  # large waveforms can take a while to transfer
            try:
                self.instr.write_raw(command.encode() + data)
            finally:
                self.instr.timeout = old_timeout

        self.write('C%d:ARWV NAME,%s' % (self.channel, name))
        self.write('C%d:SRATE MODE,TARB,VALUE,%f,INTER,LINE' % (self.channel, sample_rate))
        self._read_settings()

    def GetSettings(self):
        """
        Returns
        -------
        settings: str
            Basic wave settings as reported by the generator, e.g.
            'C1:BSWV WVTP,SQUARE,FRQ,10HZ,...,HLEV,4V,LLEV,0.5V,...'
        """
        return self.query('C%d:BSWV?' % self.channel)

    def _read_settings(self):
        """
        Read the current settings from the generator and cache them, so that the GUI can poll the getters without
        querying the instrument. Called after every change, so the cache also reflects any values the generator
        adjusted (e.g. clipped to its limits).
        """
        fields = self.GetSettings().split(' ', 1)[1].split(',')
        settings = dict(zip(fields[::2], fields[1::2]))

        def number(key, default):
            # values come with units attached, e.g. '10HZ', '0.5V'
            if key not in settings:
                return default
            return float(re.match(r'[-+0-9.eE]+', settings[key]).group())

        self._waveform = settings.get('WVTP', 'SINE')
        self._frequency = number('FRQ', getattr(self, '_frequency', 1000.))
        self._high = number('HLEV', getattr(self, '_high', 1.))
        self._low = number('LLEV', getattr(self, '_low', -1.))
        self._duty = number('DUTY', getattr(self, '_duty', 50.))
        self._symmetry = number('SYM', getattr(self, '_symmetry', 50.))
        self._offset = number('OFST', getattr(self, '_offset', 0.))

    def GetWaveform(self):
        return self._waveform

    def GetFrequency(self):
        return self._frequency

    def GetHigh(self):
        return self._high

    def GetLow(self):
        return self._low

    def GetDutyCycle(self):
        return self._duty

    def GetSymmetry(self):
        return self._symmetry

    def GetOffset(self):
        return self._offset

    def GetActiveParameters(self):
        """
        Returns
        -------
        parameters: list
            Names of the parameters which apply to the current waveform. Used by the GUI to grey out the others.
        """
        if self._waveform == 'SINE':
            return ['Frequency', 'High', 'Low']
        elif self._waveform == 'SQUARE':
            return ['Frequency', 'High', 'Low', 'DutyCycle']
        elif self._waveform == 'RAMP':
            return ['Frequency', 'High', 'Low', 'Symmetry']
        elif self._waveform == 'DC':
            return ['Offset']
        else:
            return []

    def IsOn(self):
        return self.is_on

    def OutputOn(self):
        self.write('C%d:OUTP ON' % self.channel)
        self.is_on = True

    def OutputOff(self):
        self.write('C%d:OUTP OFF' % self.channel)
        # FIXME - would be nice to check this worked
        self.is_on = False

    def register(self, scope):
        self.registerStateHandlers(scope.state)

    def registerStateHandlers(self, scopeState):
        key = 'FunctionGenerators.%s.' % self.name
        scopeState.registerHandler(key + 'On', self.IsOn, lambda v: self.OutputOn() if v else self.OutputOff())
        scopeState.registerHandler(key + 'Waveform', self.GetWaveform, self.SetWaveform)
        scopeState.registerHandler(key + 'AvailableWaveforms', lambda: ['SINE', 'SQUARE', 'RAMP', 'DC'])
        scopeState.registerHandler(key + 'ActiveParameters', self.GetActiveParameters)
        scopeState.registerHandler(key + 'Frequency', self.GetFrequency, self.SetFrequency)
        scopeState.registerHandler(key + 'High', self.GetHigh, self.SetHigh)
        scopeState.registerHandler(key + 'Low', self.GetLow, self.SetLow)
        scopeState.registerHandler(key + 'DutyCycle', self.GetDutyCycle, self.SetDutyCycle)
        scopeState.registerHandler(key + 'Symmetry', self.GetSymmetry, self.SetSymmetry)
        scopeState.registerHandler(key + 'Offset', self.GetOffset, self.SetOffset)

    def Close(self):
        print('Shutting down %s' % self.name)
        self.OutputOff()
        self.instr.close()

    def __del__(self):
        try:
            self.Close()
        except Exception:
            pass
