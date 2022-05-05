
## Microphoton devices picosecond delay module

import serial
import time
import threading


mpd_psd_errors = {
    1 : 'Command not recognized',
    2 : 'Picosecond Delayer in local mode; cannot set parameters',
    3 : 'Frequency divider value too high (>999); parameter is not set',
    4 : 'Frequency divider value too low (<1); parameter is not set',
    5 : 'Trigger level is higher than maximum 2 V; parameter is not set',
    6 : 'Trigger level is lower than minimum -2 V; parameter is not set',
    7 : 'Delay value is higher than the maximum delay; parameter is not set',
    8 : 'Delay value is less than 0 ps; parameter is not set',
    9 : 'Pulse width value too high (>250 ns); parameter is not set',
    10: 'Pulse width value too low (<1 ns); parameter is not set'
}

def check_success(resp):
    if b'ERR' in resp:
        err_no = int((resp.split(b'ERR')[-1]).decode())
        try:
            raise RuntimeError(mpd_psd_errors[err_no])
        except KeyError:
            raise RuntimeError('Unknown error code; %d' % err_no)
    return resp


class PicosecondDelayer(object):
    """
    Microphoton devices picosecond delay module. This implementation uses
    serial commands without context managers due to issues experienced with
    an arduino previously 
    (see https://github.com/python-microscopy/python-microscopy/issues/1194)

    However, that would potentially be a cleaner option than opening the port
    and leaving it open continually.

    Most properties are implemented as properties with getters returning 
    cached values, while GetXXXX calls will query the board over serial.

    Echo mode means the board replies back the command it received, the string
    terminator '#' and then its response. Turning off echo mode while their
    labview software communicates with the board is problematic because it can
    interupt communication.

    Local mode means most settings (delay, pulse width, trigger level, divider,
    edge, or I/O) cannot be set over serial and must be set using the front
    panel of the box.
    """
    def __init__(self, port='COM4'):
        self.lock = threading.Lock()
        self.ser = serial.Serial(port=port, baudrate=115200, 
                                 bytesize=serial.EIGHTBITS,
                                 parity=PARITY_NONE, stopbits=STOPBITS_ONE,
                                 rtscts=False, timeout=.1, writeTimeout=2)
        
        self._delay = self.GetDelay()
        self._pulse_width = self.GetPulseWidth()
        self._trigger_level = self.GetTriggerLevel()
        self._divide_by = self.GetFrequencyDivider()
        self._edge = self.GetEdge()
        self._enabled = self.GetIO()
        self._echo_mode = self.GetEchoMode()
        self._temperature = self.GetTemperature()
        self._max_delay = self.GetMaxDelay()

        # there is no way to query high-speed mode without setting it.
        self._high_speed_mode = False  # start us off in normal mode
        
    
    @property
    def temperature(self):
        """
        Returns
        -------
        temp: float
            Temperature in units of Celsius. Should stabilize to about 55 C
        """
        return self._temperature
    
    @property
    def delay(self):
        return self._delay
    
    @delay.setter
    def delay(self, delay):
        """
        Parameters
        ----------
        delay: int
            delay setpoint in units of picoseconds. Delay can be varied in 10 ps
            steps from 0 to MAX-DELAY. MAX-DELAY is slightly nuanced, but
            something like 50 nanoseconds.
        """
        with self.lock:
            self.ser.write(b'SD%d\n' % delay)
            self._delay = int(str(self.ser.readline()))
    
    @property
    def pulse_width(self):
        return self._pulse_width
    
    @pulse_width.setter
    def pulse_width(self, pulse_width):
        """
        Parameters
        ----------
        pulse_width: int
            pulse width duration in nanoseconds. Possible values are
            non-linearly distributed from 1 ns to 250 ns, and will be rounded to
            on the unit.
        """
        with self.lock:
            self.ser.write(b'SP%d\n' % pulse_width)
            self._pulse_width = int(str(self.ser.readline()))
    
    @property
    def trigger_level(self):
        return self._trigger_level
    
    @property.setter
    def trigger_level(self, trigger_level):
        """
        Parameters
        ----------
        trigger_level: int
            threshold voltage in mV to trigger an output pulse. Can be set in
            10 mV steps from -2 V to + 2 V. Level is rounded to the nearest
            10 mV on unit.
        """
        with self.lock:
            self.ser.write(b'SH%d\n' % trigger_level)
            self._trigger_level = int(str(self.ser.readline()))
    
    @property
    def frequency_divider(self):
        return self._divide_by
    
    @frequency_divider.setter
    def frequency_divider(self, divide_by):
        """
        Parameters
        ----------
        divide_by: int
            frequency divider factor. Possible values are integers from 1 to
            999.
        """
        with self.lock:
            self.ser.write(b'SV%d\n' % divide_by)
            self._divide_by = int(str(self.ser.readline()))
    
    @property
    def edge(self):
        return self._edge
    
    @edge.setter
    def edge(self, rising):
        """
        Parameters
        ----------
        rising: bool
            significant edge for trigger input. False: falling edge, True:
            rising edge.
        """
        with self.lock:
            self.ser.write(b'SE%d\n' % rising)
            self._divide_by = bool(str(self.ser.readline()))

    @property
    def io(self, io):
        return self._enabled

    @io.setter
    def io(self, io):
        """
        Parameters
        ----------
        io: bool
            enable (True) or disable (False) output signal
        """
        with self.lock:
            self.ser.write(b'EO%d\n' % io)
            self._enabled = bool(str(self.ser.readline()))
    
    def Enable(self):
        self.io(True)
    
    def Disable(self):
        self.io(False)
    
    @property
    def echo_mode(self):
        return self._echo_mode
    
    @echo_mode.setter
    def echo_mode(self, echo_mode):
        """
        Parameters
        ----------
        echo_mode: bool
            enable (True) or disable (False) echo mode
        """
        if not echo_mode:
            raise RuntimeError('Echo mode should never be turned off over serial')
        with self.lock:
            self.ser.write(b'EM%d\n' % io)
            self._echo_mode = bool(str(self.ser.readline()))
    
    @property
    def high_speed_mode(self):
        return self._high_speed_mode
    
    @high_speed_mode.setter
    def high_speed_mode(self, high_speed):
        """
        high-speed mode stops refreshing the display on the unit in order to the
        achieve fastest set-delay update rate.

        Parameters
        ----------
        high_speed: bool
            enable (True) or disable (False) high-speed mode, 
        """
        with self.lock:
            self.ser.write(b'HS%d\n' % io)
            self._high_speed_mode = bool(str(self.ser.readline()))
    
    def GetTemperature(self):
        with self.lock:
            self.ser.write(b'RT#\n' % io)
            self._temperature = float(str(self.ser.readline()))
        return self._temperature
    
    def GetDelay(self):
        with self.lock:
            self.ser.write(b'RD#\n' % io)
            self._delay = int(str(self.ser.readline()))
        return self._delay
    
    def GetPulseWidth(self):
        with self.lock:
            self.ser.write(b'RP#\n')
            self._pulse_width = int(str(self.ser.readline()))
        return self._pulse_width
    
    def GetTriggerLevel(self):
        with self.lock:
            self.ser.write(b'RH#\n')
            self._trigger_level = int(str(self.ser.readline()))
        return self._trigger_level
    
    def GetEdge(self):
        with self.lock:
            self.ser.write(b'RE#\n')
            self._edge = bool(str(self.ser.readline()))
        return self._edge
    
    def GetIO(self):
        with self.lock:
            self.ser.write(b'RO#\n')
            self._io = bool(str(self.ser.readline()))
        return self._io
    
    def GetFrequencyDivider(self):
        with self.lock:
            self.ser.write(b'RO#\n')
            self._divide_by = int(str(self.ser.readline()))
        return self._divide_by
    
    def GetMaxDelay(self):
        with self.lock:
            self.ser.write(b'RMD#\n')
            self._max_delay = int(str(self.ser.readline()))
        return self._max_delay
