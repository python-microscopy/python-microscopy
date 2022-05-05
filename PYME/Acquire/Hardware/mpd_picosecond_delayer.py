
## Microphoton devices picosecond delay module

import serial
import threading
import logging

logger = logging.getLogger(__name__)

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
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 rtscts=False, timeout=.1, writeTimeout=2)
        
        self._echo_mode = True  # unit starts up in echo mode
        self._delay = self.GetDelay()
        self._pulse_width = self.GetPulseWidth()
        self._trigger_level = self.GetTriggerLevel()
        self._divide_by = self.GetFrequencyDivider()
        self._edge = self.GetEdge()
        self._enabled = self.GetIO()
        self._temperature = self.GetTemperature()
        self._max_delay = self.GetMaxDelay()

        # there is no way to query high-speed mode without setting it.
        self._high_speed_mode = False  # start us off in normal mode
    
    def __del__(self):
        # make sure display on box is useful
        self.high_speed_mode = False
        # close out serial connection
        with self.lock:
            self.ser.close()
    
    def send_command(self, cmd):
        base_cmd = cmd.split(b'#')[0] + b'#'
        with self.lock:
            self.ser.write(cmd + b'\n')
            resp = self.ser.readline()
        check_success(resp)
        if self.echo_mode:
            return ((resp.split(base_cmd)[-1]).rstrip(b'#')).decode()
        else:
            return (resp.rstrip(b'#')).decode()

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
        self._delay = int(self.send_command(b'SD%d#' % delay))
    
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
        self._pulse_width = int(self.send_command(b'SP%d#' % pulse_width))
    
    @property
    def trigger_level(self):
        return self._trigger_level
    
    @trigger_level.setter
    def trigger_level(self, trigger_level):
        """
        Parameters
        ----------
        trigger_level: int
            threshold voltage in mV to trigger an output pulse. Can be set in
            10 mV steps from -2 V to + 2 V. Level is rounded to the nearest
            10 mV on unit.
        """
        self._trigger_level = int(self.send_command(b'SH%d#' % trigger_level))
    
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
        self._divide_by = int(self.send_command(b'SV%d#' % divide_by))
    
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
        self._divide_by = bool(self.send_command(b'SE%d#' % rising))

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
        self._enabled = bool(self.send_command(b'EO%d#' % io))
    
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
            logger.warn('toggling echo-mode during serial operation may interrupt communication')
        self._echo_mode = bool(self.send_command(b'EM%d#' % echo_mode))
    
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
        self._high_speed_mode = bool(self.send_command(b'HS%d#' % high_speed))
    
    def GetTemperature(self):
        self._temperature = float(self.send_command(b'RT#'))
        return self._temperature
    
    def GetDelay(self):
        self._delay = int(self.send_command(b'RD#'))
        return self._delay
    
    def GetPulseWidth(self):
        self._pulse_width = int(self.send_command(b'RP#'))
        return self._pulse_width
    
    def GetTriggerLevel(self):
        self._trigger_level = int(self.send_command(b'RH#'))
        return self._trigger_level
    
    def GetEdge(self):
        self._edge = bool(self.send_command(b'RE#'))
        return self._edge
    
    def GetIO(self):
        self._io = bool(self.send_command(b'RO#'))
        return self._io
    
    def GetFrequencyDivider(self):
        self._divide_by = int(self.send_command(b'RO#'))
        return self._divide_by
    
    def GetMaxDelay(self):
        self._max_delay = int(self.send_command(b'RMD#'))
        return self._max_delay
