
import serial
import threading
import logging
logger = logging.getLogger(__name__)

try:
    import Queue
except ImportError:
    import queue as Queue


class SerialDevice(object):
    """
    Basic serial handling for devices which do not tolerate many port openings/closings, require their buffer to be
    cleared regularly, and generally do not tolerate higher-performance serial back-ends.
    """
    def __init__(self, com_port, name, timeout=1.):
        """
        Parameters
        ----------
        com_port: str
            Name of the com port to connect to, e.g. 'COM14'.
        name: str
            Name of the device
        timeout: float
            Timeout to be used in all serial reads
        """
        self.name = name
        self.timeout = timeout
        # initialize and configure the serial port without opening it
        self.com_port = serial.Serial(timeout=timeout)
        self.com_port.port = com_port
        self.lock = threading.Lock()
        self.is_on = True
        if not self.com_port.is_open:
            self.com_port.open()

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
        serial.Serial.readlines method was not used because some devices wait until each line is read before writing
        next line.
        """
        with self.lock:
            self.com_port.reset_input_buffer()
            self.com_port.write(command)
            reply = [self.com_port.readline() for line in range(lines_expected)]
            self.com_port.reset_input_buffer()
        return reply

    def close(self):
        logger.debug('Shutting down %s' % self.name)
        # stop polling
        self.is_on = False
        try:
            self.com_port.close()
        except Exception as e:
            logger.error(str(e))
