"""
Control the halogen lamp (transmitted light) on an Olympus IX81 stand.

Uses the serial command set described at https://madhadron.com/science/olympus_ix81_commands.html

TODO - move logic to an IX81 class which also allows control of other stand features and support more of the command set.
TODO - some form of error handling
TODO - consider re-writing the query function to try and match responses to commands (these are not guaranteed to be synchronous) 

"""
import serial
import time
import threading
from PYME.Acquire.Hardware.lasers import Laser

class OlympusIX81HalogenLamp(Laser):
    def __init__(self, name, portname='COM17', turn_on=False, **kwargs):
        self.ser_port = serial.Serial(portname, 19200, parity='E',
                                      timeout=2, writeTimeout=2)
        self.lock = threading.Lock()
        self.name = name
        self.powerControlable = False
        self.isOn = False
        #self.TurnOn()
        Laser.__init__(self, name, turn_on, **kwargs)

    def _query(self, command, lines_expected=1):
        with self.lock:
            self.ser_port.reset_input_buffer()
            self.ser_port.write(command)
            reply = [self.ser_port.readline() for line in range(lines_expected)]
            self.ser_port.reset_input_buffer()
        return reply

    def IsOn(self):
        return self.isOn
        
    def TurnOn(self):
        # make sure serial is open
        try:
            self.ser_port.open()
        except serial.SerialException:
            pass

        self.isOn = True

        # turn on the laser
        self._query(b'1LOG IN\r\n', lines_expected=1)
        self._query(b'1LMPSW ON\r\n', lines_expected=1)
        self._query(b'1LOG OUT\r\n', lines_expected=1)
        #self.ser_port.flush()

    def TurnOff(self):
        self._query(b'1LOG IN\r\n', lines_expected=1)
        self._query(b'1LMPSW OFF\r\n', lines_expected=1)
        self._query(b'1LOG OUT\r\n', lines_expected=1)
        #self.ser_port.flush()
        self.isOn = False

