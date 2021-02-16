# -*- coding: utf-8 -*-

# Server class for LC Driver

from msl.loadlib import Server32
import clr  # need this to import types from system
from System import Double, Byte, Boolean

class LCServer(Server32):
    def __init__(self, host, port, quiet, **kwargs):
        """
        A wrapper around 32-bit LCDriver.dll library.

        Parameters
        ----------
        host : str
            The IP address of the server.
        port : int
            The port to open on the server.
        quiet : bool
            Whether to hide :data:`sys.stdout` messages from the server.
        """
        Server32.__init__(self, 'C:\Program Files\ARCoptix\ARCoptix LC Driver 1.2\LCDriver.dll',
                            'net', host, port, quiet)
        # Need True flag to always open as if we might connect multiple LC driver devices
        self.lcdriver = self.lib.ARCoptix.LCdriver.LCdriver(True)

    def get_class_names(self):
        """ Returns the class names in the library. """
        return ';'.join(str(name) for name in self.assembly.GetTypes()).split(';')

    def get_number_of_devices_connected(self):
        """
        Returns the number of LCDrivers connected to the computer. device_number
        in other functions is in the range 0 to (number of LCDrivers - 1).

        Returns
        -------
        int
            the number of LC drivers connected to the computer
        """
        return self.lcdriver.GetNumberOfDevicesConnected()

    def get_serial_number(self, device_number=0):
        """
        Gets the serial number of the active device.

        Parameters
        ----------
        device_number : int 
            Index of the LCDriver

        Returns
        -------
        string
            Serial number of LCDriver
        """
        return self.lcdriver.GetSerialNumber(device_number)

    def get_max_voltage(self):
        """
        Gets the maximal possible output value for the LC driver.

        Returns
        -------
        float
            Max voltage
        """
        return self.lcdriver.GetMaxVoltage()

    def set_dac_voltage(self, V, ch_number, device_number=0):
        """
        Sets the output voltage in volts.

        Parameters
        ----------
        V : float
            output voltage
        ch_number : int
            channel number 0,1,2,3 or 4
        device_number : int
            Index of the LCDriver

        Returns
        -------
        bool
            True if command is sent successfully
        """
        max_volts = self.get_max_voltage()
        if V > max_volts:
            V = max_volts
        elif V < 0.0:
            V = 0.0
        return self.lcdriver.SetDACVoltage(Double(V), Byte(ch_number), device_number)

    def set_triggers(self, out0external, out1external, out2external, out3external, 
                        device_number=0):
        """
        Set the external trigger active for the outputs and for the device number device_number. 

        Only possible to use if the trigger option is available on your LCDDriver!

        Parameters
        ----------
        out0external : bool
            channel 0 external trigger
        out1external : bool
            channel 1 external trigger
        out2external : bool
            channel 2 external trigger
        out3external : bool
            channel 3 external trigger

        Returns
        -------
        bool
            True if command is sent successfully
        """
        return self.lcdriver.SetTriggers(Boolean(out0external), 
                                         Boolean(out1external), 
                                         Boolean(out2external),
                                         Boolean(out3external), 
                                         device_number)