# -*- coding: utf-8 -*-

"""
Controls for the ARCoptix liquid crystal driver.

Requirements: 
    - Download and install the LC driver from http://www.arcoptix.com/pdf/LC_Driver_1_2.zip.
      This should install to C:\Program Files\ARCoptix\ARCoptix LC Driver 1.2
    - pip install pythonnet msl-loadlib

This uses inter-process communication to access a 32-bit .NET DLL from 64-bit python.
Pre-flight to create the 32-bit server:
    - Create a 32-bit conda environment
        - set CONDA_FORCE_32BIT=1
        - conda create -n py37_32 python=3.7.9  # change this to the python used in your pyme install
        - set CONDA_FORCE_32BIT=1
        - activate py37_32
        - pip install msl-loadlib pyinstaller comtypes pythonnet numpy  # make sure the versions match the versions in your 64-bit environment
                                                                        # numpy is optional, but if you don't use it make sure you cast everything
                                                                        # to float before passing to the server
    - python
        - from msl.loadlib import freeze_server32
        - freeze_server32.main()
        ... PyInstaller logging messages ...
        Server saved to: ...
    - Copy the generated server32-windows.exe file to 
      miniconda3\envs\<64-bit pyme environment>\lib\site-packages\msl\loadlib
    - Copy C:\Program Files\ARCoptix\ARCoptix LC Driver 1.2\CyUSB.dll to
      miniconda3\envs\<pyme environment>\lib\site-packages\msl\loadlib.
      NOTE: We shouldn't need to do this, but setting os.environ['PATH'] doesn't
      update the path for server32-windows.exe. Ideally we'd find a way to do this
      and add C:\Program Files\ARCoptix\ARCoptix LC Driver 1.2 to server32-windows.exe's
      path.
    - set CONDA_FORCE_32BIT=
    - activate <pyme environment>

Created on Fri January 08 2021

@author: zacsimile
"""

import os
import platform
from msl.loadlib import Client64

sys = platform.system()
if sys != 'Windows':
    raise Exception("Operating system is not supported.")

class LCDriver(Client64):
    def __init__(self):
        """
        Communicates with 32-bit LCDriver.dll library via lcserver32.py.
        """
        Client64.__init__(self, module32='lcserver32', 
                            append_sys_path=os.path.dirname(__file__))
    
    def get_class_names(self):
        """ Returns the class names in the library. """
        return self.request32('get_class_names')

    def get_number_of_devices_connected(self):
        """
        Returns the number of LCDrivers connected to the computer. device_number
        in other functions is in the range 0 to (number of LCDrivers - 1).

        Returns
        -------
        int
            the number of LC drivers connected to the computer
        """
        return self.request32('get_number_of_devices_connected')

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
        return self.request32('get_serial_number', device_number)

    def get_max_voltage(self):
        """
        Gets the maximal possible output value for the LC driver.

        Returns
        -------
        float
            Max voltage
        """
        return self.request32('get_max_voltage')

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
        return self.request32('set_dac_voltage', float(V), int(ch_number), int(device_number))

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
        return self.request32('set_triggers', 
                                out0external,
                                out1external, 
                                out2external, 
                                out3external, 
                                device_number)
