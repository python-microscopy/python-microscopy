#!/usr/bin/python

###############
# HamamatsuDCAM.py
#
# Created: 18 September 2017
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

import ctypes
import ctypes.util
import platform

import numpy as np

from PYME.Acquire.Hardware.sCMOSCamera import sCMOSCamera

# Set up POINTER to the DCAM4 API
sys = platform.system()
if sys == 'Windows':
    dcam = ctypes.windll.dcamapi
else:
    raise Exception("Operating system is not supported.")

# Hamamatsu constants (DCAM4)
DCAM_IDSTR_MODEL = ctypes.c_int32(int("0x04000104", 0))
DCAM_IDSTR_CAMERAID = ctypes.c_int32(int("0x04000102", 0))

# Hamamatsu structures (DCAM4)
class DCAMPROP_ATTR(ctypes.Structure):
    """
    Each property has "attributes". Each attribute indicates the characteristics
    of the property.
    """
    _fields_ = [("cbSize", ctypes.c_int32),
                ("iProp", ctypes.c_int32),
                ("option", ctypes.c_int32),
                ("iReserved1", ctypes.c_int32),
                ("attribute", ctypes.c_int32),
                ("iGroup", ctypes.c_int32),
                ("iUnit", ctypes.c_int32),
                ("attribute2", ctypes.c_int32),
                ("valuemin", ctypes.c_double),
                ("valuemax", ctypes.c_double),
                ("valuestep", ctypes.c_double),
                ("valuedefault", ctypes.c_double),
                ("nMaxChannel", ctypes.c_int32),
                ("iReserved3", ctypes.c_int32),
                ("nMaxView", ctypes.c_int32),
                ("iProp_NumberOfElement", ctypes.c_int32),
                ("iProp_ArrayBase", ctypes.c_int32),
                ("iPropStep_Element", ctypes.c_int32)]


class DCAMPROP_VALUETEXT(ctypes.Structure):
    """
    Provides the value text of a specified value of a property.
    """
    _fields_ = [("cbSize", ctypes.c_int32),
                ("iProp", ctypes.c_int32),
                ("value", ctypes.c_double),
                ("text", ctypes.c_char_p),
                ("textbytes", ctypes.c_int32)]


class DCAMAPI_INIT(ctypes.Structure):
    """
    Specifies initialization options and receives the number of connected
    devices.
    """
    _fields_ = [("size", ctypes.c_int32),
                ("iDeviceCount", ctypes.c_int32),
                ("reserved", ctypes.c_int32),
                ("initoptionbytes", ctypes.c_int32),
                ("initoption", ctypes.POINTER(ctypes.c_int32)),
                ("guid", ctypes.c_void_p)]

class DCAMDEV_OPEN(ctypes.Structure):
    """
    Specifies the device index to be opened and receives the HDCAM handle of
    the device.
    """
    _fields_ = [("size", ctypes.c_int32),
                ("index", ctypes.c_int32),
                ("hdcam", ctypes.c_void_p)]


class DCAMDEV_STRING(ctypes.Structure):
    """
    Specifies the string index and receives device information.
    """
    _fields_ = [("size", ctypes.c_int32),
                ("iString", ctypes.c_int32),
                ("text", ctypes.c_char_p),
                ("textbytes", ctypes.c_int32)]

class DCAMException(Exception):
    """
    Raises an Exception related to Hamamatsu DCAM.
    """
    def __init__(self, message):
        Exception.__init__(self, message)


# Initialize the API
paraminit = DCAMAPI_INIT()
paraminit.size = ctypes.sizeof(paraminit)
if int(dcam.dcamapi_init(ctypes.byref(paraminit))) < 0:
    raise DCAMException("DCAM initialization failed.")
nCams = paraminit.iDeviceCount


class HamamatsuDCAM(sCMOSCamera):

    def __init__(self, camNum):
        sCMOSCamera.__init__(self, camNum)
        self.handle = ctypes.c_void_p(0)

    def Init(self):
        if self.camNum < nCams:
            paramopen = DCAMDEV_OPEN()
            paramopen.size = ctypes.sizeof(paramopen)
            paramopen.index = self.camNum
            self.checkStatus(dcam.dcamdev_open(ctypes.byref(paramopen)),
                             "dcamdev_open")
            self.handle = paramopen.hdcam

        self.CameraModel = self.getCamInfo(DCAM_IDSTR_MODEL)
        self.SerialNumber = self.getCamInfo(DCAM_IDSTR_CAMERAID)

    def getCamInfo(self, idStr):
        """
        Get attribute value from the camera by IDSTR (or ERR) (as defined by
        DCAM4) as string.

        Parameters
        ----------
        idStr : c_int32

        Returns
        -------
        Value of IDSTR or ERR as a str.
        """

        c_buf_len = 256
        c_buf = ctypes.create_string_buffer(c_buf_len)
        param = DCAMDEV_STRING()
        param.size = ctypes.sizeof(param)
        param.text = ctypes.addressof(c_buf)
        param.textbytes = ctypes.sizeof(c_buf)
        param.iString = idStr
        dcam.dcamdev_getstring(self.handle, ctypes.byref(param))

        return c_buf.value

    def getCamPropValue(self, iProp):
        """
        Get DCAM property value for property iProp.

        Parameters
        ----------
        iProp : c_int32
            DCAM property (e.g. DCAM_IDPROP_IMAGE_WIDTH)

        Returns
        -------
        value : float
            Value of DCAM property.
        """
        value = ctypes.c_double
        self.checkStatus(dcam.dcamprop_getvalue(self.handle, iProp,
                                                ctypes.byref(value)))

        return value

    def setCamPropValue(self, iProp, value):
        """
        Get DCAM property value for property iProp.

        Parameters
        ----------
        iProp : c_int32
            DCAM property (e.g. DCAM_IDPROP_IMAGE_WIDTH)
        value : float
            Value to set DCAM property.

        Returns
        -------
        None.
        """
        self.checkStatus(dcam.dcamprop_setvalue(self.handle, iProp,
                                                ctypes.cdouble(value)))


    def checkStatus(self, fn_return, fn_name="unknown"):
        """
        Check that the DCAM function call worked.

        Parameters
        ----------
        fn_return :
            Return value of the function call.
        fn_name : str
            Name of the function. Used for debugging.

        Returns
        -------
        fn_return :
            Return value of the function call.
        """
        if int(fn_return) < 0:
            raise DCAMException("DCAM error for " + str(fn_name) + " with "
                                "return value " + str(fn_return) + ": " +
                                self.getCamInfo(fn_return))
        return fn_return

    def Shutdown(self):
        self.checkStatus(dcam.dcamdev_close(self.handle), "dcamdev_close")

    def __del__(self):
        sCMOSCamera.__del__(self)
        dcam.dcamapi_uninit()
