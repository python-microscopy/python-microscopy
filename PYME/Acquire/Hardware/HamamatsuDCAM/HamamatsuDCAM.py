#!/usr/bin/python

###############
# HamamatsuDCAM.py
#
# Created: 18 September 2017
# Author : Z Marin
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

from PYME.Acquire.Hardware.Camera import Camera

# Set up POINTER to the DCAM4 API
sys = platform.system()
if sys == 'Windows':
    dcam = ctypes.windll.dcamapi
else:
    raise Exception("Operating system is not supported.")

# Hamamatsu constants (DCAM4)
DCAM_IDSTR_MODEL = ctypes.c_int32(int("0x04000104", 0))
DCAM_IDSTR_CAMERAID = ctypes.c_int32(int("0x04000102", 0))

DCAMPROP_OPTION_NEAREST = ctypes.c_int32(int("0x80000000", 0))
DCAMPROP_OPTION_NEXT = ctypes.c_int32(int("0x01000000", 0))
DCAMPROP_OPTION_SUPPORT = ctypes.c_int32(int("0x00000000", 0))

DCAMPROP_ATTR_HASVALUETEXT= ctypes.c_int32(int("0x10000000", 0))
DCAMPROP_ATTR_WRITABLE = ctypes.c_int32(int("0x00020000", 0))
DCAMPROP_ATTR_READABLE = ctypes.c_int32(int("0x00010000", 0))

DCAM_IDPROP_EXPOSURETIME = ctypes.c_int32(int("0x001F0110", 0))

DCAM_IDPROP_SENSORTEMPERATURE = ctypes.c_int32(int("0x00200310", 0))
DCAM_IDPROP_SENSORTEMPERATURETARGET = ctypes.c_int32(int("0x00200330", 0))


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


# Helper functions
def convertPropertyName(name):
    return name.lower().replace(" ", "_")


# Initialize the API
paraminit = DCAMAPI_INIT()
paraminit.size = ctypes.sizeof(paraminit)
if int(dcam.dcamapi_init(ctypes.byref(paraminit))) < 0:
    raise DCAMException("DCAM initialization failed.")
nCams = paraminit.iDeviceCount


class HamamatsuDCAM(Camera):

    def __init__(self, camNum):
        Camera.__init__(self, camNum)
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

        self.SetIntegTime(self._intTime)

    def StartExposure(self):
        self.StopAq()
        self._temp = self.getCamPropValue(DCAM_IDPROP_SENSORTEMPERATURE)
        self._frameRate = self.getCamPropValue(DCAM_IDPROP_EXPOSURETIME)

        # Custom start exposure for sCMOS or EMCCD (might need to flush buffers)

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
        param.textbytes = ctypes.c_int32(c_buf_len)
        param.iString = idStr
        dcam.dcamdev_getstring(self.handle, ctypes.byref(param))

        return c_buf.value

    def getCamProperties(self):
        """
        Get a list of the camera properties for use in get/set.
        """

        # Create a blank dictionary for the camera properties
        properties = {}

        # Go to the beginning of the property list
        iProp = ctypes.c_int32(0)
        self.checkStatus(dcam.dcamprop_getnextid(self.handle,
                                                 ctypes.byref(iProp),
                                                 DCAMPROP_OPTION_SUPPORT),
                        "dcamprop_getnextid")

        # Loop over the properties and store
        while 1:
            text_len = 64
            text = ctypes.create_string_buffer(text_len)
            self.checkStatus(dcam.dcamprop_getname(self.handle, iProp, text,
                                                   ctypes.c_int32(text_len)),
                             "dcamprop_getname")

            properties[text.value] = iProp.value

            if dcam.dcamprop_getnextid(self.handle, ctypes.byref(iProp),
                                       DCAMPROP_OPTION_SUPPORT) < 0:
                break

        return properties

    def getCamPropAttr(self, iProp):
        """
        Get DCAM property attributes for property iProp.

        Parameters
        ----------
        iProp : c_int32
            DCAM property (e.g. DCAM_IDPROP_IMAGE_WIDTH)

        Returns
        -------
        attr : DCAMPROP_ATTR
            Struct of DCAM property attributes, which includes range and mode.
        """
        attr = DCAMPROP_ATTR()
        attr.cbSize = ctypes.sizeof(attr)
        attr.iProp = iProp

        self.checkStatus(dcam.dcamprop_getattr(self.handle,
                                           ctypes.byref(attr)),
                     "dcamprop_getattr")

        return attr


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
                                                ctypes.byref(value)),
                         "dcamprop_getvalue")

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
                                                ctypes.c_double(value)),
                        "dcamprop_setvalue")

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

    def SetIntegTime(self, intTime):
        self.setCamPropValue(DCAM_IDPROP_EXPOSURETIME, intTime)
        self._intTime = intTime

    def Shutdown(self):
        self.checkStatus(dcam.dcamdev_close(self.handle), "dcamdev_close")
        #if self.camNum < nCams:
        #    dcam.dcamapi_uninit()
