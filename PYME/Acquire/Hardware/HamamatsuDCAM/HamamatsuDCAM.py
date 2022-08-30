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

DCAMPROP_OPTION_SUPPORT = ctypes.c_int32(int("0x00000000", 0))

DCAMPROP_ATTR_HASVALUETEXT= ctypes.c_int32(int("0x10000000", 0))
DCAMPROP_ATTR_WRITABLE = ctypes.c_int32(int("0x00020000", 0))
DCAMPROP_ATTR_READABLE = ctypes.c_int32(int("0x00010000", 0))

DCAM_IDPROP_SENSORTEMPERATURE = ctypes.c_int32(int("0x00200310", 0))
DCAM_IDPROP_SENSORTEMPERATURETARGET = ctypes.c_int32(int("0x00200330", 0))

DCAMWAIT_CAPEVENT_FRAMEREADY = ctypes.c_int32(int("0x0002", 0))
DCAMWAIT_TIMEOUT_INFINITE = ctypes.c_int32(int("0x80000000", 0))

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


class DCAMBUF_ATTACH(ctypes.Structure):
    """
    Specifies the buffer to attach to the camera.
    """
    _fields_ = [("size", ctypes.c_int32),
                ("iKind", ctypes.c_int32),
                ("buffer", ctypes.POINTER(ctypes.c_void_p)),
                ("buffercount", ctypes.c_int32)]


class DCAM_TIMESTAMP(ctypes.Structure):
    _fields_ = [("sec", ctypes.c_uint32),
                ("microsec", ctypes.c_int32)]

class DCAMBUF_FRAME(ctypes.Structure):
    """
    Frame structure for dcambuf_lockframe().
    """
    _fields_ = [("size", ctypes.c_int32),
                ("iKind", ctypes.c_int32),
                ("option", ctypes.c_int32),
                ("iFrame", ctypes.c_int32),
                ("buf", ctypes.c_void_p),
                ("rowbytes", ctypes.c_int32),
                ("type", ctypes.c_int32),
                ("width", ctypes.c_int32),
                ("height", ctypes.c_int32),
                ("left", ctypes.c_int32),
                ("top", ctypes.c_int32),
                ("timestamp_sec", ctypes.c_int32),
                ("timestamp_usec", ctypes.c_int32),
                ("framestamp", ctypes.c_int32),
                ("camerastamp", ctypes.c_int32)]

class DCAMWAIT_OPEN(ctypes.Structure):
    """
    Create wait handle.
    """
    _fields_ = [("size", ctypes.c_int32),
                ("supportevent", ctypes.c_int32),
                ("hdcamwait", ctypes.c_void_p),
                ("hdcam", ctypes.c_void_p)]

class DCAMWAIT_START(ctypes.Structure):
    """
    Create wait start handle.
    """
    _fields_ = [("size", ctypes.c_int32),
                ("eventhappened", ctypes.c_int32),
                ("eventmask", ctypes.c_int32),
                ("timeout", ctypes.c_int32)]

class DCAMCAP_TRANSFERINFO(ctypes.Structure):
    """
    Transfer info structure for newest frame and frame count.
    """
    _fields_ = [("size", ctypes.c_int32),
                ("iKind", ctypes.c_int32),
                ("nNewestFrameIndex", ctypes.c_int32),
                ("nFrameCount", ctypes.c_int32)]


HDCAM = ctypes.c_void_p

#Function prototypes
####################
dcam.dcamprop_getnextid.argtypes = [HDCAM, ctypes.c_void_p, ctypes.c_int32]
dcam.dcamdev_getstring.argtypes = [HDCAM, ctypes.c_void_p]
dcam.dcamprop_getname.argtypes = [HDCAM, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
dcam.dcamprop_getattr.argtypes = [HDCAM, ctypes.c_void_p]
dcam.dcamprop_getvalue.argtypes = [HDCAM, ctypes.c_int32, ctypes.c_void_p]
dcam.dcamprop_setvalue.argtypes = [HDCAM, ctypes.c_int32, ctypes.c_double]
dcam.dcamprop_getvaluetext.argtypes = [HDCAM, ctypes.c_void_p]
dcam.dcamdev_close.argtypes = [HDCAM,]

dcam.dcambuf_alloc.argtypes=[HDCAM,ctypes.c_int32]
dcam.dcambuf_release.argtypes=[HDCAM,ctypes.c_int32]
dcam.dcambuf_lockframe.argtypes=[HDCAM,ctypes.c_void_p]

dcam.dcamcap_start.argtypes=[HDCAM,ctypes.c_int32]
dcam.dcamcap_stop.argtypes=[HDCAM,]
dcam.dcamcap_transferinfo.argtypes=[HDCAM,ctypes.c_void_p]
dcam.dcamcap_firetrigger.argtypes=[HDCAM, ctypes.c_int32]

dcam.dcamwait_start.argtypes=[ctypes.c_void_p,ctypes.c_void_p]

class DCAMException(Exception):
    """
    Raises an Exception related to Hamamatsu DCAM.
    """
    def __init__(self, message):
        Exception.__init__(self, message)


# Helper functions
def convertPropertyName(name):
    return name.lower().replace(" ", "_")


class camReg(object):
    """
    Keep track of the number of cameras initialised so we can initialise and
    finalise the library.
    """
    numCameras = -1
    maxCameras = 0

    @classmethod
    def regCamera(cls):
        if cls.numCameras == -1:
            # Initialize the API
            paraminit = DCAMAPI_INIT()
            paraminit.size = ctypes.sizeof(paraminit)
            if int(dcam.dcamapi_init(ctypes.byref(paraminit))) < 0:
                raise DCAMException("DCAM initialization failed.")
            cls.maxCameras = paraminit.iDeviceCount

        cls.numCameras += 1

    @classmethod
    def unregCamera(cls):
        cls.numCameras -= 1
        if cls.numCameras == 0:
            dcam.dcamapi_uninit()

# make sure the library is intitalised
camReg.regCamera()

class HamamatsuDCAM(Camera):

    def __init__(self, camNum):
        Camera.__init__(self)
        self.camNum = camNum
        self.handle = ctypes.c_void_p(0)
        self.properties = {}
        self.camWidth = 0
        self.camHeight = 0

    def Init(self):
        # TODO: Note that this check does not prevent someone from creating
        #       multiple camera ones (or twos, etc.)
        #print "Cam num: " + str(self.camNum)
        #print "Max cams: " + str(camReg.maxCameras)
        if self.camNum < camReg.maxCameras:
            paramopen = DCAMDEV_OPEN()
            paramopen.size = ctypes.sizeof(paramopen)
            paramopen.index = self.camNum
            self.checkStatus(dcam.dcamdev_open(ctypes.byref(paramopen)),
                             "dcamdev_open")
            self.handle = paramopen.hdcam
            camReg.regCamera()

            self.properties = self.getCamProperties()

            self._camera_model = self.getCamInfo(DCAM_IDSTR_MODEL)
            self._serial_number = self.getCamInfo(DCAM_IDSTR_CAMERAID).strip('S/N: ')

            self.SetIntegTime(0.1)
            
    def GetSerialNumber(self):
        return self._serial_number
    
    def GetHeadModel(self):
        return  self._camera_model

    def StartExposure(self):
        self.StopAq()
        #self._temp = self.getCamPropValue(DCAM_IDPROP_SENSORTEMPERATURE)
        #self._frameRate = self._intTime

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

        return c_buf.value.decode()

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

            properties[text.value.decode()] = iProp.value

            if dcam.dcamprop_getnextid(self.handle, ctypes.byref(iProp),
                                       DCAMPROP_OPTION_SUPPORT) < 0:
                break

        return properties

    def getCamPropAttr(self, prop_name):
        """
        Get DCAM property attributes for property iProp.

        Parameters
        ----------
        prop_name : str
            DCAM property string (e.g. 'EXPOSURE TIME')

        Returns
        -------
        attr : DCAMPROP_ATTR
            Struct of DCAM property attributes, which includes range and mode.
        """
        attr = DCAMPROP_ATTR()
        attr.cbSize = ctypes.sizeof(attr)
        attr.iProp = self.checkProp(prop_name)

        self.checkStatus(dcam.dcamprop_getattr(self.handle,
                                           ctypes.byref(attr)),
                     "dcamprop_getattr")

        return attr

    def getCamPropRange(self, prop_name):
        """
        Get the range of allowed values for this camera property.
        """

        attr = self.getCamPropAttr(prop_name)
        ub = attr.valuemax
        lb = attr.valuemin

        return lb, ub

    def getCamPropRW(self, prop_name):
        attr = self.getCamPropAttr(prop_name)
        rw = attr.attribute
        return (rw & int(DCAMPROP_ATTR_READABLE.value)), \
               (rw & int(DCAMPROP_ATTR_WRITABLE.value))

    def getCamPropValue(self, prop_name):
        """
        Get DCAM property value for property iProp.

        Parameters
        ----------
        prop_name : str
            DCAM property string (e.g. 'EXPOSURE TIME')

        Returns
        -------
        value : float
            Value of DCAM property.
        """
        # Get the property id (if the property exists)
        iProp = self.checkProp(prop_name)

        # Is this property readable?
        r, _ = self.getCamPropRW(prop_name)
        if not r:
            raise DCAMException(prop_name + " is not readable.")

        # Get the property value
        val = ctypes.c_double(0)
        self.checkStatus(dcam.dcamprop_getvalue(self.handle, iProp,
                                                ctypes.byref(val)),
                         "dcamprop_getvalue")

        return float(val.value)

    def setCamPropValue(self, prop_name, val):
        """
        Get DCAM property value for property iProp.

        Parameters
        ----------
        prop_name : str
            DCAM property string (e.g. 'EXPOSURE TIME')
        value : float
            Value to set DCAM property.

        Returns
        -------
        None.
        """
        # Get the property ID (if the property exists)
        iProp = self.checkProp(prop_name)

        # Is this property writable?
        _, w = self.getCamPropRW(prop_name)
        if not w:
            raise DCAMException(prop_name + " is not writable.")

        # Is our value in this property's range?
        lb, ub = self.getCamPropRange(prop_name)
        if val < lb or val > ub:
            raise ValueError("Value out of range " + str(lb) + " to " + str(ub))

        # Set the property value
        self.checkStatus(dcam.dcamprop_setvalue(self.handle, iProp,
                                                ctypes.c_double(val)),
                        "dcamprop_setvalue")

    def checkProp(self, prop_name):
        """
        Checks if camera property exists and returns the property ID. If
        property does not exist, an exception is raised.

        Parameters
        ----------
        prop_name : str
            The name of the camera property.

        Returns
        -------
        c_int32
            Camera property ID.
        """
        if prop_name in self.properties:
            return ctypes.c_int32(self.properties[prop_name])
        else:
            raise DCAMException(prop_name + " is not a camera property or "
                                            "camera is not initialized (use "
                                            "[camera object].Init()).")

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

    def getCamPropValueText(self, prop_name):
        """returns text associated with current property setting. Assumes
        prop_name is a text/mode setting, not just a float.

        Parameters
        ----------
        prop_name : str
             DCAM property string (e.g. 'OUTPUT TRIGGER SOURCE[0]')
        
        Returns
        -------
        value_text : str
            name associated with current property value
        """
        # Get the property id (if the property exists)
        iProp = self.checkProp(prop_name)

        # Get the property value
        val = ctypes.c_double(0)
        self.checkStatus(dcam.dcamprop_getvalue(self.handle, iProp, 
                                                ctypes.byref(val)),
                         "dcamprop_getvalue")

        # get the property value text
        c_buf_len = 64
        c_buf = ctypes.create_string_buffer(c_buf_len)
        value_text = DCAMPROP_VALUETEXT()
        value_text.iProp = iProp
        value_text.value = val
        value_text.size = ctypes.sizeof(value_text)
        value_text.text = ctypes.addressof(c_buf)
        value_text.textbytes = ctypes.c_int32(c_buf_len)
        self.checkStatus(dcam.dcamprop_getvaluetext(self.handle, 
                                                    ctypes.byref(value_text)),
                         "dcamprop_getvaluetext")
        return value_text.text.decode()
    
    def getCamPropValueTextOptions(self, prop_name):
        """enumerate available options for this camera property, with 
        associated text name. Assumes prop_name is a text/mode setting. Helpful 
        for development - see Notes.

        Parameters
        ----------
        prop_name : str
             DCAM property string (e.g. 'OUTPUT TRIGGER KIND[0]')
        
        Returns
        -------
        options : dict
            keys are floating point settings that would actually be passed to 
            DCAM functions, and values are text name
        
        Notes
        -----
        DCAM API (SKD4_v21066291) appears to have a quirk or two, e.g. 
        'OUTPUT TRIGGER SOURCE[0]' property has a property range of 2.0 to 5.0,
        for 5 text-value options in the SDK, but returns 
        {2.0: 'READOUT END', 3.0: 'VSYNC', 
        4.0: 'DCAM error for dcamprop_getvaluetext with return value -2147481567: Invalid Value!', 
        5.0: 'DCAM error for dcamprop_getvaluetext with return value -2147481567: Invalid Value!'}
        yet dcamprop.h shows 5 options:
        1: EXPOSURE, 2: READOUT_END, 3: VSYNC, 4: HSYNC, 6,: TRIGGER
        A bit odd.
        """
        # Get the property id (if the property exists)
        iProp = self.checkProp(prop_name)

        # get the property range
        lb, ub = self.getCamPropRange(prop_name)

        options = {}
        for v in np.arange(lb, ub):
            try:
                # get the property value text
                c_buf_len = 64
                c_buf = ctypes.create_string_buffer(c_buf_len)
                value_text = DCAMPROP_VALUETEXT()
                value_text.iProp = iProp
                value_text.value = ctypes.c_double(v)
                value_text.size = ctypes.sizeof(value_text)
                value_text.text = ctypes.addressof(c_buf)
                value_text.textbytes = ctypes.c_int32(c_buf_len)
                self.checkStatus(dcam.dcamprop_getvaluetext(self.handle, ctypes.byref(value_text)), "dcamprop_getvaluetext")
                options[v] = value_text.text.decode()
            except DCAMException as e:
                # DCAM is a special API - sometimes get invalid value errors for in-range values
                options[v] = str(e)
        
        return options

    def Shutdown(self):
        self.checkStatus(dcam.dcamdev_close(self.handle), "dcamdev_close")
        camReg.unregCamera()
