__all__ = [ ]

import os
import sys
import textwrap
import numpy as np
from numpy import ctypeslib
import ctypes
import ctypes.util

import warnings
import six

from .uc480_h import *

#wintypes fails on linux, copy definitions here so that we are cross platform
BYTE = ctypes.c_byte
WORD = ctypes.c_uint16
DWORD = ctypes.c_uint32
BOOL = ctypes.c_long

HANDLE = ctypes.c_void_p

HCAM = HANDLE
HWND = HANDLE

ULONG = ctypes.c_ulong
LONG = ctypes.c_long
USHORT = ctypes.c_ushort
SHORT = ctypes.c_short
INT = ctypes.c_int
    
    
from ctypes import Structure
c_char = ctypes.c_byte
c_char_p = ctypes.POINTER(ctypes.c_byte)
c_int_p = ctypes.POINTER(ctypes.c_int)
from ctypes import c_int
IS_CHAR = ctypes.c_byte

BINNING_FACTORS = [2, 3, 4, 5, 6, 8, 16]

class CAMINFO(ctypes.Structure):
    _fields_ = [("SerNo",ctypes.c_char*12),  # (11 char)
                ("ID",ctypes.c_char*20),  # e.g. "Company Name"
                ("Version",ctypes.c_char*10),  # e.g. "V1.00"  (9 char)
                ("Date",ctypes.c_char*12),  # e.g "11.03.2004" (11 char)
                ("Select",ctypes.c_byte	 ),  # 0 (contains camera select number for multi camera support)
                ("Type",ctypes.c_byte	 ),  # 1 (contains camera type)
                ("Reserved",ctypes.c_char*8 )]  # (7 char)
#} CAMINFO, *PCAMINFO;
PCAMINFO = ctypes.POINTER(CAMINFO)

class IS_RECT(ctypes.Structure):
    _fields_ = [('s32X', ctypes.c_int32), 
                ('s32Y', ctypes.c_int32),
                ('s32Width', ctypes.c_int32),
                ('s32Height', ctypes.c_int32)]

class SENSORINFO(Structure):
    _fields_ = [("SensorID" , WORD   ),  # e.g. IS_SENSOR_C0640R13M
                ("strSensorName" , IS_CHAR*32),  # e.g. "C0640R13M"  	
                ("nColorMode" , c_char ),  # e.g. IS_COLORMODE_BAYER  
                ("nMaxWidth" , DWORD  ),  # e.g. 1280  
                ("nMaxHeight" , DWORD  ),  # e.g. 1024  
                ("bMasterGain" , BOOL   ),  # e.g. FALSE  
                ("bRGain" , BOOL   ),  # e.g. TRUE  
                ("bGGain" , BOOL   ),  # e.g. TRUE  
                ("bBGain" , BOOL   ),  # e.g. TRUE  
                ("bGlobShutter" , BOOL   ),  # e.g. TRUE  
                ("Reserved[16]" , c_char*16)]  #  not used 				
#typedef struct _SENSORINFO{
#} SENSORINFO, *PSENSORINFO;
PSENSORINFO = ctypes.POINTER(SENSORINFO)

class REVISIONINFO(ctypes.Structure):
#{
    _fields_ = [("size", WORD  ),       # 2
                ("Sensor", WORD  ),       # 2
                ("Cypress", WORD  ),       # 2
                ("Blackfin", DWORD ),       # 4
                ("DspFirmware", WORD  ),       # 2
                ("USB_Board", WORD  ),       # 2
                ("Sensor_Board", WORD  ),       # 2
                ("Processing_Board", WORD  ),       # 2
                ("Memory_Board", WORD  ),       # 2
                ("Housing", WORD  ),       # 2
                ("Filter", WORD  ),       # 2
                ("Timing_Board", WORD  ),       # 2
                ("Product", WORD  ),       # 2
                ("reserved[100]", BYTE*100  )]       # --128
#} REVISIONINFO, *PREVISIONINFO;
PREVISIONINFO = ctypes.POINTER(REVISIONINFO)



class UC480_CAMERA_INFO(ctypes.Structure):
    _fields_ = [("dwCameraID",DWORD  ),	# this is the user defineable camera ID
                ("dwDeviceID",DWORD  ),	# this is the systems enumeration ID
                ("dwSensorID",DWORD  ),	# this is the sensor ID e.g. IS_SENSOR_C0640R13M
                ("dwInUse",DWORD  ),	# flag, whether the camera is in use or not
                ("SerNo[16]",IS_CHAR*16),	# serial numer of the camera
                ("Model[16]",IS_CHAR*16),	# model name of the camera
                ("dwReserved[16]",DWORD  *16)] #
#}UC480_CAMERA_INFO, *PUC480_CAMERA_INFO;
PUC480_CAMERA_INFO = ctypes.POINTER(UC480_CAMERA_INFO)

class UEYE_CAMERA_INFO(ctypes.Structure):
    _fields_ = [("dwCameraID",DWORD  ),	# this is the user defineable camera ID
                ("dwDeviceID",DWORD  ),	# this is the systems enumeration ID
                ("dwSensorID",DWORD  ),	# this is the sensor ID e.g. IS_SENSOR_C0640R13M
                ("dwInUse",DWORD  ),	# flag, whether the camera is in use or not
                ("SerNo[16]",IS_CHAR*16),	# serial numer of the camera
                ("Model[16]",IS_CHAR*16),	# model name of the camera
                       ("dwStatus", DWORD),
                       ("dwReserved[2]", DWORD*2),
                       ("FullModelName", IS_CHAR*32),
                ("dwReserved2[5]",DWORD  *5)] #
#}UC480_CAMERA_INFO, *PUC480_CAMERA_INFO;
PUEYE_CAMERA_INFO = ctypes.POINTER(UEYE_CAMERA_INFO)

libuc480=None # initialise at module level
cameraType = None # initialise

def loadLibrary(cameratype='uc480'):
    import platform
    arch, plat = platform.architecture()

    global libuc480
    global cameraType

    # make sure we load only once - if requested again make sure cameratype matches with already loaded
    if libuc480 is not None:
        if cameratype != cameraType:
            raise ValueError('requesting loading library for type %s but already loaded for type %s - conflict!'
                             % (cameratype, cameraType))
        return
    
    cameraType = cameratype # store for later availability by outside modules
    #libuc480 = ctypes.cdll.LoadLibrary(lib)
    if plat.startswith('Windows'):
        if cameratype=='uc480':
            try:
                libuc480 = ctypes.WinDLL('uc480_64')
            except OSError:
                # see https://stackoverflow.com/questions/59330863/cant-import-dll-module-in-python
                # winmode=0 enforces windows default dll search mechanism including searching the path set
                # necessary since python 3.8.x
                libuc480 = ctypes.WinDLL('uc480_64',winmode=0)
            print("loading uc480_64")
        elif cameratype=='ueye':
            try:
                libuc480 = ctypes.WinDLL('ueye_api_64')
            except OSError:
                # see https://stackoverflow.com/questions/59330863/cant-import-dll-module-in-python
                # winmode=0 enforces windows default dll search mechanism including searching the path set
                # necessary since python 3.8.x
                libuc480 = ctypes.WinDLL('ueye_api_64',winmode=0))  
                print("loading ueye_api_64")
        else:
                raise RuntimeError("unknown camera type")
    else:
        #linux or osx
        if cameratype=='ueye':
                libuc480 = ctypes.CDLL('libueye_api.so')
                print("loading 'libueye_api.so'")
        else:
                raise RuntimeError("unknown camera type")


def CALL(name, *args):
    """
    Calls libuc480 function "name" and arguments "args".
    """
    funcname = 'is_' + name
    #print name
    func = getattr(libuc480, funcname)
    new_args = []
    for a in args:
        if isinstance(a, six.string_types) and not isinstance(a, bytes):
            print((name, 'argument',a, 'is unicode'))
            new_args.append(a.encode())
        else:
            new_args.append (a)
    r = func(*new_args)
    #print r
  # r = CHK(r, funcname, *new_args)
    return r

