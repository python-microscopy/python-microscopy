# -*- coding: utf-8 -*-

"""
Created on Fri May 14 2021

@author: zacsimile

To run, install the pco. sdk, available at 
https://www.pco.de/software/development-tools/pcosdk/, as admin.

The goal of this file is to provide Python access to pco.sdk as closely to how it
is implemented in pco.sdk as possible. There's no mapping of flags to readable
strings such as "on"/"off". The idea is to have raw access to the DLL. For a
friendlier--but less comprehensive--version, see sdk.py in https://pypi.org/project/pco/.

For more information, see the pco.sdk manual,
https://www.pco.de/fileadmin/fileadmin/user_upload/pco-manuals/pco.sdk_manual.pdf

NOTE: I cheated a little off https://pypi.org/project/pco/.
"""

import ctypes
import ctypes.wintypes
import platform
import os
import sys

os_desc = platform.system()
if os_desc != 'Windows':
    raise Exception("Operating system is not supported.")

dll = 'SC2_Cam.dll'
dll_path = "C:\\Program Files (x86)\\PCO Digital Camera Toolbox\\pco.sdk\\bin64\\" + dll
try:
    sc2_cam = ctypes.windll.LoadLibrary(dll_path)
except:
    raise

# ---------------------------------------------------------------------
# Structures and types
# ---------------------------------------------------------------------
class PCO_Description(ctypes.Structure):
    # _pack_ = 1
    _fields_ = [("wSize", ctypes.wintypes.WORD),
                ("wSensorTypeDESC", ctypes.wintypes.WORD),
                ("wSensorSubTypeDESC", ctypes.wintypes.WORD),
                ("wMaxHorzResStdDESC", ctypes.wintypes.WORD),
                ("wMaxVertResStdDESC", ctypes.wintypes.WORD),
                ("wMaxHorzResExtDESC", ctypes.wintypes.WORD),
                ("wMaxVertResExtDESC", ctypes.wintypes.WORD),
                ("wDynResDESC", ctypes.wintypes.WORD),
                ("wMaxBinHorzDESC", ctypes.wintypes.WORD),
                ("wBinHorzSteppingDESC", ctypes.wintypes.WORD),
                ("wMaxBinVertDESC", ctypes.wintypes.WORD),
                ("wBinVertSteppingDESC", ctypes.wintypes.WORD),
                ("wRoiHorStepsDESC", ctypes.wintypes.WORD),
                ("wRoiVertStepsDESC", ctypes.wintypes.WORD),
                ("wNumADCsDESC", ctypes.wintypes.WORD),
                ("wMinSizeHorzDESC", ctypes.wintypes.WORD),
                ("dwPixelRateDESC", ctypes.wintypes.DWORD * 4),
                ("ZzdwDummypr", ctypes.wintypes.DWORD * 20),
                ("wConvFactDESC", ctypes.wintypes.WORD * 4),
                ("sCoolingSetpoints", ctypes.c_short * 10),
                ("ZZdwDummycv", ctypes.wintypes.WORD * 8),
                ("wSoftRoiHorStepsDESC", ctypes.wintypes.WORD),
                ("wSoftRoiVertStepsDESC", ctypes.wintypes.WORD),
                ("wIRDESC", ctypes.wintypes.WORD),
                ("wMinSizeVertDESC", ctypes.wintypes.WORD),
                ("dwMinDelayDESC", ctypes.wintypes.DWORD),
                ("dwMaxDelayDESC", ctypes.wintypes.DWORD),
                ("dwMinDelayStepDESC", ctypes.wintypes.DWORD),
                ("dwMinExposDESC", ctypes.wintypes.DWORD),
                ("dwMaxExposDESC", ctypes.wintypes.DWORD),
                ("dwMinExposStepDESC", ctypes.wintypes.DWORD),
                ("dwMinDelayIRDESC", ctypes.wintypes.DWORD),
                ("dwMaxDelayIRDESC", ctypes.wintypes.DWORD),
                ("dwMinExposIRDESC", ctypes.wintypes.DWORD),
                ("dwMaxExposIRDESC", ctypes.wintypes.DWORD),
                ("wTimeTableDESC", ctypes.wintypes.WORD),
                ("wDoubleImageDESC", ctypes.wintypes.WORD),
                ("sMinCoolSetDESC", ctypes.c_short),
                ("sMaxCoolSetDESC", ctypes.c_short),
                ("sDefaultCoolSetDESC", ctypes.c_short),
                ("wPowerDownModeDESC", ctypes.wintypes.WORD),
                ("wOffsetRegulationDESC", ctypes.wintypes.WORD),
                ("wColorPatternDESC", ctypes.wintypes.WORD),
                ("wPatternTypeDESC", ctypes.wintypes.WORD),
                ("wDummy1", ctypes.wintypes.WORD),
                ("wDummy2", ctypes.wintypes.WORD),
                ("wNumCoolingSetpoints", ctypes.wintypes.WORD),
                ("dwGeneralCapsDESC1", ctypes.wintypes.DWORD),
                ("dwGeneralCapsDESC2", ctypes.wintypes.DWORD),
                ("dwExtSyncFrequency", ctypes.wintypes.DWORD * 4),
                ("dwGeneralCapsDESC3", ctypes.wintypes.DWORD),
                ("dwGeneralCapsDESC4", ctypes.wintypes.DWORD),
                ("ZzdwDummy", ctypes.wintypes.DWORD)]
class PCO_SC2_Hardware_DESC(ctypes.Structure):
    # _pack_ = 1
    _fields_ = [
        ("szName", ctypes.c_char * 16),
        ("wBatchNo", ctypes.wintypes.WORD),
        ("wRevision", ctypes.wintypes.WORD),
        ("wVariant", ctypes.wintypes.WORD),
        ("ZZwDummy", ctypes.wintypes.WORD * 20)]

class PCO_SC2_Firmware_DESC(ctypes.Structure):
    # _pack_ = 1
    _fields_ = [("szName", ctypes.c_char * 16),
                ("bMinorRev", ctypes.c_byte),
                ("bMajorRev", ctypes.c_byte),
                ("wVariant", ctypes.wintypes.WORD),
                ("ZZwDummy", ctypes.wintypes.WORD * 22)]

class PCO_HW_Vers(ctypes.Structure):
    #_pack_ = 1
    _fields_ = [
        ("wBoardNum", ctypes.wintypes.WORD),
        ("Board", PCO_SC2_Hardware_DESC * 10)]

class PCO_FW_Vers(ctypes.Structure):
    # _pack_ = 1
    _fields_ = [("wDeviceNum", ctypes.wintypes.WORD),
                ("Device", PCO_SC2_Firmware_DESC * 10)]

class PCO_CameraType(ctypes.Structure):
    # _pack_ = 1
    _fields_ = [("wSize", ctypes.wintypes.WORD),
                ("wCamType", ctypes.wintypes.WORD),
                ("wCamSubType", ctypes.wintypes.WORD),
                ("ZZwAlignDummy1", ctypes.wintypes.WORD),
                ("dwSerialNumber", ctypes.wintypes.DWORD),
                ("dwHWVersion", ctypes.wintypes.DWORD),
                ("dwFWVersion", ctypes.wintypes.DWORD),
                ("wInterfaceType", ctypes.wintypes.WORD),
                ("strHardwareVersion", PCO_HW_Vers),
                ("strFirmwareVersion", PCO_FW_Vers),
                ("ZZwDummy", ctypes.wintypes.PWORD)]  # ("ZZwDummy", ctypes.wintypes.WORD * 39)

class PCO_ImageTiming(ctypes.Structure):
    # _pack_ = 1
    _fields_ = [("wSize", ctypes.wintypes.WORD),
                ("wDummy", ctypes.wintypes.WORD),
                ("FrameTime_ns", ctypes.wintypes.DWORD),
                ("FrameTime_s", ctypes.wintypes.DWORD),
                ("ExposureTime_ns", ctypes.wintypes.DWORD),
                ("ExposureTime_s", ctypes.wintypes.DWORD),
                ("TriggerSystemDelay_ns", ctypes.wintypes.DWORD),
                ("TriggerSystemJitter_ns", ctypes.wintypes.DWORD),
                ("TriggerDelay_ns", ctypes.wintypes.DWORD),
                ("TriggerDelay_s", ctypes.wintypes.DWORD),
                ("ZZdwDummy", ctypes.wintypes.DWORD)]

class PCO_Recording(ctypes.Structure):
    # _pack_ = 1
    _fields_ = [("wSize", ctypes.wintypes.WORD),
                ("wStorageMode", ctypes.wintypes.WORD),
                ("wRecSubmode", ctypes.wintypes.WORD),
                ("wRecState", ctypes.wintypes.WORD),
                ("wAcquMode", ctypes.wintypes.WORD),
                ("wAcquEnableStatus", ctypes.wintypes.WORD),
                ("ucDay", ctypes.c_byte),
                ("ucMonth", ctypes.c_byte),
                ("wYear", ctypes.wintypes.WORD),
                ("wHour", ctypes.wintypes.WORD),
                ("ucMin", ctypes.c_byte),
                ("ucSec", ctypes.c_byte),
                ("wTimeStampMode", ctypes.wintypes.WORD),
                ("wRecordStopEventMode", ctypes.wintypes.WORD),
                ("dwRecordStopDelayImages", ctypes.wintypes.DWORD),
                ("wMetaDataMode", ctypes.wintypes.WORD),
                ("wMetaDataSize", ctypes.wintypes.WORD),
                ("wMetaDataVersion", ctypes.wintypes.WORD),
                ("ZZwDummy1", ctypes.wintypes.WORD),
                ("dwAcquModeExNumberImages", ctypes.wintypes.DWORD),
                ("dwAcquModeExReserved", ctypes.wintypes.DWORD * 4),
                ("ZZwDummy", ctypes.wintypes.WORD * 22)]

class PCO_Metadata_Struct(ctypes.Structure):
    # _pack_ = 1
    _fields_ = [("wSize", ctypes.wintypes.WORD),
                ("wVersion", ctypes.wintypes.WORD),
                ("bIMAGE_COUNTER_BCD", ctypes.c_byte * 4),
                ("bIMAGE_TIME_US_BCD", ctypes.c_byte * 3),
                ("bIMAGE_TIME_SEC_BCD", ctypes.c_byte),
                ("bIMAGE_TIME_MIN_BCD", ctypes.c_byte),
                ("bIMAGE_TIME_HOUR_BCD", ctypes.c_byte),
                ("bIMAGE_TIME_DAY_BCD", ctypes.c_byte),
                ("bIMAGE_TIME_MON_BCD", ctypes.c_byte),
                ("bIMAGE_TIME_YEAR_BCD", ctypes.c_byte),
                ("bIMAGE_TIME_STATUS", ctypes.c_byte),
                ("wEXPOSURE_TIME_BASE", ctypes.wintypes.WORD),
                ("dwEXPOSURE_TIME", ctypes.wintypes.DWORD),
                ("dwFRAMERATE_MILLIHZ", ctypes.wintypes.DWORD),
                ("sSENSOR_TEMPERATURE", ctypes.c_short),
                ("wIMAGE_SIZE_X", ctypes.wintypes.WORD),
                ("wIMAGE_SIZE_Y", ctypes.wintypes.WORD),
                ("bBINNING_X", ctypes.c_byte),
                ("bBINNING_Y", ctypes.c_byte),
                ("dwSENSOR_READOUT_FREQUENCY", ctypes.wintypes.DWORD),
                ("wSENSOR_CONV_FACTOR", ctypes.wintypes.WORD),
                ("dwCAMERA_SERIAL_NO", ctypes.wintypes.DWORD),
                ("wCAMERA_TYPE", ctypes.wintypes.WORD),
                ("bBIT_RESOLUTION", ctypes.c_byte),
                ("bSYNC_STATUS", ctypes.c_byte),
                ("wDARK_OFFSET", ctypes.wintypes.WORD),
                ("bTRIGGER_MODE", ctypes.c_byte),
                ("bDOUBLE_IMAGE_MODE", ctypes.c_byte),
                ("bCAMERA_SYNC_MODE", ctypes.c_byte),
                ("bIMAGE_TYPE", ctypes.c_byte),
                ("wCOLOR_PATTERN", ctypes.wintypes.WORD)]

PCO_NOERROR = 0x00000000
class PCO_Buflist(ctypes.Structure):
    # _pack_ = 1
    _fields_ = [("SBufNr", ctypes.c_short),
                ("reserved", ctypes.wintypes.WORD),
                ("dwStatusDll", ctypes.wintypes.DWORD),  # PCO_BUFFER_ALLOCATED, PCO_BUFFER_CREATED, PCO_BUFFER_EXTERNAL, PCO_BUFFER_SET
                ("dwStatusDrv", ctypes.wintypes.DWORD)]  # PCO_NOERROR or see pco.sdk for error codes

PCO_CAMERA_TYPES = {
    0x1300 : 'pco.edge 5.5 CL',
    0x1302 : 'pco.edge 4.2 CL',
    0x1310 : 'pco.edge GL',
    0x1320 : 'pco.edge USB3',
    0x1340 : 'pco.edge CLHS',
    0x1304 : 'pco.edge MT',
    0x1000 : 'pco.dimax',
    0x1010 : 'pco.dimax_TV',
    0x1020 : 'pco.dimax CS',
    0x1400 : 'pco.flim',
    0x1500 : 'pco.panda',
    0x0800 : 'pco.pixelfly usb',
    0x0100 : 'pco.1200HS',
    0x0200 : 'pco.1300',
    0x0220 : 'pco.1600',
    0x0240 : 'pco.2000',
    0x0260 : 'pco.4000',
    0x0830 : 'pco.1400'
}

PCO_INTERFACE_TYPES = {
    0x0001 : 'FireWire',
    0x0002 : 'Camera Link',
    0x0003 : 'USB 2.0',
    0x0004 : 'GigE',
    0x0005 : 'Serial Interface',
    0x0006 : 'USB 3.0',
    0x0007 : 'CLHS'
}

PCO_TRIGGER_MODES = {
    0x0000 : 'auto sequence',
    0x0001 : 'software trigger',
    0x0002 : 'external exposure start & software trigger',
    0x0003 : 'external exposure control',
    0x0004 : 'external synchronized',
    0x0005 : 'fast external exposure control',
    0x0006 : 'external CDS control',
    0x0007 : 'slow external exposure control',
    0x0102 : 'external synchronized HDSDI'
}

# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------
class PcoSdkException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

# TODO: Instead of wrapping everything with check_status, move to ctypes
# errcheck https://docs.python.org/3/library/ctypes.html#ctypes._FuncPtr.errcheck
def check_status(err, function_name=None):
    if err:
        if function_name is None:
            function_name = sys._getframe(1).f_code.co_name  # who called me?
        err_desc = get_error_text(err)
        raise PcoSdkException(f"Error during {function_name}: {err_desc}")

# ---------------------------------------------------------------------
# 2.1.1 PCO_OpenCamera
# ---------------------------------------------------------------------
sc2_cam.PCO_OpenCamera.argtypes = [ctypes.POINTER(ctypes.wintypes.HANDLE), 
                                   ctypes.wintypes.WORD]
def open_camera():
    """
    Open a pco. camera, return the handle to that camera.
    This uses a scan process, so call this multiple times 
    to open multiple cameras. Call get_camera_type() to
    figure out which camera was grabbed.

    To open a specific camera, use open_camera_ex().

    Returns
    -------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    """
    handle = ctypes.c_void_p(0)
    cam_num = ctypes.wintypes.WORD()  # unused
    check_status(sc2_cam.PCO_OpenCamera(handle, cam_num))
    return handle
# ---------------------------------------------------------------------
# 2.1.2 PCO_OpenCameraEx
# ---------------------------------------------------------------------
def open_camera_ex(**kwargs):
    """
    Choose which camera to open, rather than using a scan mode.
    """
    raise NotImplementedError("Not implemented, but shouldn't be too hard. Check pco.sdk for details.")

# ---------------------------------------------------------------------
# 2.1.3 PCO_CloseCamera
# ---------------------------------------------------------------------
sc2_cam.PCO_CloseCamera.argtypes = [ctypes.wintypes.HANDLE]
def close_camera(handle):
    """
    Close a pco. camera.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    """
    check_status(sc2_cam.PCO_CloseCamera(handle))

# ---------------------------------------------------------------------
# 2.1.4 PCO_ResetLib
# ---------------------------------------------------------------------
def reset_lib():
    """
    Reset the pco.sdk to its initial state. Can only be called when
    no cameras are open.
    """
    check_status(sc2_cam.PCO_ResetLib())

# ---------------------------------------------------------------------
# 2.2.1 PCO_GetCameraDescription
# ---------------------------------------------------------------------
sc2_cam.PCO_GetCameraDescription.argtypes = [ctypes.wintypes.HANDLE,
                                             ctypes.POINTER(PCO_Description)]
def get_camera_description(handle):
    """
    Get camera description. Includes properties such as image size, pixel rate, etctypes.,
    but not camera names, serial numbers, etctypes. For the latter, call get_camera_type().

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    PCO_Description
        Struct of camera properties.
    """
    desc = PCO_Description()
    desc.wSize = ctypes.sizeof(PCO_Description)
    check_status(sc2_cam.PCO_GetCameraDescription(handle, desc))
    return desc

# ---------------------------------------------------------------------
# 2.3.2 PCO_GetCameraType
# ---------------------------------------------------------------------
sc2_cam.PCO_GetCameraType.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PCO_CameraType)]
def get_camera_type(handle):
    """
    Get camera name, serial number, etc.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    
    Returns
    -------
    PCO_CameraType
        Struct containing hardware/firmware version, serial number, etc.
    """
    camera_type = PCO_CameraType()
    camera_type.wSize = ctypes.sizeof(PCO_CameraType)
    check_status(sc2_cam.PCO_GetCameraType(handle, camera_type))
    return camera_type

# ---------------------------------------------------------------------
# 2.3.3 PCO_GetCameraHealthStatus
# ---------------------------------------------------------------------
sc2_cam.PCO_GetCameraHealthStatus.argtypes = [ctypes.wintypes.HANDLE,
                                              ctypes.wintypes.PDWORD,
                                              ctypes.wintypes.PDWORD,
                                              ctypes.wintypes.PDWORD]
def get_camera_health_status(handle):
    """
    Get information about how well the camera is doing. See pco.sdk
    manual for bit definitions. Generally, things are OK as long as
    all the bits are 0, and bad otherwise. Except a status bit of 1
    is OK.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    warn : int
        Warning bit 
    err : int
        Error bit 
    status : int
        Status bit 
    """
    warn = ctypes.wintypes.DWORD()
    err = ctypes.wintypes.DWORD()
    status = ctypes.wintypes.DWORD()
    check_status(sc2_cam.PCO_GetCameraHealthStatus(handle, warn, err, status))
    return warn.value, err.value, status.value

# ---------------------------------------------------------------------
# 2.3.4 PCO_GetTemperature
# ---------------------------------------------------------------------
sc2_cam.PCO_GetTemperature.argtypes = [ctypes.wintypes.HANDLE, 
                                       ctypes.POINTER(ctypes.c_short),
                                       ctypes.POINTER(ctypes.c_short), 
                                       ctypes.POINTER(ctypes.c_short)]
def get_temperature(handle):
    """
    Get camera temperatures.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    ccd_temp : int
        Image sensor temp in 1/10 degree i.e. 100 = 10.0 C
    cam_temp : int
        Internal camera temp in C
    pow_temp : int
        Temp of additional devices (e.g. power supply)
    """
    ccd_temp = ctypes.c_short()
    cam_temp = ctypes.c_short()
    pow_temp = ctypes.c_short()
    check_status(sc2_cam.PCO_GetTemperature(handle, ccd_temp, cam_temp, pow_temp))
    return (ccd_temp.value/10.0), cam_temp.value, pow_temp.value


# ---------------------------------------------------------------------
# 2.3.6 PCO_GetCameraName
# ---------------------------------------------------------------------
sc2_cam.PCO_GetCameraName.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_char_p,
                                      ctypes.wintypes.WORD]
def get_camera_name(handle):
    """
    Get the camera name

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    string
        Camera name.
    """
    c_buf_len = 40
    c_buf = ctypes.create_string_buffer(c_buf_len)

    check_status(sc2_cam.PCO_GetCameraName(handle, c_buf, ctypes.wintypes.WORD(c_buf_len)))

    return str(c_buf.value.decode('ascii'))

# ---------------------------------------------------------------------
# 2.4.1 PCO_ArmCamera
# ---------------------------------------------------------------------
sc2_cam.PCO_ArmCamera.argtypes = [ctypes.wintypes.HANDLE]
def arm_camera(handle):
    """
    Prepare the camera for recording. Must be called after the change of 
    any camera parameter except for delay and exposure time.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    """
    check_status(sc2_cam.PCO_ArmCamera(handle))

# ---------------------------------------------------------------------
# 2.4.3 PCO_SetImageParameters
# ---------------------------------------------------------------------
sc2_cam.PCO_SetImageParameters.argtypes = [ctypes.wintypes.HANDLE, 
                                           ctypes.wintypes.WORD,
                                           ctypes.wintypes.WORD,
                                           ctypes.wintypes.DWORD,
                                           ctypes.c_void_p,
                                           ctypes.c_int]
PCO_IMAGEPARAMETERS_READ_WHILE_RECORDING = 0x00000001
PCO_IMAGEPARAMETERS_READ_FROM_SEGMENTS = 0x00000002
def set_image_parameters(handle, lx, ly, image_flag):
    """
    Set image parameters for internal allocated resources before image transfer.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    lx : int
        Image width
    ly : int
        Image height
    image_flag : int
        PCO_IMAGEPARAMETERS_READ_WHILE_RECORDING or PCO_IMAGEPARAMETERS_READ_FROM_SEGMENTS
    """
    param = ctypes.c_void_p(0)
    ilen = ctypes.c_int(0)
    check_status(sc2_cam.PCO_SetImageParameters(handle, ctypes.wintypes.WORD(lx), 
                                                ctypes.wintypes.WORD(ly), 
                                                ctypes.wintypes.DWORD(image_flag), 
                                                param, ilen))

# ---------------------------------------------------------------------
# 2.4.4 PCO_ResetSettingsToDefault
# ---------------------------------------------------------------------
sc2_cam.PCO_ResetSettingsToDefault.argtypes = [ctypes.wintypes.HANDLE]
def reset_settings_to_default(handle):
    """
    Reset the camera settings to default. Executed by default during
    camera power-up.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    """
    check_status(sc2_cam.PCO_ResetSettingsToDefault(handle))


# ---------------------------------------------------------------------
# 2.4.10 PCO_GetFanControlParameters
# ---------------------------------------------------------------------
sc2_cam.PCO_GetFanControlParameters.argtypes = [ctypes.wintypes.HANDLE, 
                                           ctypes.wintypes.PWORD,
                                           ctypes.wintypes.PWORD,
                                           ctypes.wintypes.PWORD,
                                           ctypes.wintypes.WORD]
PCO_FAN_CONTROL_MODE_AUTO = 0x0000
PCO_FAN_CONTROL_MODE_USER = 0x0001
def get_fan_control_parameters(handle):
    """
    Get fan speed

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    mode : int
        FAN_CONTROL_MODE_AUTO or FAN_CONTROL_MODE_USER
    speed : int
        Fan speed in range (slowest) 0..100 (fastest)
    """
    mode = ctypes.wintypes.WORD()
    speed = ctypes.wintypes.WORD()
    reserved = ctypes.wintypes.WORD()
    num_reserved = ctypes.wintypes.WORD(0)
    check_status(sc2_cam.PCO_GetFanControlParameters(handle, mode, speed, 
                                                     reserved, num_reserved))
    return mode.value, speed.value

# ---------------------------------------------------------------------
# 2.4.11 PCO_SetFanControlParameters
# ---------------------------------------------------------------------
sc2_cam.PCO_SetFanControlParameters.argtypes = [ctypes.wintypes.HANDLE, 
                                           ctypes.wintypes.WORD,
                                           ctypes.wintypes.WORD,
                                           ctypes.wintypes.WORD]
def set_fan_control_parameters(handle, mode, speed):
    """
    Set image parameters for internal allocated resources before image transfer.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    mode : int
        FAN_CONTROL_MODE_AUTO or FAN_CONTROL_MODE_USER
    speed : int
        Fan speed in range (slowest) 0..100 (fastest)
    """
    reserved = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_SetFanControlParameters(handle, ctypes.wintypes.WORD(mode), 
                                                     ctypes.wintypes.WORD(speed), 
                                                     reserved))

# ---------------------------------------------------------------------
# 2.5.3 PCO_GetSizes
# ---------------------------------------------------------------------
sc2_cam.PCO_GetSizes.argtypes = [ctypes.wintypes.HANDLE, 
                                 ctypes.wintypes.PWORD, 
                                 ctypes.wintypes.PWORD, 
                                 ctypes.wintypes.PWORD,
                                 ctypes.wintypes.PWORD]
def get_sizes(handle):
    """
    Get the current image size of the camera.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    lx : int
        Image width
    ly : int
        Image height
    lx_max : int
        Max image width
    ly_max : int
        Max image height
    """
    lx = ctypes.wintypes.WORD()
    ly = ctypes.wintypes.WORD()
    lx_max = ctypes.wintypes.WORD()
    ly_max = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetSizes(handle, lx, ly, lx_max, ly_max))
    return lx.value, ly.value, lx_max.value, ly_max.value

# ---------------------------------------------------------------------
# 2.5.6 PCO_GetROI
# ---------------------------------------------------------------------
sc2_cam.PCO_GetROI.argtypes = [ctypes.wintypes.HANDLE, 
                               ctypes.wintypes.PWORD,
                               ctypes.wintypes.PWORD,
                               ctypes.wintypes.PWORD,
                               ctypes.wintypes.PWORD]
def get_roi(handle):
    """
    Get the current region of interest as (horiz. start coordinate, vert. 
    start coordinate, horiz. end coordinate, vert. end coordinate).

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    
    Returns
    -------
    x0 : int
        Horizontal (column) start coordinate
    y0 : int
        Vertical (row) start coordinate
    x1 : int
        Horizontal end coordinate
    y1 : int
        Vertical end coordinate
    """
    x0 = ctypes.wintypes.WORD()
    x1 = ctypes.wintypes.WORD()
    y0 = ctypes.wintypes.WORD()
    y1 = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetROI(handle, x0, y0, x1, y1))
    return x0.value, x1.value, y0.value, y1.value

# ---------------------------------------------------------------------
# 2.5.7 PCO_SetROI
# ---------------------------------------------------------------------
sc2_cam.PCO_SetROI.argtypes = [ctypes.wintypes.HANDLE, 
                               ctypes.wintypes.WORD,
                               ctypes.wintypes.WORD,
                               ctypes.wintypes.WORD,
                               ctypes.wintypes.WORD]
def set_roi(handle, x0, y0, x1, y1):
    """
    Set the current region of interest as (horiz. start coordinate, vert. 
    start coordinate, horiz. end coordinate, vert. end coordinate).

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    x0 : int
        Horizontal (column) start coordinate
    y0 : int
        Vertical (row) start coordinate
    x1 : int
        Horizontal end coordinate
    y1 : int
        Vertical end coordinate
    """
    check_status(sc2_cam.PCO_SetROI(handle, ctypes.wintypes.WORD(x0), 
                                    ctypes.wintypes.WORD(y0), 
                                    ctypes.wintypes.WORD(x1), 
                                    ctypes.wintypes.WORD(y1)))

# ---------------------------------------------------------------------
# 2.5.8 PCO_GetBinning
# ---------------------------------------------------------------------
sc2_cam.PCO_GetBinning.argtypes = [ctypes.wintypes.HANDLE, 
                                   ctypes.wintypes.PWORD,
                                   ctypes.wintypes.PWORD]
def get_binning(handle):
    """
    Get horizontal and vertical binning on the camera.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    bin_horiz : int
        Horizontal (column) binning in pixels
    bin_vert : int
        Vertical (row) binning in pixels
    """
    bin_horiz = ctypes.wintypes.WORD()
    bin_vert = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetBinning(handle, bin_horiz, bin_vert))
    return bin_horiz.value, bin_vert.value

# ---------------------------------------------------------------------
# 2.5.9 PCO_SetBinning
# ---------------------------------------------------------------------
sc2_cam.PCO_SetBinning.argtypes = [ctypes.wintypes.HANDLE, 
                                   ctypes.wintypes.WORD,
                                   ctypes.wintypes.WORD]
def set_binning(handle, bin_horiz, bin_vert):
    """
    Set the horizonal and vertical binning on the camera. Possible
    values can be calculate from the binning parameters returned in 
    get_camera_description(). set_roi() must be called after set_binning()
    and before arm_camera().


    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    bin_horiz : int
        Horizontal (column) binning in pixels
    bin_vert : int
        Vertical (row) binning in pixels
    """
    check_status(sc2_cam.PCO_SetBinning(handle, ctypes.wintypes.WORD(bin_horiz), 
                                        ctypes.wintypes.WORD(bin_vert)))

# ---------------------------------------------------------------------
# 2.5.10 PCO_GetPixelRate
# ---------------------------------------------------------------------
sc2_cam.PCO_GetPixelRate.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.PDWORD]
def get_pixel_rate(handle):
    """
    Get the current pixel rate (sensor readout speed) of the camera in Hz.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    
    Returns
    -------
    pixel_rate : int
        Pixel rate of the camera in Hz
    """
    pixel_rate = ctypes.wintypes.DWORD()
    check_status(sc2_cam.GetPixelRate(handle, pixel_rate))
    return pixel_rate.value

# ---------------------------------------------------------------------
# 2.5.10 PCO_SetPixelRate
# ---------------------------------------------------------------------
sc2_cam.PCO_SetPixelRate.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD]
def set_pixel_rate(handle, pixel_rate):
    """
    Set the current pixel rate (sensor readout speed) of the camera in Hz.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    pixel_rate : int
        Pixel rate of the camera in Hz
    """
    check_status(sc2_cam.SetPixelRate(handle, ctypes.wintypes.DWORD(pixel_rate)))

# ---------------------------------------------------------------------
# 2.5.20 PCO_GetCoolingSetpointTemperature
# ---------------------------------------------------------------------
sc2_cam.PCO_GetCoolingSetpointTemperature.argtypes = [ctypes.wintypes.HANDLE, 
                                                      ctypes.POINTER(ctypes.c_short)]
def get_cooling_setpoint_temperature(handle):
    """
    Get temperature set point for image sensor. 

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    temp : int
        Setpoint in degrees C
    """
    temp = ctypes.c_short()
    check_status(sc2_cam.PCO_GetCoolingSetpointTemperature(handle,temp))
    return temp.value

# ---------------------------------------------------------------------
# 2.5.21 PCO_SetCoolingSetpointTemperature
# ---------------------------------------------------------------------
sc2_cam.PCO_SetCoolingSetpointTemperature.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_short]
def set_cooling_setpoint_temperature(handle, temp):
    """
    Set temperature set point for image sensor. 

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    temp : int
        Setpoint in degrees C. Must be in between sMinCoolSetDESC and
        sMaxCoolSetDESC (see output of get_camera_description())
    """
    check_status(sc2_cam.PCO_SetCoolingSetpointTemperature(handle, ctypes.c_short(temp)))
# ---------------------------------------------------------------------
# 2.6.4 PCO_GetDelayExposureTime
# ---------------------------------------------------------------------
sc2_cam.PCO_GetDelayExposureTime.argtypes = [ctypes.wintypes.HANDLE, 
                                             ctypes.wintypes.PDWORD,
                                             ctypes.wintypes.PDWORD,
                                             ctypes.wintypes.PWORD,
                                             ctypes.wintypes.PWORD]
PCO_TIMEBASE_NS = 0x0000  # nanoseconds
PCO_TIMEBASE_US = 0x0001  # microseconds
PCO_TIMEBASE_MS = 0x0002  # milliseconds
def get_delay_exposure_time(handle):
    """
    Get delay and exposure times for camera sensor.
    
    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    
    Returns
    -------
    delay : unsigned long int
        Delay time in units of timebase_delay
    exposure : unsigned long int
        Exposure time in units of timebase_exposure
    timebase_delay : unsigned short int
        Unit for delay time. One of PCO_TIMEBASE_NS, PCO_TIMEBASE_US, PCO_TIMEBASE_MS.
    timebase_exposure : unsigned short int
        Unit for exposure time. One of PCO_TIMEBASE_NS, PCO_TIMEBASE_US, PCO_TIMEBASE_MS.
    """
    delay = ctypes.wintypes.DWORD()
    exposure = ctypes.wintypes.DWORD()
    timebase_delay = ctypes.wintypes.WORD()
    timebase_exposure = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetDelayExposureTime(handle, delay, exposure,
                                                  timebase_delay, timebase_exposure))
    return delay.value, exposure.value, timebase_delay.value, timebase_exposure.value

# ---------------------------------------------------------------------
# 2.6.5 PCO_SetDelayExposureTime
# ---------------------------------------------------------------------
sc2_cam.PCO_SetDelayExposureTime.argtypes = [ctypes.wintypes.HANDLE, 
                                             ctypes.wintypes.DWORD,
                                             ctypes.wintypes.DWORD,
                                             ctypes.wintypes.WORD,
                                             ctypes.wintypes.WORD]
def set_delay_exposure_time(handle, delay, exposure, timebase_delay, timebase_exposure):
    """
    Set delay and exposure times for camera sensor.
    
    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    delay : int
        Delay time in units of timebase_delay. In range 
        dwMinDelayDESC .. dwMinDelayStepDESC .. dwMaxDelayDESC
        (see output of get_camera_description()).
    exposure : int
        Exposure time in units of timebase_exposure. In range
        dwMinExposDESC ..  dwMinExposStepDESC .. dwMaxExposDESC
        (see output of get_camera_description()).
    timebase_delay : int
        Unit for delay time. One of PCO_TIMEBASE_NS, PCO_TIMEBASE_US, PCO_TIMEBASE_MS.
    timebase_exposure : int
        Unit for exposure time. One of PCO_TIMEBASE_NS, PCO_TIMEBASE_US, PCO_TIMEBASE_MS.
    """
    check_status(sc2_cam.PCO_SetDelayExposureTime(handle, ctypes.wintypes.DWORD(delay), 
                                                  ctypes.wintypes.DWORD(exposure),
                                                  ctypes.wintypes.WORD(timebase_delay), 
                                                  ctypes.wintypes.WORD(timebase_exposure)))

# ---------------------------------------------------------------------
# 2.6.12 PCO_GetTriggerMode
# ---------------------------------------------------------------------
sc2_cam.PCO_GetTriggerMode.argtypes = [ctypes.wintypes.HANDLE,
                                       ctypes.wintypes.PWORD]
def get_trigger_mode(handle, mode):
    """
    Get the camera trigger mode. See pco.sdk for details.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    
    Returns
    -------
    mode : unsigned short int
        Trigger mode. One of PCO_TRIGGER_MODES.
    """
    mode = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetTriggerMode(handle, mode))

    return mode.value
# ---------------------------------------------------------------------
# 2.6.13 PCO_SetTriggerMode
# ---------------------------------------------------------------------
sc2_cam.PCO_SetTriggerMode.argtypes = [ctypes.wintypes.HANDLE,
                                       ctypes.wintypes.WORD]
def set_trigger_mode(handle, mode):
    """
    Set the camera trigger mode. See pco.sdk for details.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    mode : int
        Trigger mode. One of PCO_TRIGGER_MODES.
    """
    check_status(sc2_cam.PCO_SetTriggerMode(handle, ctypes.wintypes.WORD(mode)))

# ---------------------------------------------------------------------
# 2.6.14 PCO_ForceTrigger
# ---------------------------------------------------------------------
sc2_cam.PCO_ForceTrigger.argtypes = [ctypes.wintypes.HANDLE,
                                     ctypes.wintypes.PWORD]
TRIGGER_FAIL = 0x0000
TRIGGER_SUCCESS = 0x0001  # NOTE: This is counter to every other function
                          # where a 0 indicates success
def force_trigger(handle):
    """
    Trigger image acquisition in software triggered mode.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    triggered : unsigned short int
        Sucess of trigger call, one of TRIGGER_FAIL (0), TRIGGER_SUCCESS (1)
    """
    triggered = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_ForceTrigger(handle, triggered))
    return triggered.value

# ---------------------------------------------------------------------
# 2.6.15 PCO_GetCameraBusyStatus
# ---------------------------------------------------------------------
sc2_cam.PCO_GetCameraBusyStatus.argtypes = [ctypes.wintypes.HANDLE,
                                            ctypes.wintypes.PWORD]
PCO_CAMERA_NOT_BUSY = 0x0000
PCO_CAMERA_BUSY = 0x0001
def get_camera_busy_status(handle):
    """
    Return the busy status of the camera. Ideally checked before
    a force_trigger() call.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    state : unsigned short int
        Camera busy status, one of PCO_CAMERA_NOT_BUSY (0), PCO_CAMERA_BUSY (1)
    """
    state = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetCameraBusyStatus(handle, state))
    return state.value

# ---------------------------------------------------------------------
# 2.6.26 PCO_GetImageTiming
# ---------------------------------------------------------------------
sc2_cam.PCO_GetImageTiming.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PCO_ImageTiming)]
def get_image_timing(handle):
    """
    Get image timing with nanosecond resolution, plus additional trigger
    system information. Necessary for accurate timing information on pco.edge.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    image_timing : PCO_ImageTiming
        Struct of precise image timing and trigger information.
    """
    image_timing = PCO_ImageTiming()
    check_status(sc2_cam.PCO_GetImageTiming(handle, image_timing))
    return image_timing

# ---------------------------------------------------------------------
# 2.7.3 PCO_GetRecordingState
# ---------------------------------------------------------------------
sc2_cam.PCO_GetRecordingState.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.PWORD]
PCO_CAMERA_STOPPED = 0x0000
PCO_CAMERA_RUNNING = 0x0001
def get_recording_state(handle):
    """
    Get the current recording state of the camera.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    state : int
        Recording state, one of PCO_CAMERA_STOPPED (0), PCO_CAMERA_RUNNING (1).
    """
    state = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetRecordingState(handle, state))
    return state.value

# ---------------------------------------------------------------------
# 2.7.4 PCO_SetRecordingState
# ---------------------------------------------------------------------
sc2_cam.PCO_SetRecordingState.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.WORD]
def set_recording_state(handle, state):
    """
    Set the current recording state of the camera.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    state : int
        Recording state, one of PCO_CAMERA_STOPPED (0), PCO_CAMERA_RUNNING (1).
    """
    check_status(sc2_cam.PCO_SetRecordingState(handle, ctypes.wintypes.WORD(state)))

# ---------------------------------------------------------------------
# 2.7.5 PCO_GetStorageMode
# ---------------------------------------------------------------------
sc2_cam.PCO_GetStorageMode.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.PWORD]
PCO_STORAGE_RECORDER = 0x0000
PCO_STORAGE_FIFO = 0x0001
def get_storage_mode(handle):
    """
    Get camera image storage mode. One of "recorder" or "FIFO buffer".

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    mode : int
        Camera image storage mode. One of PCO_STORAGE_RECORDER (0) or PCO_STORAGE_FIFO (1).
    """
    mode = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetStorageMode(handle, mode))
    return mode.value

# ---------------------------------------------------------------------
# 2.7.6 PCO_SetStorageMode
# ---------------------------------------------------------------------
sc2_cam.PCO_SetStorageMode.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.WORD]
def set_storage_mode(handle, mode):
    """
    Get camera image storage mode. One of "recorder" or "FIFO buffer".

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    mode : unsigned short int
        Camera image storage mode. One of PCO_STORAGE_RECORDER (0) or PCO_STORAGE_FIFO (1).
    """
    check_status(sc2_cam.PCO_SetStorageMode(handle, ctypes.wintypes.WORD(mode)))

# ---------------------------------------------------------------------
# 2.7.9 PCO_GetAcquireMode
# ---------------------------------------------------------------------
sc2_cam.PCO_GetAcquireMode.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.PWORD]
PCO_ACQUIRE_AUTO = 0x0000
PCO_ACQUIRE_EXTERNAL = 0x0000
PCO_ACQUIRE_EXTERNAL_MODULATE = 0x0000
def get_acquire_mode(handle):
    """
    Get the current acquisition mode of the camera. One of "auto",
    "external" or "external modulate".

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    mode : unsigned short int
        Camera acquisiton mode. One of PCO_ACQUIRE_AUTO (0), PCO_ACQUIRE_EXTERNAL (1), 
        PCO_ACQUIRE_EXTERNAL_MODULATE (2).
    """
    mode = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetAcquireMode(handle, mode))
    return mode.value

# ---------------------------------------------------------------------
# 2.7.10 PCO_SetAcquireMode
# ---------------------------------------------------------------------
sc2_cam.PCO_SetAcquireMode.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.WORD]
def set_acquire_mode(handle, mode):
    """
    Set the current acquisition mode of the camera. One of "auto",
    "external" or "external modulate".

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    mode : unsigned short int
        Camera acquisiton mode. One of PCO_ACQUIRE_AUTO (0), PCO_ACQUIRE_EXTERNAL (1), 
        PCO_ACQUIRE_EXTERNAL_MODULATE (2).
    """
    check_status(sc2_cam.PCO_GetAcquireMode(handle, ctypes.wintypes.WORD(mode)))

# ---------------------------------------------------------------------
# 2.7.14 PCO_GetMetaDataMode
# ---------------------------------------------------------------------
sc2_cam.PCO_GetMetaDataMode.argtypes = [ctypes.wintypes.HANDLE, 
                                        ctypes.wintypes.PWORD,
                                        ctypes.wintypes.PWORD,
                                        ctypes.wintypes.PWORD]
PCO_METADATA_OFF = 0x0000
PCO_METADATA_ON = 0x0000
def get_metadata_mode(handle):
    """
    Get the metadata mode of the camera and information about the size
    and version of the metadata block.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    
    Returns
    -------
    mode : unsigned short int
        Metadata mode. One of PCO_METADATA_OFF (0) or PCO_METADATA_ON (1).
    size : unsigned short int
        Size of the matadata block added to the image.
    version : unsigned short int
        Version number of the metadata mode
    """
    mode = ctypes.wintypes.WORD()
    size = ctypes.wintypes.WORD()
    version = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetMetaDataMode(handle, mode, size, version))
    return mode.value, size.value, version.value

# ---------------------------------------------------------------------
# 2.7.15 PCO_SetMetaDataMode
# ---------------------------------------------------------------------
sc2_cam.PCO_SetMetaDataMode.argtypes = [ctypes.wintypes.HANDLE, 
                                        ctypes.wintypes.WORD,
                                        ctypes.wintypes.PWORD,
                                        ctypes.wintypes.PWORD]
def set_metadata_mode(handle, mode):
    """
    Set the metadata mode of the camera and get information about the size
    and version of the metadata block.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    mode : unsigned short int
        Metadata mode. One of PCO_METADATA_OFF (0) or PCO_METADATA_ON (1).
    
    Returns
    -------
    size : unsigned short int
        Size of the matadata block added to the image.
    version : unsigned short int
        Version number of the metadata mode
    """
    size = ctypes.wintypes.WORD()
    version = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_SetMetaDataMode(handle, ctypes.wintypes.WORD(mode), 
                                             size, version))
    return size.value, version.value

# ---------------------------------------------------------------------
# 2.9.5 PCO_GetBitAlignment
# ---------------------------------------------------------------------
sc2_cam.PCO_GetBitAlignment.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.PWORD]
PCO_ALIGNMENT_MSB = 0x0000  # align to most significant bit
PCO_ALIGNMENT_LSB = 0x0001  # align to least significant bit
def get_bit_alignment(handle):
    """
    Get the bit alignment of the transferred image data.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    alignment : unsigned short int
        Bit alignment of the transferred image data. One of PCO_ALIGNMENT_MSB (0)
        or PCO_ALIGNMENT_LSB (1).
    """
    alignment = ctypes.wintypes.WORD() 
    check_status(sc2_cam.PCO_GetBitAlignment(handle, alignment))
    return alignment.value

# ---------------------------------------------------------------------
# 2.9.6 PCO_SetBitAlignment
# ---------------------------------------------------------------------
sc2_cam.PCO_SetBitAlignment.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.WORD]
def set_bit_alignment(handle, alignment):
    """
    Set the bit alignment of the transferred image data.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    alignment : unsigned short int
        Bit alignment of the transferred image data. One of PCO_ALIGNMENT_MSB (0)
        or PCO_ALIGNMENT_LSB (1).
    """
    check_status(sc2_cam.PCO_SetBitAlignment(handle, ctypes.wintypes.WORD(alignment)))

# ---------------------------------------------------------------------
# 2.9.7 PCO_GetHotPixelCorrectionMode
# ---------------------------------------------------------------------
sc2_cam.PCO_GetHotPixelCorrectionMode.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.PWORD]
PCO_HOTPIXELCORRECTION_OFF = 0x0000
PCO_HOTPIXELCORRECTION_ON = 0x0001
def get_hot_pixel_correction_mode(handle):
    """
    Get camera chip hot pixel correction mode. 

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    mode : unsigned short int
        Hot pixel correction mode. One of PCO_HOTPIXELCORRECTION_OFF (0) or 
        PCO_HOTPIXELCORRECTION_ON (1).
    """
    mode = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetHotPixelCorrectionMode(handle, mode))
    return mode.value

# ---------------------------------------------------------------------
# 2.9.8 PCO_SetHotPixelCorrectionMode
# ---------------------------------------------------------------------
sc2_cam.PCO_SetHotPixelCorrectionMode.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.WORD]
def set_hot_pixel_correction_mode(handle, mode):
    """
    Set camera chip hot pixel correction mode. 

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    mode : unsigned short int
        Hot pixel correction mode. One of PCO_HOTPIXELCORRECTION_OFF (0) or 
        PCO_HOTPIXELCORRECTION_ON (1).
    """
    check_status(sc2_cam.PCO_SetHotPixelCorrectionMode(handle, ctypes.wintypes.WORD(mode)))

# ---------------------------------------------------------------------
# 2.10.1 PCO_AllocateBuffer
# ---------------------------------------------------------------------
sc2_cam.PCO_AllocateBuffer.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.c_short),
                                       ctypes.wintypes.DWORD,
                                       ctypes.POINTER(ctypes.wintypes.PWORD),
                                       ctypes.POINTER(ctypes.wintypes.HANDLE)]

def allocate_buffer(handle, index, size, buffer=None, event=None):
    """
    Allocate a buffer

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    index : int
        Buffer index to access from a previous call 
        or -1 to create a new buffer.
    size : int
        Buffer size in bytes
    buffer : double pointer
        Pointer to a memory block, default None (gets allocated internally)
    event : pointer
        Pointer to an event handle, default None (gets allocated internally)

    Returns
    -------
    index : int
        Buffer index
    buffer : double pointer
        Pointer to a memory block
    event : pointer
        Pointer to an event handle
    """
    if buffer is None:
        buffer = ctypes.c_void_p(0)
    if event is None:
        event = ctypes.c_void_p(0)
    check_status(sc2_cam.PCO_AllocateBuffer(handle, ctypes.c_short(index), ctypes.wintypes.DWORD(size), 
                                            buffer, event))
    if buffer is None and event is None:
        return index.contents, buffer, event

# ---------------------------------------------------------------------
# 2.10.2 PCO_FreeBuffer
# ---------------------------------------------------------------------
sc2_cam.PCO_FreeBuffer.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_short]
def free_buffer(handle, index):
    """
    Release a previously-allocated buffer with the given index.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    index : int
        Buffer index to free
    """
    check_status(sc2_cam.PCO_FreeBuffer(handle, ctypes.c_short(index)))

# ---------------------------------------------------------------------
# 2.10.3 PCO_GetBufferStatus
# ---------------------------------------------------------------------
sc2_cam.PCO_GetBufferStatus.argtypes = [ctypes.wintypes.HANDLE, 
                                        ctypes.c_short,
                                        ctypes.wintypes.PDWORD,
                                        ctypes.wintypes.PDWORD]
PCO_BUFFER_ALLOCATED = 0x80000000  # Buffer is allocated
PCO_BUFFER_CREATED = 0x40000000    # Buffer event created inside the SDK DLL
PCO_BUFFER_EXTERNAL = 0x20000000   # Buffer is allocated externally
PCO_BUFFER_SET = 0x00008000        # Buffer event is set
def get_buffer_status(handle, index):
    """
    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    index : int
        Buffer index to get status

    Returns
    -------
    status_dll : unsigned long int
        Status of buffer inside the DLL. One of PCO_BUFFER_ALLOCATED, PCO_BUFFER_CREATED,
        PCO_BUFFER_EXTERNAL, PCO_BUFFER_SET.
    status_drv : unsigned long int
        Status of the image transfer. One of PCO_NOERROR or see pco.sdk for error
        codes.
    """
    status_dll = ctypes.wintypes.DWORD()
    status_drv = ctypes.wintypes.DWORD()
    check_status(sc2_cam.PCO_GetBufferStatus(handle, ctypes.c_short(index), 
                                             status_dll, status_drv))
    return status_dll.value, status_drv.value

# ---------------------------------------------------------------------
# 2.10.4 PCO_GetBuffer
# ---------------------------------------------------------------------
sc2_cam.PCO_GetBuffer.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_short, 
                                  ctypes.POINTER(ctypes.wintypes.PWORD),
                                  ctypes.POINTER(ctypes.wintypes.HANDLE)]
def get_buffer(handle, index):
    """
    Get the buffer at index.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    index : int
        Buffer index to get

    Returns
    -------
    buffer : double pointer
        Double pointer to a memory region
    event: pointer
        Pointer to event handle
    """
    buffer = ctypes.c_void_p(0)
    event = ctypes.c_void_p(0)
    check_status(sc2_cam.PCO_GetBuffer(handle, ctypes.c_short(index), buffer, event))
    return buffer, event

# ---------------------------------------------------------------------
# 2.11.1 PCO_GetImageEx
# ---------------------------------------------------------------------
sc2_cam.PCO_GetImageEx.argtypes = [ctypes.wintypes.HANDLE, 
                                   ctypes.wintypes.WORD,
                                   ctypes.wintypes.DWORD,
                                   ctypes.wintypes.DWORD,
                                   ctypes.c_short,
                                   ctypes.wintypes.WORD,
                                   ctypes.wintypes.WORD,
                                   ctypes.wintypes.WORD]
def get_image_ex(handle, segment, first_image, buffer_index, lx, ly, 
                 bits_per_pixel):
    """
    Get a single image from the camera.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    segment : int
        Index of memory segment. 1 is the default memory segment
    first_image : int
        Image number. 1 if PCO_GetRecordingState is PCO_CAMERA_STOPPED, 0 if it is PCO_CAMERA_RUNNING
    buffer_index : int
        Buffer index.
    lx : int
        Image width
    ly : int
        Image height
    bits_per_pixel : int
        Bit resolution of the transferred image (16 is the common choice, see pco.sdk manual)
    """
    image_index = ctypes.wintypes.DWORD(first_image)
    check_status(sc2_cam.PCO_GetImageEx(handle, ctypes.wintypes.WORD(segment), image_index,
                                        image_index, ctypes.c_short(buffer_index), 
                                        ctypes.wintypes.WORD(lx), ctypes.wintypes.WORD(ly), 
                                        ctypes.wintypes.WORD(bits_per_pixel)))
# ---------------------------------------------------------------------
# 2.11.3 PCO_AddBufferEx
# ---------------------------------------------------------------------
sc2_cam.PCO_AddBufferEx.argtypes = [ctypes.wintypes.HANDLE, 
                                    ctypes.wintypes.DWORD,
                                    ctypes.wintypes.DWORD,
                                    ctypes.c_short,
                                    ctypes.wintypes.WORD,
                                    ctypes.wintypes.WORD,
                                    ctypes.wintypes.WORD]
def add_buffer_ex(handle, first_image, buffer_index, lx, ly, 
                  bits_per_pixel):
    """
    Set up buffer for single image transfer from the camera. Can add multiple
    buffers for fast live recording.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    first_image : int
        Image number. 1 if PCO_GetRecordingState is PCO_CAMERA_STOPPED, 0 if it is PCO_CAMERA_RUNNING
    buffer_index : int
        Buffer index.
    lx : int
        Image width
    ly : int
        Image height
    bits_per_pixel : int
        Bit resolution of the transferred image (16 is the common choice, see pco.sdk manual)
    """
    image_index = ctypes.wintypes.DWORD(first_image)
    check_status(sc2_cam.PCO_AddBufferEx(handle, image_index, image_index,
                                         ctypes.c_short(buffer_index), 
                                         ctypes.wintypes.WORD(lx), ctypes.wintypes.WORD(ly), 
                                         ctypes.wintypes.WORD(bits_per_pixel)))

# ---------------------------------------------------------------------
# 2.11.5 PCO_AddBufferExtern
# ---------------------------------------------------------------------
sc2_cam.PCO_AddBufferExtern.argtypes = [ctypes.wintypes.HANDLE,
                                        ctypes.wintypes.HANDLE,
                                        ctypes.wintypes.WORD,
                                        ctypes.wintypes.DWORD,
                                        ctypes.wintypes.DWORD,
                                        ctypes.wintypes.DWORD,
                                        ctypes.c_void_p,         # alternatively numpy.ctypeslib.ndpointer
                                        ctypes.wintypes.DWORD,
                                        ctypes.wintypes.PDWORD]
def add_buffer_extern(handle, event, segment, first_image, buf, n_bytes, status):
    """
    Set up buffer for single image transfer from the camera. Can add multiple
    buffers for fast live recording. This function lets us use buffers we
    create externally from the pco.sdk DLL.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    event : ctypes.wintypes.HANDLE
        Event handle
    segment : int
        Camera internal memory segment. Most of the time this will have a value 
        of 1. In a default state, all memory is distributed to segement 1 and 
        segment 1 is  the active segment. Camera internal memory is arranged as 
        an array with 4 segments. 
    first_image : int
        Image number. 1 if PCO_GetRecordingState is PCO_CAMERA_STOPPED, 0 if it 
        is PCO_CAMERA_RUNNING
    buf : ctypes.c_void_p
        Pointer to a 2D image buffer, probably a numpy array. See 
        https://numpy.org/doc/stable/reference/routines.ctypeslib.html
    n_bytes : int
        Number of bytes in the 2D image buffer, buf
    status : ctypes.wintypes.PDWORD
        Pointer to buffer status bit
    """
    image_index = ctypes.wintypes.DWORD(first_image)
    synch = ctypes.wintypes.DWORD(0)
    check_status(sc2_cam.PCO_AddBufferExtern(handle, event, ctypes.wintypes.WORD(segment), 
                                             image_index, image_index, synch, buf, 
                                             ctypes.wintypes.DWORD(n_bytes), 
                                             status))

# ---------------------------------------------------------------------
# 2.11.6 PCO_CancelImages
# ---------------------------------------------------------------------
sc2_cam.PCO_CancelImages.argtypes = [ctypes.wintypes.HANDLE]
def cancel_images(handle):
    """

    Remove all remaining buffers from the internal queue, reset the internal queue
    and also reset the transfer state machine in the camera.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    """
    check_status(sc2_cam.PCO_CancelImages(handle))

# ---------------------------------------------------------------------
# 2.11.9 PCO_WaitforBuffer
# ---------------------------------------------------------------------
sc2_cam.PCO_WaitforBuffer.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_int, 
                                      ctypes.POINTER(PCO_Buflist),
                                      ctypes.c_int]
def wait_for_buffer(handle, num_buffers, buffer_list, timeout):
    """
    Wait for one or buffers, which are in the requeest queue of the driver.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    num_buffers : int
        How many buffers in buffer_list?
    buffer_list : list
        List of PCO_Buflist containing all allocated buffers
        we are waiting on.
    timeout : int
        Timeout in milliseconds.
    """
    check_status(sc2_cam.PCO_WaitForBuffer(handle, ctypes.c_int(num_buffers), 
                                           buffer_list, ctypes.c_int(timeout)))

# ---------------------------------------------------------------------
# 2.11.9 PCO_GetMetaData
# ---------------------------------------------------------------------
sc2_cam.PCO_GetMetaData.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_short, PCO_Metadata_Struct,
                                    ctypes.wintypes.DWORD, ctypes.wintypes.DWORD]
def get_metadata(handle, index):
    """
    Get metadata associated with image in the buffer index.

    Parameters
    ----------
    handle : ctypes.wintypes.HANDLE
        Unique pco. camera handle (pointer).
    index : int
        Buffer index to get metadata

    Returns
    -------
    metadata : PCO_Metadata_Struct
        Metadata from the buffer at index
    """
    reserved1 = ctypes.wintypes.DWORD()
    reserved2 = ctypes.wintypes.DWORD()
    metadata = PCO_Metadata_Struct()
    check_status(sc2_cam.PCO_GetMetaData(handle, ctypes.c_short(index), metadata, 
                                         reserved1, reserved2))
    return metadata

# ---------------------------------------------------------------------
# 5.1 PCO_GetErrorText
# ---------------------------------------------------------------------
sc2_cam.PCO_GetErrorText.argtypes = [ctypes.wintypes.DWORD,
                                     ctypes.c_char_p,
                                     ctypes.wintypes.DWORD]
def get_error_text(err):
    """
    Get detailed description for a pco.sdk error.

    Parameters
    ----------
    err : int
        Error number

    Returns
    -------
    string
        Error description.
    """
    c_buf_len = 512
    c_buf = ctypes.create_string_buffer(c_buf_len)
    sc2_cam.PCO_GetErrorText(ctypes.wintypes.DWORD(err), c_buf, c_buf_len)

    return str(c_buf.value.decode('ascii'))
