# -*- coding: utf-8 -*-

"""
Created on Fri May 14 2021

@author: zacsimile
"""

# NOTE: I cheated a little off https://pypi.org/project/pco/

# For more information, see the pco.sdk manual,
# https://www.pco.de/fileadmin/fileadmin/user_upload/pco-manuals/pco.sdk_manual.pdf

import ctypes
import ctypes.wintypes
import platform
import os
import sys

os_desc = platform.system()
if os_desc != 'Windows':
    raise Exception("Operating system is not supported.")

dll = 'SC2_Cam.dll'
dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dll)
try:
    sc2_cam = ctypes.windll.LoadLibrary(dll_path)
except:
    raise OSError(f"{dll} not found in {dll_path}.")

# ---------------------------------------------------------------------
# Structures and types
# ---------------------------------------------------------------------
HANDLE = ctypes.c_void_p
HANDLE_P = ctypes.POINTER(HANDLE)

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
                ("wMinSizeVertDESC", ctypes.wintypes.WORD),
                ("dwPixelRateDESC", ctypes.wintypes.DWORD * 4),
                ("ZzdwDummy", ctypes.wintypes.DWORD),
                ("wConvFactDESC", ctypes.wintypes.WORD * 4),
                ("sCoolingSetpoints", ctypes.c_short * 10),
                ("ZZdwDummycv", ctypes.wintypes.WORD),
                ("wSoftRoiHorStepsDESC", ctypes.wintypes.WORD),
                ("wSoftRoiVertStepsDESC", ctypes.wintypes.WORD),
                ("wIRDESC", ctypes.wintypes.WORD),
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

class PCO_HW_Vers(C.Structure):
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

# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------
class PcoSDKException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

def check_status(err, function_name=None):
    if err:
        if function_name is None:
            function_name = sys._getframe(1).f_code.co_name  # who called me?
        err_desc = get_error_text(err)
        raise PcoSDKException(f"Error during {function_name}: {err_desc}")

# ---------------------------------------------------------------------
# 2.1.1 PCO_OpenCamera
# ---------------------------------------------------------------------
sc2_cam.PCO_OpenCamera.argtypes = [HANDLE_P, 
                                   ctypes.wintypes.WORD]
def open_camera():
    """
    Open a pco. camera, return the handle to that camera.
    This uses a scan process, so call this multiple times 
    to open multiple cameras. Call get_camera_type() to
    figure out which camera was grabbed.

    Returns
    -------
    handle : HANDLE
        Unique pco. camera handle (pointer).
    """
    handle = ctypes.c_void_p(0)
    cam_num = ctypes.wintypes.WORD()  # unused
    check_status(sc2_cam.PCO_OpenCamera(handle, cam_num))
    return handle

# ---------------------------------------------------------------------
# 2.1.3 PCO_CloseCamera
# ---------------------------------------------------------------------
sc2_cam.PCO_CloseCamera.argtypes = [HANDLE]
def close_camera(handle):
    """
    Close a pco. camera.

    Parameters
    ----------
    handle : HANDLE
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
sc2_cam.PCO_GetCameraDescription.argtypes = [HANDLE,
                                             ctypes.POINTER(PCO_Description)]
def get_camera_description(handle):
    """
    Get camera description. Includes properties such as image size, pixel rate, etc.,
    but not camera names, serial numbers, etc. For the latter, call get_camera_type().

    Parameters
    ----------
    handle : HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    PCO_Description
        Struct of camera properties.
    """
    desc = PCO_Description()
    check_status(sc2_cam.PCO_GetCameraDescription(handle, desc))
    return desc

# ---------------------------------------------------------------------
# 2.3.3 PCO_GetCameraType
# ---------------------------------------------------------------------
sc2_cam.PCO_GetCameraType.argtypes = [HANDLE, ctypes.POINTER(PCO_CameraType)]
def get_camera_type(handle):
    """
    Get camera name, serial number, etc.

    Parameters
    ----------
    handle : HANDLE
        Unique pco. camera handle (pointer).
    
    Returns
    -------
    PCO_CameraType
        Struct containing hardware/firmware version, serial number, etc.
    """
    camera_type = PCO_CameraType()
    check_status(sc2_cam.PCO_GetCameraType(handle, camera_type))
    return camera_type

# ---------------------------------------------------------------------
# 2.3.3 PCO_GetCameraHealthStatus
# ---------------------------------------------------------------------
sc2_cam.PCO_GetCameraHealthStatus.argtypes = [HANDLE,
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
    handle : HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    warn : unsigned long int
        Warning bit 
    err : unsigned long int
        Error bit 
    status : unsigned long int
        Status bit 
    """
    warn = ctypes.wintypes.DWORD()
    err = ctypes.wintypes.DWORD()
    status = ctypes.wintypes.DWORD()
    check_status(sc2_cam.PCO_GetCameraHealthStatus(handle, warn, err, status))
    return warn, err, status

# ---------------------------------------------------------------------
# 2.3.4 PCO_GetTemperature
# ---------------------------------------------------------------------
sc2_cam.PCO_GetTemperature.argtypes = [HANDLE, ctypes.POINTER(ctypes.c_short),
                                       ctypes.POINTER(ctypes.c_short), 
                                       ctypes.POINTER(ctypes.c_short)]
def get_temperature(handle):
    """
    Get camera temperatures.

    Parameters
    ----------
    handle : HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    ccd_temp : short
        Image sensor temp in 1/10 degree i.e. 100 = 10.0 C
    cam_temp : short
        Internal camera temp in C
    pow_temp : short
        Temp of additional devices (e.g. power supply)
    """
    ccd_temp = ctypes.c_short()
    cam_temp = ctypes.c_short()
    pow_temp = ctypes.c_short()
    check_status(sc2_cam.PCO_GetTemperature(handle, ccd_temp, cam_temp, pow_temp))
    return ccd_temp, cam_temp, pow_temp


# ---------------------------------------------------------------------
# 2.4.1 PCO_ArmCamera
# ---------------------------------------------------------------------
sc2_cam.PCO_ArmCamera.argtypes = [HANDLE]
def arm_camera(handle):
    """
    Prepare the camera for recording. Must be called after the change of 
    any camera parameter except for delay and exposure time.

    Parameters
    ----------
    handle : HANDLE
        Unique pco. camera handle (pointer).
    """
    check_status(sc2_cam.PCO_ArmCamera(handle))

# ---------------------------------------------------------------------
# 2.4.3 PCO_SetImageParameters
# ---------------------------------------------------------------------
sc2_cam.PCO_SetImageParameters.argtypes = [HANDLE, ctypes.wintypes.WORD,
                                           ctypes.wintypes.WORD,
                                           ctypes.wintypes.DWORD,
                                           ctypes.c_void_p,
                                           ctypes.c_int]
IMAGEPARAMETERS_READ_WHILE_RECORDING = ctypes.wintypes.DWORD(int("0x00000001", 0))
IMAGEPARAMETERS_READ_FROM_SEGMENTS = ctypes.wintypes.DWORD(int("0x00000002", 0))
def set_image_parameters(handle, lx, ly, image_flag):
    """
    Set image parameters for internal allocated resources before image transfer.

    Parameters
    ----------
    handle : HANDLE
        Unique pco. camera handle (pointer).
    lx : int
        Image width
    ly : int
        Image height
    image_flag : int
        IMAGEPARAMETERS_READ_WHILE_RECORDING or IMAGEPARAMETERS_READ_FROM_SEGMENTS
    """
    param = ctypes.c_void_p(0)
    ilen = ctypes.c_int(0)
    check_status(sc2_cam.PCO_SetImageParameters(handle, lx, ly, image_flag, param, ilen))

# ---------------------------------------------------------------------
# 2.4.4 PCO_ResetSettingsToDefault
# ---------------------------------------------------------------------
sc2_cam.PCO_ResetSettingsToDefault.argtypes = [HANDLE]
def reset_settings_to_default(handle):
    """
    Reset the camera settings to default. Executed by default during
    camera power-up.

    Parameters
    ----------
    handle : HANDLE
        Unique pco. camera handle (pointer).
    """
    check_status(sc2_cam.PCO_ResetSettingsToDefault(handle))
# ---------------------------------------------------------------------
# 2.5.3 PCO_GetSizes
# ---------------------------------------------------------------------
sc2_cam.PCO_GetSizes.argtypes = [HANDLE, ctypes.wintypes.PWORD, 
                                 ctypes.wintypes.PWORD, 
                                 ctypes.wintypes.PWORD,
                                 ctypes.wintypes.PWORD]
def get_sizes(handle):
    """
    Get the current image size of the camera.

    Parameters
    ----------
    handle : HANDLE
        Unique pco. camera handle (pointer).

    Returns
    -------
    lx : unsigned short int
        Image width
    ly : unsigned short int
        Image height
    lx_max : unsigned short int
        Max image width
    ly_max : unsigned short int
        Max image height
    """
    lx = ctypes.wintypes.WORD()
    ly = ctypes.wintypes.WORD()
    lx_max = ctypes.wintypes.WORD()
    ly_max = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetSizes(handle, lx, ly, lx_max, ly_max))
    return lx, ly, lx_max, ly_max

# ---------------------------------------------------------------------
# 2.5.6 PCO_GetROI
# ---------------------------------------------------------------------
sc2_cam.PCO_GetROI.argtypes = [HANDLE, ctypes.wintypes.PWORD,
                               ctypes.wintypes.PWORD,
                               ctypes.wintypes.PWORD,
                               ctypes.wintypes.PWORD]
def get_roi(handle):
    """
    Get the current region of interest as (horiz. start coordinate, vert. 
    start coordinate, horiz. end coordinate, vert. end coordinate).

    Parameters
    ----------
    handle : HANDLE
        Unique pco. camera handle (pointer).
    
    Returns
    -------
    x0 : unsigned short int
        Horizontal (column) start coordinate
    y0 : unsigned short int
        Vertical (row) start coordinate
    x1 : unsigned short int
        Horizontal end coordinate
    y1 : unsigned short int
        Vertical end coordinate
    """
    x0 = ctypes.wintypes.WORD()
    x1 = ctypes.wintypes.WORD()
    y0 = ctypes.wintypes.WORD()
    y1 = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetROI(handle, x0, y0, x1, y1))
    return x0, x1, y0, y1

# ---------------------------------------------------------------------
# 2.5.7 PCO_SetROI
# ---------------------------------------------------------------------
sc2_cam.PCO_SetROI.argtypes = [HANDLE, ctypes.wintypes.WORD,
                               ctypes.wintypes.WORD,
                               ctypes.wintypes.WORD,
                               ctypes.wintypes.WORD]
def set_roi(handle, x0, y0, x1, y1):
    """
    Set the current region of interest as (horiz. start coordinate, vert. 
    start coordinate, horiz. end coordinate, vert. end coordinate).

    Parameters
    ----------
    handle : HANDLE
        Unique pco. camera handle (pointer).
    x0 : unsigned short int
        Horizontal (column) start coordinate
    y0 : unsigned short int
        Vertical (row) start coordinate
    x1 : unsigned short int
        Horizontal end coordinate
    y1 : unsigned short int
        Vertical end coordinate
    """
    check_status(sc2_cam.PCO_SetROI(handle, x0, y0, x1, y1))

# ---------------------------------------------------------------------
# 2.5.8 PCO_GetBinning
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.5.9 PCO_SetBinning
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.5.10 PCO_GetPixelRate
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.5.10 PCO_SetPixelRate
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.5.20 PCO_GetCoolingSetpointTemperature
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.5.21 PCO_SetCoolingSetpointTemperature
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.6.4 PCO_GetDelayExposureTime
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.6.5 PCO_SetDelayExposureTime
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.3 PCO_GetRecordingState
# ---------------------------------------------------------------------
sc2_cam.PCO_GetRecordingState.argtypes = [HANDLE, ctypes.wintypes.PWORD]
def get_recording_state(handle=ctypes.c_void_p(0)):
    state = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetRecordingState(handle, state))
    return state

# ---------------------------------------------------------------------
# 2.7.4 PCO_SetRecordingState
# ---------------------------------------------------------------------
sc2_cam.PCO_SetRecordingState.argtypes = [HANDLE, ctypes.wintypes.WORD]
def set_recording_state(handle=ctypes.c_void_p(0), state=0):
    check_status(sc2_cam.PCO_SetRecordingState(handle, state))

# ---------------------------------------------------------------------
# 2.7.5 PCO_GetStorageMode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.6 PCO_SetStorageMode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.7 PCO_GetRecorderSubmode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.8 PCO_SetRecorderSubmode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.9 PCO_GetAcquireMode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.10 PCO_SetAcquireMode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.11 PCO_GetAcquireModeEx
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.12 PCO_SetAcquireModeEx
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.14 PCO_GetMetaDataMode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.15 PCO_SetMetaDataMode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.20 PCO_GetTimestampMode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.7.21 PCO_SetTimestampMode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.9.5 PCO_GetBitAlignment
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.9.6 PCO_SetBitAlignment
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.9.7 PCO_GetHotPixelCorrectionMode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.9.8 PCO_SetHotPixelCorrectionMode
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.10.1 PCO_AllocateBuffer
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.10.2 PCO_FreeBuffer
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.10.3 PCO_GetBufferStatus
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.10.4 PCO_GetBuffer
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.11.1 PCO_GetImageEx
# ---------------------------------------------------------------------
sc2_cam.PCO_GetImageEx.argtypes = [HANDLE, 
                                   ctypes.wintypes.WORD,
                                   ctypes.wintypes.DWORD,
                                   ctypes.wintypes.DWORD,
                                   ctypes.c_short,
                                   ctypes.wintypes.WORD,
                                   ctypes.wintypes.WORD,
                                   ctypes.wintypes.WORD]
def get_image_ex(handle, segment, first_image, buffer_index, lx, ly, 
                 bits_per_pixel):
    check_status(sc2_cam.PCO_GetImageEx(handle, segment, first_image,
                 first_image, buffer_index, lx, ly, bits_per_pixel))
# ---------------------------------------------------------------------
# 2.11.3 PCO_AddBufferEx
# ---------------------------------------------------------------------
sc2_cam.PCO_AddBufferEx.argtypes = [HANDLE, 
                                    ctypes.wintypes.DWORD,
                                    ctypes.wintypes.DWORD,
                                    ctypes.c_short,
                                    ctypes.wintypes.WORD,
                                    ctypes.wintypes.WORD,
                                    ctypes.wintypes.WORD]
def add_buffer_ex(handle, first_image, last_image, buffer_index, lx, ly, 
                  bits_per_pixel):
    check_status(sc2_cam.PCO_AddBufferEx(handle, first_image, last_image,
                 buffer_index, lx, ly, bits_per_pixel))
# ---------------------------------------------------------------------
# 2.11.6 PCO_CancelImages
# ---------------------------------------------------------------------
sc2_cam.PCO_CancelImages.argtypes = [HANDLE]
def cancel_images(handle=ctypes.c_void_p(0)):
    check_status(sc2_cam.PCO_CancelImages(handle))

# ---------------------------------------------------------------------
# 2.11.9 PCO_WaitforBuffer
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.11.9 PCO_GetMetaData
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 5.1 PCO_GetErrorText
# ---------------------------------------------------------------------
sc2_cam.PCO_GetErrorText.argtypes = [ctypes.wintypes.DWORD,
                                     ctypes.c_char_p,
                                     ctypes.wintypes.DWORD]
def get_error_text(err):
    c_buf_len = 512
    c_buf = ctypes.create_string_buffer(c_buf_len)
    sc2_cam.PCO_GetErrorText(err, c_buf, c_buf_len)

    return c_buf.value.decode('ascii')
