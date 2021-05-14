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

# ---------------------------------------------------------------------
# Function prototypes
# ---------------------------------------------------------------------
sc2_cam.PCO_OpenCamera.argtypes = [HANDLE_P, 
                                   ctypes.wintypes.WORD]
sc2_cam.PCO_GetCameraDescription.argtypes = [HANDLE,
                                             ctypes.POINTER(PCO_Description)]
sc2_cam.PCO_GetCameraHealthStatus.argtypes = [HANDLE,
                                              ctypes.wintypes.PDWORD,
                                              ctypes.wintypes.PDWORD,
                                              ctypes.wintypes.PDWORD]
sc2_cam.PCO_ArmCamera.argtypes = [HANDLE]
sc2_cam.PCO_GetRecordingState.argtypes = [HANDLE, ctypes.wintypes.PWORD]
sc2_cam.PCO_SetRecordingState.argtypes = [HANDLE, ctypes.wintypes.WORD]
sc2_cam.PCO_GetImageEx.argtypes = [HANDLE, 
                                   ctypes.wintypes.WORD,
                                   ctypes.wintypes.DWORD,
                                   ctypes.wintypes.DWORD,
                                   ctypes.c_short,
                                   ctypes.wintypes.WORD,
                                   ctypes.wintypes.WORD,
                                   ctypes.wintypes.WORD]
sc2_cam.PCO_AddBufferEx.argtypes = [HANDLE, 
                                    ctypes.wintypes.DWORD,
                                    ctypes.wintypes.DWORD,
                                    ctypes.c_short,
                                    ctypes.wintypes.WORD,
                                    ctypes.wintypes.WORD,
                                    ctypes.wintypes.WORD]
sc2_cam.PCO_CancelImages.argtypes = [HANDLE]
sc2_cam.PCO_GetErrorText.argtypes = [ctypes.wintypes.DWORD,
                                     ctypes.c_char_p,
                                     ctypes.wintypes.DWORD]

# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------
class PcoSDKException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

def check_status(err, function_name="unknown function"):
    if err:
        err_desc = get_error_text(err)
        raise PcoSDKException(f"Error during {function_name}: {err_desc}")

# ---------------------------------------------------------------------
# 2.1.1 PCO_OpenCamera
# ---------------------------------------------------------------------
def open_camera(handle=ctypes.c_void_p(0), cam_num=0):
    check_status(sc2_cam.PCO_OpenCamera(handle, cam_num), 
                 sys._getframe().f_code.co_name)

# ---------------------------------------------------------------------
# 2.2.1 PCO_GetCameraDescription
# ---------------------------------------------------------------------
def get_camera_description(handle=ctypes.c_void_p(0)):
    desc = PCO_Description()
    check_status(sc2_cam.PCO_GetCameraDescription(handle, desc),
                 sys._getframe().f_code.co_name)
    return desc

# ---------------------------------------------------------------------
# 2.3.3 PCO_GetCameraHealthStatus
# ---------------------------------------------------------------------
def get_camera_health_status(handle=ctypes.c_void_p(0)):
    warn = ctypes.wintypes.DWORD()
    err = ctypes.wintypes.DWORD()
    status = ctypes.wintypes.DWORD()
    check_status(sc2_cam.PCO_GetCameraHealthStatus(handle, warn, err, status),
                 sys._getframe().f_code.co_name)
    return warn, err, status

# ---------------------------------------------------------------------
# 2.4.1 PCO_ArmCamera
# ---------------------------------------------------------------------
def arm_camera(handle=ctypes.c_void_p(0)):
    check_status(sc2_cam.PCO_ArmCamera(handle), sys._getframe().f_code.co_name)

# ---------------------------------------------------------------------
# 2.7.3 PCO_GetRecordingState
# ---------------------------------------------------------------------
def get_recording_state(handle=ctypes.c_void_p(0)):
    state = ctypes.wintypes.WORD()
    check_status(sc2_cam.PCO_GetRecordingState(handle, state),
                 sys._getframe().f_code.co_name)
    return state

# ---------------------------------------------------------------------
# 2.7.4 PCO_SetRecordingState
# ---------------------------------------------------------------------
def set_recording_state(handle=ctypes.c_void_p(0), state=0):
    check_status(sc2_cam.PCO_SetRecordingState(handle, state), 
                 sys._getframe().f_code.co_name)

# ---------------------------------------------------------------------
# 2.11.1 PCO_GetImageEx
# ---------------------------------------------------------------------
def get_image_ex(handle=ctypes.c_void_p(0), segment=1, first_image=0, 
                 buffer_index=0, lx=256, ly=256, bits_per_pixel=16):
    check_status(sc2_cam.PCO_GetImageEx(handle, segment, first_image,
                 first_image, buffer_index, lx, ly, bits_per_pixel))
# ---------------------------------------------------------------------
# 2.11.3 PCO_AddBufferEx
# ---------------------------------------------------------------------
def add_buffer_ex(handle=ctypes.c_void_p(0), first_image=0, last_image=0,
                  buffer_index=0, lx=256, ly=256, bits_per_pixel=16):
    check_status(sc2_cam.PCO_AddBufferEx(handle, first_image, last_image,
                 buffer_index, lx, ly, bits_per_pixel))
# ---------------------------------------------------------------------
# 2.11.6 PCO_CancelImages
# ---------------------------------------------------------------------
def cancel_images(handle=ctypes.c_void_p(0)):
    check_status(sc2_cam.PCO_CancelImages(handle), sys._getframe().f_code.co_name)

# ---------------------------------------------------------------------
# 5.1 PCO_GetErrorText
# ---------------------------------------------------------------------
def get_error_text(err):
    c_buf_len = 512
    c_buf = ctypes.create_string_buffer(c_buf_len)
    sc2_cam.PCO_GetErrorText(err, c_buf, c_buf_len)

    return c_buf.value.decode('ascii')
    