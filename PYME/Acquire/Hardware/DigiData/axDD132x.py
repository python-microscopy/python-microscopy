#!/usr/bin/python

##################
# axDD132x.py
#
# Copyright David Baddeley, 2009
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
##################

from ctypes import *

from ctypes.wintypes import UINT
from ctypes.wintypes import BYTE
from ctypes.wintypes import ULONG
STRING = c_char_p
from ctypes.wintypes import DWORD
from ctypes.wintypes import BOOL
from ctypes.wintypes import WORD
from ctypes.wintypes import HANDLE
from ctypes.wintypes import HWND
from ctypes.wintypes import LPSTR
from ctypes.wintypes import LPCSTR
_stdcall_libraries = {}
_stdcall_libraries['axdd132x.dll'] = WinDLL('axdd132x.dll')


DD132X_Bit0Tag = 3
DD132X_NoOutputPulse = 0
DD132X_ADC_level_Triggered = 1
DD132X_Bit0Tag_Bit1Line = 5
DD132X_DAC_bit0_Triggered = 2
DD132X_Bit0Data = 0
DD132X_StartImmediately = 0
DD132X_ExternalStart = 1
DD132X_Bit0ExtStart = 1
DD132X_Bit0Tag_Bit1ExtStart = 4
DD132X_Bit0Line = 2
DD132X_LineTrigger = 2
ADC_VALUE = c_short
DAC_VALUE = c_short
class DATABUFFER(Structure):
    pass
DATABUFFER._pack_ = 1
DATABUFFER._fields_ = [
    ('uNumSamples', UINT),
    ('uFlags', UINT),
    ('pnData', POINTER(ADC_VALUE)),
    ('psDataFlags', POINTER(BYTE)),
    ('pNextBuffer', POINTER(DATABUFFER)),
    ('pPrevBuffer', POINTER(DATABUFFER)),
]
PDATABUFFER = POINTER(DATABUFFER)
class FLOATBUFFER(Structure):
    pass
FLOATBUFFER._pack_ = 1
FLOATBUFFER._fields_ = [
    ('uNumSamples', UINT),
    ('uFlags', UINT),
    ('pfData', POINTER(c_float)),
    ('pNextBuffer', POINTER(FLOATBUFFER)),
    ('pPrevBuffer', POINTER(FLOATBUFFER)),
]
PFLOATBUFFER = POINTER(FLOATBUFFER)
PULONG = POINTER(ULONG)
USHORT = c_ushort
PUSHORT = POINTER(USHORT)
UCHAR = c_ubyte
PUCHAR = POINTER(UCHAR)
PSZ = STRING
LONGLONG = c_long
FLOAT = c_float
PFLOAT = POINTER(FLOAT)
PBOOL = POINTER(BOOL)
LPBOOL = POINTER(BOOL)
PBYTE = POINTER(BYTE)
LPBYTE = POINTER(BYTE)
PINT = POINTER(c_int)
LPINT = POINTER(c_int)
PWORD = POINTER(WORD)
LPWORD = POINTER(WORD)
LPLONG = POINTER(c_long)
PDWORD = POINTER(DWORD)
LPDWORD = POINTER(DWORD)
LPVOID = c_void_p
LPCVOID = c_void_p
INT = c_int
PUINT = POINTER(c_uint)
class _SYSTEMTIME(Structure):
    pass
_SYSTEMTIME._pack_ = 1
_SYSTEMTIME._fields_ = [
    ('wYear', WORD),
    ('wMonth', WORD),
    ('wDayOfWeek', WORD),
    ('wDay', WORD),
    ('wHour', WORD),
    ('wMinute', WORD),
    ('wSecond', WORD),
    ('wMilliseconds', WORD),
]
PSYSTEMTIME = POINTER(_SYSTEMTIME)
SYSTEMTIME = _SYSTEMTIME
LPSYSTEMTIME = POINTER(_SYSTEMTIME)
class DD132X_Info(Structure):
    pass
DD132X_Info._pack_ = 1
DD132X_Info._fields_ = [
    ('uLength', UINT),
    ('byAdaptor', BYTE),
    ('byTarget', BYTE),
    ('byImageType', BYTE),
    ('byResetType', BYTE),
    ('szManufacturer', c_char * 16),
    ('szName', c_char * 32),
    ('szProductVersion', c_char * 8),
    ('szFirmwareVersion', c_char * 16),
    ('uInputBufferSize', UINT),
    ('uOutputBufferSize', UINT),
    ('uSerialNumber', UINT),
    ('uClockResolution', UINT),
    ('uMinClockTicks', UINT),
    ('uMaxClockTicks', UINT),
    ('byUnused', BYTE * 280),
]

# values for enumeration 'DD132X_Triggering'
DD132X_Triggering = c_int # enum

# values for enumeration 'DD132X_AIDataBits'
DD132X_AIDataBits = c_int # enum

# values for enumeration 'DD132X_OutputPulseType'
DD132X_OutputPulseType = c_int # enum
class DD132X_Protocol(Structure):
    pass
DD132X_Protocol._pack_ = 1
DD132X_Protocol._fields_ = [
    ('uLength', UINT),
    ('dSampleInterval', c_double),
    ('dwFlags', DWORD),
    ('eTriggering', DD132X_Triggering),
    ('eAIDataBits', DD132X_AIDataBits),
    ('uAIChannels', UINT),
    ('anAIChannels', c_int * 64),
    ('pAIBuffers', POINTER(DATABUFFER)),
    ('uAIBuffers', UINT),
    ('uAOChannels', UINT),
    ('anAOChannels', c_int * 64),
    ('pAOBuffers', POINTER(DATABUFFER)),
    ('uAOBuffers', UINT),
    ('uTerminalCount', LONGLONG),
    ('eOutputPulseType', DD132X_OutputPulseType),
    ('bOutputPulsePolarity', c_short),
    ('nOutputPulseChannel', c_short),
    ('wOutputPulseThreshold', WORD),
    ('wOutputPulseHystDelta', WORD),
    ('uChunksPerSecond', UINT),
    ('byUnused', BYTE * 248),
]
class DD132X_PowerOnData(Structure):
    pass
DD132X_PowerOnData._pack_ = 1
DD132X_PowerOnData._fields_ = [
    ('uLength', UINT),
    ('dwDigitalOuts', DWORD),
    ('anAnalogOuts', c_short * 16),
]
class DD132X_CalibrationData(Structure):
    pass
DD132X_CalibrationData._pack_ = 1
DD132X_CalibrationData._fields_ = [
    ('uLength', UINT),
    ('uEquipmentStatus', UINT),
    ('dADCGainRatio', c_double),
    ('nADCOffset', c_short),
    ('byUnused1', BYTE * 46),
    ('wNumberOfDACs', WORD),
    ('byUnused2', BYTE * 6),
    ('anDACOffset', c_short * 16),
    ('adDACGainRatio', c_double * 16),
    ('byUnused4', BYTE * 24),
]
class DD132X_StartAcqInfo(Structure):
    pass
DD132X_StartAcqInfo._pack_ = 1
DD132X_StartAcqInfo._fields_ = [
    ('uLength', UINT),
    ('m_StartTime', SYSTEMTIME),
    ('m_n64PreStartAcq', c_longlong),
    ('m_n64PostStartAcq', c_longlong),
]
HDD132X = HANDLE
DD132X_RescanSCSIBus = _stdcall_libraries['axdd132x.dll'].DD132X_RescanSCSIBus
DD132X_RescanSCSIBus.restype = BOOL
DD132X_RescanSCSIBus.argtypes = [POINTER(c_int)]
DD132X_FindDevices = _stdcall_libraries['axdd132x.dll'].DD132X_FindDevices
DD132X_FindDevices.restype = UINT
DD132X_FindDevices.argtypes = [POINTER(DD132X_Info), UINT, POINTER(c_int)]
DD132X_OpenDevice = _stdcall_libraries['axdd132x.dll'].DD132X_OpenDevice
DD132X_OpenDevice.restype = HDD132X
DD132X_OpenDevice.argtypes = [BYTE, BYTE, POINTER(c_int)]
DD132X_OpenDeviceEx = _stdcall_libraries['axdd132x.dll'].DD132X_OpenDeviceEx
DD132X_OpenDeviceEx.restype = HDD132X
DD132X_OpenDeviceEx.argtypes = [BYTE, BYTE, POINTER(BYTE), UINT, POINTER(c_int)]
DD132X_CloseDevice = _stdcall_libraries['axdd132x.dll'].DD132X_CloseDevice
DD132X_CloseDevice.restype = BOOL
DD132X_CloseDevice.argtypes = [HDD132X, POINTER(c_int)]
DD132X_GetDeviceInfo = _stdcall_libraries['axdd132x.dll'].DD132X_GetDeviceInfo
DD132X_GetDeviceInfo.restype = BOOL
DD132X_GetDeviceInfo.argtypes = [HDD132X, POINTER(DD132X_Info), POINTER(c_int)]
DD132X_Reset = _stdcall_libraries['axdd132x.dll'].DD132X_Reset
DD132X_Reset.restype = BOOL
DD132X_Reset.argtypes = [HDD132X, POINTER(c_int)]
DD132X_DownloadRAMware = _stdcall_libraries['axdd132x.dll'].DD132X_DownloadRAMware
DD132X_DownloadRAMware.restype = BOOL
DD132X_DownloadRAMware.argtypes = [HDD132X, POINTER(BYTE), UINT, POINTER(c_int)]
DD132X_SetProtocol = _stdcall_libraries['axdd132x.dll'].DD132X_SetProtocol
DD132X_SetProtocol.restype = BOOL
DD132X_SetProtocol.argtypes = [HDD132X, POINTER(DD132X_Protocol), POINTER(c_int)]
DD132X_GetProtocol = _stdcall_libraries['axdd132x.dll'].DD132X_GetProtocol
DD132X_GetProtocol.restype = BOOL
DD132X_GetProtocol.argtypes = [HDD132X, POINTER(DD132X_Protocol), POINTER(c_int)]
DD132X_StartAcquisition = _stdcall_libraries['axdd132x.dll'].DD132X_StartAcquisition
DD132X_StartAcquisition.restype = BOOL
DD132X_StartAcquisition.argtypes = [HDD132X, POINTER(c_int)]
DD132X_StopAcquisition = _stdcall_libraries['axdd132x.dll'].DD132X_StopAcquisition
DD132X_StopAcquisition.restype = BOOL
DD132X_StopAcquisition.argtypes = [HDD132X, POINTER(c_int)]
DD132X_PauseAcquisition = _stdcall_libraries['axdd132x.dll'].DD132X_PauseAcquisition
DD132X_PauseAcquisition.restype = BOOL
DD132X_PauseAcquisition.argtypes = [HDD132X, BOOL, POINTER(c_int)]
DD132X_IsAcquiring = _stdcall_libraries['axdd132x.dll'].DD132X_IsAcquiring
DD132X_IsAcquiring.restype = BOOL
DD132X_IsAcquiring.argtypes = [HDD132X]
DD132X_IsPaused = _stdcall_libraries['axdd132x.dll'].DD132X_IsPaused
DD132X_IsPaused.restype = BOOL
DD132X_IsPaused.argtypes = [HDD132X]
DD132X_GetTimeAtStartOfAcquisition = _stdcall_libraries['axdd132x.dll'].DD132X_GetTimeAtStartOfAcquisition
DD132X_GetTimeAtStartOfAcquisition.restype = BOOL
DD132X_GetTimeAtStartOfAcquisition.argtypes = [HDD132X, POINTER(DD132X_StartAcqInfo)]
DD132X_StartReadLast = _stdcall_libraries['axdd132x.dll'].DD132X_StartReadLast
DD132X_StartReadLast.restype = BOOL
DD132X_StartReadLast.argtypes = [HDD132X, POINTER(c_int)]
DD132X_ReadLast = _stdcall_libraries['axdd132x.dll'].DD132X_ReadLast
DD132X_ReadLast.restype = BOOL
DD132X_ReadLast.argtypes = [HDD132X, POINTER(ADC_VALUE), UINT, POINTER(c_int)]
DD132X_GetAcquisitionPosition = _stdcall_libraries['axdd132x.dll'].DD132X_GetAcquisitionPosition
DD132X_GetAcquisitionPosition.restype = BOOL
DD132X_GetAcquisitionPosition.argtypes = [HDD132X, POINTER(LONGLONG), POINTER(c_int)]
DD132X_GetNumSamplesOutput = _stdcall_libraries['axdd132x.dll'].DD132X_GetNumSamplesOutput
DD132X_GetNumSamplesOutput.restype = BOOL
DD132X_GetNumSamplesOutput.argtypes = [HDD132X, POINTER(LONGLONG), POINTER(c_int)]
DD132X_GetAIValue = _stdcall_libraries['axdd132x.dll'].DD132X_GetAIValue
DD132X_GetAIValue.restype = BOOL
DD132X_GetAIValue.argtypes = [HDD132X, UINT, POINTER(c_short), POINTER(c_int)]
DD132X_GetDIValues = _stdcall_libraries['axdd132x.dll'].DD132X_GetDIValues
DD132X_GetDIValues.restype = BOOL
DD132X_GetDIValues.argtypes = [HDD132X, POINTER(DWORD), POINTER(c_int)]
DD132X_PutAOValue = _stdcall_libraries['axdd132x.dll'].DD132X_PutAOValue
DD132X_PutAOValue.restype = BOOL
DD132X_PutAOValue.argtypes = [HDD132X, UINT, c_short, POINTER(c_int)]
DD132X_PutDOValues = _stdcall_libraries['axdd132x.dll'].DD132X_PutDOValues
DD132X_PutDOValues.restype = BOOL
DD132X_PutDOValues.argtypes = [HDD132X, DWORD, POINTER(c_int)]
DD132X_GetTelegraphs = _stdcall_libraries['axdd132x.dll'].DD132X_GetTelegraphs
DD132X_GetTelegraphs.restype = BOOL
DD132X_GetTelegraphs.argtypes = [HDD132X, UINT, POINTER(c_short), UINT, POINTER(c_int)]
DD132X_SetPowerOnOutputs = _stdcall_libraries['axdd132x.dll'].DD132X_SetPowerOnOutputs
DD132X_SetPowerOnOutputs.restype = BOOL
DD132X_SetPowerOnOutputs.argtypes = [HDD132X, POINTER(DD132X_PowerOnData), POINTER(c_int)]
DD132X_GetPowerOnOutputs = _stdcall_libraries['axdd132x.dll'].DD132X_GetPowerOnOutputs
DD132X_GetPowerOnOutputs.restype = BOOL
DD132X_GetPowerOnOutputs.argtypes = [HDD132X, POINTER(DD132X_PowerOnData), POINTER(c_int)]
DD132X_Calibrate = _stdcall_libraries['axdd132x.dll'].DD132X_Calibrate
DD132X_Calibrate.restype = BOOL
DD132X_Calibrate.argtypes = [HDD132X, POINTER(DD132X_CalibrationData), POINTER(c_int)]
DD132X_GetCalibrationData = _stdcall_libraries['axdd132x.dll'].DD132X_GetCalibrationData
DD132X_GetCalibrationData.restype = BOOL
DD132X_GetCalibrationData.argtypes = [HDD132X, POINTER(DD132X_CalibrationData), POINTER(c_int)]
DD132X_GetScsiTermStatus = _stdcall_libraries['axdd132x.dll'].DD132X_GetScsiTermStatus
DD132X_GetScsiTermStatus.restype = BOOL
DD132X_GetScsiTermStatus.argtypes = [HDD132X, POINTER(BYTE), POINTER(c_int)]
DD132X_DTermRead = _stdcall_libraries['axdd132x.dll'].DD132X_DTermRead
DD132X_DTermRead.restype = BOOL
DD132X_DTermRead.argtypes = [HDD132X, LPSTR, UINT, POINTER(c_int)]
DD132X_DTermWrite = _stdcall_libraries['axdd132x.dll'].DD132X_DTermWrite
DD132X_DTermWrite.restype = BOOL
DD132X_DTermWrite.argtypes = [HDD132X, LPCSTR, POINTER(c_int)]
DD132X_DTermSetBaudRate = _stdcall_libraries['axdd132x.dll'].DD132X_DTermSetBaudRate
DD132X_DTermSetBaudRate.restype = BOOL
DD132X_DTermSetBaudRate.argtypes = [HDD132X, UINT, POINTER(c_int)]
DD132X_GetLastErrorText = _stdcall_libraries['axdd132x.dll'].DD132X_GetLastErrorText
DD132X_GetLastErrorText.restype = BOOL
DD132X_GetLastErrorText.argtypes = [HDD132X, STRING, UINT, POINTER(c_int)]
DD132X_SetDebugMsgLevel = _stdcall_libraries['axdd132x.dll'].DD132X_SetDebugMsgLevel
DD132X_SetDebugMsgLevel.restype = BOOL
DD132X_SetDebugMsgLevel.argtypes = [HDD132X, UINT, POINTER(c_int)]
DD132X_UpdateThresholdLevel = _stdcall_libraries['axdd132x.dll'].DD132X_UpdateThresholdLevel
DD132X_UpdateThresholdLevel.restype = BOOL
DD132X_UpdateThresholdLevel.argtypes = [HDD132X, POINTER(WORD), POINTER(WORD)]
__all__ = ['DD132X_Bit0ExtStart', 'PDWORD', 'ADC_VALUE',
           'DD132X_DTermWrite', 'DD132X_SetDebugMsgLevel',
           'DD132X_UpdateThresholdLevel', 'DD132X_Bit0Tag_Bit1Line',
           'DAC_VALUE', 'PFLOATBUFFER', 'PUSHORT', 'PUCHAR',
           'DD132X_PutAOValue', '_SYSTEMTIME', 'DD132X_GetDeviceInfo',
           'DD132X_StartAcquisition', 'FLOATBUFFER', 'LPVOID',
           'DD132X_Calibrate', 'LONGLONG', 'DD132X_GetLastErrorText',
           'DD132X_PowerOnData', 'LPINT', 'DD132X_GetScsiTermStatus',
           'DD132X_Info', 'DD132X_StartReadLast', 'PUINT',
           'DD132X_GetPowerOnOutputs', 'DD132X_GetNumSamplesOutput',
           'DD132X_GetAIValue', 'DD132X_GetDIValues', 'LPCVOID',
           'DD132X_DTermSetBaudRate', 'DD132X_SetPowerOnOutputs',
           'DD132X_CloseDevice', 'PBOOL', 'DD132X_OutputPulseType',
           'UCHAR', 'DD132X_DTermRead', 'INT',
           'DD132X_PauseAcquisition', 'DD132X_Bit0Line', 'PSZ',
           'PSYSTEMTIME', 'PULONG', 'DD132X_RescanSCSIBus',
           'SYSTEMTIME', 'DD132X_StartAcqInfo', 'DD132X_ReadLast',
           'DD132X_IsAcquiring', 'LPWORD', 'DATABUFFER', 'LPBYTE',
           'DD132X_AIDataBits', 'DD132X_ADC_level_Triggered',
           'DD132X_GetAcquisitionPosition',
           'DD132X_DAC_bit0_Triggered', 'DD132X_NoOutputPulse',
           'DD132X_Triggering', 'DD132X_Reset', 'DD132X_GetProtocol',
           'DD132X_ExternalStart', 'DD132X_OpenDeviceEx', 'USHORT',
           'LPDWORD', 'DD132X_IsPaused', 'HDD132X', 'FLOAT', 'LPLONG',
           'DD132X_LineTrigger', 'DD132X_GetCalibrationData',
           'LPBOOL', 'DD132X_GetTelegraphs', 'LPSYSTEMTIME',
           'DD132X_StartImmediately', 'PDATABUFFER',
           'DD132X_Protocol', 'DD132X_DownloadRAMware',
           'DD132X_FindDevices', 'DD132X_Bit0Tag',
           'DD132X_OpenDevice', 'DD132X_PutDOValues',
           'DD132X_Bit0Data', 'DD132X_SetProtocol', 'PBYTE',
           'DD132X_CalibrationData', 'DD132X_StopAcquisition',
           'DD132X_Bit0Tag_Bit1ExtStart', 'PWORD', 'PFLOAT',
           'DD132X_GetTimeAtStartOfAcquisition', 'PINT']
