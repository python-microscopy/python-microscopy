#!/usr/bin/python

##################
# AndorCam.py
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
import platform
#import ctypes

_stdcall_libraries = {}

arch, plat = platform.architecture()


#if 'WinDLL' in dir():
if plat.startswith('Windows'):
    if arch == '32bit':
        _stdcall_libraries['ATMCD32D'] = WinDLL('ATMCD32D')
    else:
        try:
            _stdcall_libraries['ATMCD32D'] = WinDLL('atmcd64d')
        except OSError:
            # see https://stackoverflow.com/questions/59330863/cant-import-dll-module-in-python
            # winmode=0 enforces windows default dll search mechanism including searching the path set
            # necessary since python 3.8.x
            _stdcall_libraries['ATMCD32D'] = WinDLL('atmcd64d',winmode=0)
    from ctypes.wintypes import ULONG, DWORD, BOOL, BYTE, WORD, UINT, HANDLE, HWND
else:
    _stdcall_libraries['ATMCD32D'] = CDLL('libandor.so')

    #from ctypes.wintypes import ULONG
    ULONG = c_ulong
    #from ctypes.wintypes import DWORD
    DWORD = c_uint16
    #from ctypes.wintypes import BOOL
    BOOL = c_int
    #from ctypes.wintypes import BYTE
    BYTE = c_ubyte
    #from ctypes.wintypes import WORD
    WORD = c_int16
    #from ctypes.wintypes import UINT
    #from ctypes.wintypes import HANDLE
    HANDLE=WORD
    HWND = WORD
    #from ctypes.wintypes import HWND


STRING = c_char_p

PULONG = POINTER(ULONG)
USHORT = c_ushort
PUSHORT = POINTER(USHORT)
UCHAR = c_ubyte
PUCHAR = POINTER(UCHAR)
PSZ = STRING
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

class ANDORCAPS(Structure):
    pass
ANDORCAPS._fields_ = [
    ('ulSize', ULONG),
    ('ulAcqModes', ULONG),
    ('ulReadModes', ULONG),
    ('ulTriggerModes', ULONG),
    ('ulCameraType', ULONG),
    ('ulPixelMode', ULONG),
    ('ulSetFunctions', ULONG),
    ('ulGetFunctions', ULONG),
    ('ulFeatures', ULONG),
    ('ulPCICard', ULONG),
    ('ulEMGainCapability', ULONG),
]
AndorCapabilities = ANDORCAPS
class COLORDEMOSAICINFO(Structure):
    pass
COLORDEMOSAICINFO._fields_ = [
    ('iX', c_int),
    ('iY', c_int),
    ('iAlgorithm', c_int),
    ('iXPhase', c_int),
    ('iYPhase', c_int),
    ('iBackground', c_int),
]
ColorDemosaicInfo = COLORDEMOSAICINFO
AbortAcquisition = _stdcall_libraries['ATMCD32D'].AbortAcquisition
AbortAcquisition.restype = c_uint
AbortAcquisition.argtypes = []
CancelWait = _stdcall_libraries['ATMCD32D'].CancelWait
CancelWait.restype = c_uint
CancelWait.argtypes = []
CoolerOFF = _stdcall_libraries['ATMCD32D'].CoolerOFF
CoolerOFF.restype = c_uint
CoolerOFF.argtypes = []
CoolerON = _stdcall_libraries['ATMCD32D'].CoolerON
CoolerON.restype = c_uint
CoolerON.argtypes = []
DemosaicImage = _stdcall_libraries['ATMCD32D'].DemosaicImage
DemosaicImage.restype = c_uint
DemosaicImage.argtypes = [POINTER(WORD), POINTER(WORD), POINTER(WORD), POINTER(WORD), POINTER(ColorDemosaicInfo)]
FreeInternalMemory = _stdcall_libraries['ATMCD32D'].FreeInternalMemory
FreeInternalMemory.restype = c_uint
FreeInternalMemory.argtypes = []
GetAcquiredData = _stdcall_libraries['ATMCD32D'].GetAcquiredData
GetAcquiredData.restype = c_uint
GetAcquiredData.argtypes = [POINTER(c_long), c_ulong]
GetAcquiredData16 = _stdcall_libraries['ATMCD32D'].GetAcquiredData16
GetAcquiredData16.restype = c_uint
GetAcquiredData16.argtypes = [POINTER(WORD), c_ulong]
GetAcquiredFloatData = _stdcall_libraries['ATMCD32D'].GetAcquiredFloatData
GetAcquiredFloatData.restype = c_uint
GetAcquiredFloatData.argtypes = [POINTER(c_float), c_ulong]
GetAcquisitionProgress = _stdcall_libraries['ATMCD32D'].GetAcquisitionProgress
GetAcquisitionProgress.restype = c_uint
GetAcquisitionProgress.argtypes = [POINTER(c_long), POINTER(c_long)]
GetAcquisitionTimings = _stdcall_libraries['ATMCD32D'].GetAcquisitionTimings
GetAcquisitionTimings.restype = c_uint
GetAcquisitionTimings.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float)]
GetAllDMAData = _stdcall_libraries['ATMCD32D'].GetAllDMAData
GetAllDMAData.restype = c_uint
GetAllDMAData.argtypes = [POINTER(c_long), c_ulong]
GetAmpDesc = _stdcall_libraries['ATMCD32D'].GetAmpDesc
GetAmpDesc.restype = c_uint
GetAmpDesc.argtypes = [c_int, STRING, c_int]
GetAmpMaxSpeed = _stdcall_libraries['ATMCD32D'].GetAmpMaxSpeed
GetAmpMaxSpeed.restype = c_uint
GetAmpMaxSpeed.argtypes = [c_int, POINTER(c_float)]
GetAvailableCameras = _stdcall_libraries['ATMCD32D'].GetAvailableCameras
GetAvailableCameras.restype = c_uint
GetAvailableCameras.argtypes = [POINTER(c_long)]
GetBackground = _stdcall_libraries['ATMCD32D'].GetBackground
GetBackground.restype = c_uint
GetBackground.argtypes = [POINTER(c_long), c_ulong]
GetBitDepth = _stdcall_libraries['ATMCD32D'].GetBitDepth
GetBitDepth.restype = c_uint
GetBitDepth.argtypes = [c_int, POINTER(c_int)]
#GetCameraEventStatus = _stdcall_libraries['ATMCD32D'].GetCameraEventStatus
#GetCameraEventStatus.restype = c_uint
#GetCameraEventStatus.argtypes = [POINTER(DWORD)]
GetCameraHandle = _stdcall_libraries['ATMCD32D'].GetCameraHandle
GetCameraHandle.restype = c_uint
GetCameraHandle.argtypes = [c_long, POINTER(c_long)]
GetCameraInformation = _stdcall_libraries['ATMCD32D'].GetCameraInformation
GetCameraInformation.restype = c_uint
GetCameraInformation.argtypes = [c_int, POINTER(c_long)]
GetCameraSerialNumber = _stdcall_libraries['ATMCD32D'].GetCameraSerialNumber
GetCameraSerialNumber.restype = c_uint
GetCameraSerialNumber.argtypes = [POINTER(c_int)]
GetCapabilities = _stdcall_libraries['ATMCD32D'].GetCapabilities
GetCapabilities.restype = c_uint
GetCapabilities.argtypes = [POINTER(AndorCapabilities)]
GetControllerCardModel = _stdcall_libraries['ATMCD32D'].GetControllerCardModel
GetControllerCardModel.restype = c_uint
GetControllerCardModel.argtypes = [STRING]
GetCurrentCamera = _stdcall_libraries['ATMCD32D'].GetCurrentCamera
GetCurrentCamera.restype = c_uint
GetCurrentCamera.argtypes = [POINTER(c_long)]
GetDDGIOCPulses = _stdcall_libraries['ATMCD32D'].GetDDGIOCPulses
GetDDGIOCPulses.restype = c_uint
GetDDGIOCPulses.argtypes = [POINTER(c_int)]
GetDDGPulse = _stdcall_libraries['ATMCD32D'].GetDDGPulse
GetDDGPulse.restype = c_uint
GetDDGPulse.argtypes = [c_double, c_double, POINTER(c_double), POINTER(c_double)]
GetDetector = _stdcall_libraries['ATMCD32D'].GetDetector
GetDetector.restype = c_uint
GetDetector.argtypes = [POINTER(c_int), POINTER(c_int)]
GetDICameraInfo = _stdcall_libraries['ATMCD32D'].GetDICameraInfo
GetDICameraInfo.restype = c_uint
GetDICameraInfo.argtypes = [c_void_p]
GetEMCCDGain = _stdcall_libraries['ATMCD32D'].GetEMCCDGain
GetEMCCDGain.restype = c_uint
GetEMCCDGain.argtypes = [POINTER(c_int)]
GetEMGainRange = _stdcall_libraries['ATMCD32D'].GetEMGainRange
GetEMGainRange.restype = c_uint
GetEMGainRange.argtypes = [POINTER(c_int), POINTER(c_int)]
GetFastestRecommendedVSSpeed = _stdcall_libraries['ATMCD32D'].GetFastestRecommendedVSSpeed
GetFastestRecommendedVSSpeed.restype = c_uint
GetFastestRecommendedVSSpeed.argtypes = [POINTER(c_int), POINTER(c_float)]
GetFilterMode = _stdcall_libraries['ATMCD32D'].GetFilterMode
GetFilterMode.restype = c_uint
GetFilterMode.argtypes = [POINTER(c_int)]
GetFIFOUsage = _stdcall_libraries['ATMCD32D'].GetFIFOUsage
GetFIFOUsage.restype = c_uint
GetFIFOUsage.argtypes = [POINTER(c_int)]
GetFKExposureTime = _stdcall_libraries['ATMCD32D'].GetFKExposureTime
GetFKExposureTime.restype = c_uint
GetFKExposureTime.argtypes = [POINTER(c_float)]
GetFKVShiftSpeed = _stdcall_libraries['ATMCD32D'].GetFKVShiftSpeed
GetFKVShiftSpeed.restype = c_uint
GetFKVShiftSpeed.argtypes = [c_int, POINTER(c_int)]
GetFKVShiftSpeedF = _stdcall_libraries['ATMCD32D'].GetFKVShiftSpeedF
GetFKVShiftSpeedF.restype = c_uint
GetFKVShiftSpeedF.argtypes = [c_int, POINTER(c_float)]
GetHardwareVersion = _stdcall_libraries['ATMCD32D'].GetHardwareVersion
GetHardwareVersion.restype = c_uint
GetHardwareVersion.argtypes = [POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), POINTER(c_uint)]
GetHeadModel = _stdcall_libraries['ATMCD32D'].GetHeadModel
GetHeadModel.restype = c_uint
GetHeadModel.argtypes = [STRING]
GetHorizontalSpeed = _stdcall_libraries['ATMCD32D'].GetHorizontalSpeed
GetHorizontalSpeed.restype = c_uint
GetHorizontalSpeed.argtypes = [c_int, POINTER(c_int)]
GetHSSpeed = _stdcall_libraries['ATMCD32D'].GetHSSpeed
GetHSSpeed.restype = c_uint
GetHSSpeed.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
GetHVflag = _stdcall_libraries['ATMCD32D'].GetHVflag
GetHVflag.restype = c_uint
GetHVflag.argtypes = [POINTER(c_int)]
GetID = _stdcall_libraries['ATMCD32D'].GetID
GetID.restype = c_uint
GetID.argtypes = [c_int, POINTER(c_int)]
GetImages = _stdcall_libraries['ATMCD32D'].GetImages
GetImages.restype = c_uint
GetImages.argtypes = [c_long, c_long, POINTER(c_long), c_ulong, POINTER(c_long), POINTER(c_long)]
GetImages16 = _stdcall_libraries['ATMCD32D'].GetImages16
GetImages16.restype = c_uint
GetImages16.argtypes = [c_long, c_long, POINTER(WORD), c_ulong, POINTER(c_long), POINTER(c_long)]
GetImagesPerDMA = _stdcall_libraries['ATMCD32D'].GetImagesPerDMA
GetImagesPerDMA.restype = c_uint
GetImagesPerDMA.argtypes = [POINTER(c_ulong)]
GetMaximumBinning = _stdcall_libraries['ATMCD32D'].GetMaximumBinning
GetMaximumBinning.restype = c_uint
GetMaximumBinning.argtypes = [c_int, c_int, POINTER(c_int)]
GetMaximumExposure = _stdcall_libraries['ATMCD32D'].GetMaximumExposure
GetMaximumExposure.restype = c_uint
GetMaximumExposure.argtypes = [POINTER(c_float)]
GetMinimumImageLength = _stdcall_libraries['ATMCD32D'].GetMinimumImageLength
GetMinimumImageLength.restype = c_uint
GetMinimumImageLength.argtypes = [POINTER(c_int)]
GetMCPGain = _stdcall_libraries['ATMCD32D'].GetMCPGain
GetMCPGain.restype = c_uint
GetMCPGain.argtypes = [c_int, POINTER(c_int), POINTER(c_float)]
GetMCPVoltage = _stdcall_libraries['ATMCD32D'].GetMCPVoltage
GetMCPVoltage.restype = c_uint
GetMCPVoltage.argtypes = [POINTER(c_int)]
GetMostRecentColorImage16 = _stdcall_libraries['ATMCD32D'].GetMostRecentColorImage16
GetMostRecentColorImage16.restype = c_uint
GetMostRecentColorImage16.argtypes = [c_ulong, c_int, POINTER(WORD), POINTER(WORD), POINTER(WORD)]
GetMostRecentImage = _stdcall_libraries['ATMCD32D'].GetMostRecentImage
GetMostRecentImage.restype = c_uint
GetMostRecentImage.argtypes = [POINTER(c_long), c_ulong]
GetMostRecentImage16 = _stdcall_libraries['ATMCD32D'].GetMostRecentImage16
GetMostRecentImage16.restype = c_uint
GetMostRecentImage16.argtypes = [POINTER(WORD), c_ulong]
GetNewData = _stdcall_libraries['ATMCD32D'].GetNewData
GetNewData.restype = c_uint
GetNewData.argtypes = [POINTER(c_long), c_ulong]
GetNewData16 = _stdcall_libraries['ATMCD32D'].GetNewData16
GetNewData16.restype = c_uint
GetNewData16.argtypes = [POINTER(WORD), c_ulong]
GetNewFloatData = _stdcall_libraries['ATMCD32D'].GetNewFloatData
GetNewFloatData.restype = c_uint
GetNewFloatData.argtypes = [POINTER(c_float), c_ulong]
GetNumberADChannels = _stdcall_libraries['ATMCD32D'].GetNumberADChannels
GetNumberADChannels.restype = c_uint
GetNumberADChannels.argtypes = [POINTER(c_int)]
GetNumberAmp = _stdcall_libraries['ATMCD32D'].GetNumberAmp
GetNumberAmp.restype = c_uint
GetNumberAmp.argtypes = [POINTER(c_int)]
GetNumberDevices = _stdcall_libraries['ATMCD32D'].GetNumberDevices
GetNumberDevices.restype = c_uint
GetNumberDevices.argtypes = [POINTER(c_int)]
GetNumberFKVShiftSpeeds = _stdcall_libraries['ATMCD32D'].GetNumberFKVShiftSpeeds
GetNumberFKVShiftSpeeds.restype = c_uint
GetNumberFKVShiftSpeeds.argtypes = [POINTER(c_int)]
GetNumberHorizontalSpeeds = _stdcall_libraries['ATMCD32D'].GetNumberHorizontalSpeeds
GetNumberHorizontalSpeeds.restype = c_uint
GetNumberHorizontalSpeeds.argtypes = [POINTER(c_int)]
GetNumberHSSpeeds = _stdcall_libraries['ATMCD32D'].GetNumberHSSpeeds
GetNumberHSSpeeds.restype = c_uint
GetNumberHSSpeeds.argtypes = [c_int, c_int, POINTER(c_int)]
GetNumberNewImages = _stdcall_libraries['ATMCD32D'].GetNumberNewImages
GetNumberNewImages.restype = c_uint
GetNumberNewImages.argtypes = [POINTER(c_long), POINTER(c_long)]
GetNumberPreAmpGains = _stdcall_libraries['ATMCD32D'].GetNumberPreAmpGains
GetNumberPreAmpGains.restype = c_uint
GetNumberPreAmpGains.argtypes = [POINTER(c_int)]
GetNumberVerticalSpeeds = _stdcall_libraries['ATMCD32D'].GetNumberVerticalSpeeds
GetNumberVerticalSpeeds.restype = c_uint
GetNumberVerticalSpeeds.argtypes = [POINTER(c_int)]
GetNumberVSAmplitudes = _stdcall_libraries['ATMCD32D'].GetNumberVSAmplitudes
GetNumberVSAmplitudes.restype = c_uint
GetNumberVSAmplitudes.argtypes = [POINTER(c_int)]
GetNumberVSSpeeds = _stdcall_libraries['ATMCD32D'].GetNumberVSSpeeds
GetNumberVSSpeeds.restype = c_uint
GetNumberVSSpeeds.argtypes = [POINTER(c_int)]
GetOldestImage = _stdcall_libraries['ATMCD32D'].GetOldestImage
GetOldestImage.restype = c_uint
GetOldestImage.argtypes = [POINTER(c_long), c_ulong]
GetOldestImage16 = _stdcall_libraries['ATMCD32D'].GetOldestImage16
GetOldestImage16.restype = c_uint
GetOldestImage16.argtypes = [POINTER(WORD), c_ulong]
GetPhysicalDMAAddress = _stdcall_libraries['ATMCD32D'].GetPhysicalDMAAddress
GetPhysicalDMAAddress.restype = c_uint
GetPhysicalDMAAddress.argtypes = [c_ulong, c_ulong]
GetPixelSize = _stdcall_libraries['ATMCD32D'].GetPixelSize
GetPixelSize.restype = c_uint
GetPixelSize.argtypes = [POINTER(c_float), POINTER(c_float)]
GetPreAmpGain = _stdcall_libraries['ATMCD32D'].GetPreAmpGain
GetPreAmpGain.restype = c_uint
GetPreAmpGain.argtypes = [c_int, POINTER(c_float)]
GetRegisterDump = _stdcall_libraries['ATMCD32D'].GetRegisterDump
GetRegisterDump.restype = c_uint
GetRegisterDump.argtypes = [POINTER(c_int)]
GetSizeOfCircularBuffer = _stdcall_libraries['ATMCD32D'].GetSizeOfCircularBuffer
GetSizeOfCircularBuffer.restype = c_uint
GetSizeOfCircularBuffer.argtypes = [POINTER(c_long)]
GetSlotBusDeviceFunction = _stdcall_libraries['ATMCD32D'].GetSlotBusDeviceFunction
GetSlotBusDeviceFunction.restype = c_uint
GetSlotBusDeviceFunction.argtypes = [POINTER(DWORD), POINTER(DWORD), POINTER(DWORD), POINTER(DWORD)]
GetSoftwareVersion = _stdcall_libraries['ATMCD32D'].GetSoftwareVersion
GetSoftwareVersion.restype = c_uint
GetSoftwareVersion.argtypes = [POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), POINTER(c_uint)]
GetSpoolProgress = _stdcall_libraries['ATMCD32D'].GetSpoolProgress
GetSpoolProgress.restype = c_uint
GetSpoolProgress.argtypes = [POINTER(c_long)]
GetStatus = _stdcall_libraries['ATMCD32D'].GetStatus
GetStatus.restype = c_uint
GetStatus.argtypes = [POINTER(c_int)]
GetTemperature = _stdcall_libraries['ATMCD32D'].GetTemperature
GetTemperature.restype = c_uint
GetTemperature.argtypes = [POINTER(c_int)]
GetTemperatureF = _stdcall_libraries['ATMCD32D'].GetTemperatureF
GetTemperatureF.restype = c_uint
GetTemperatureF.argtypes = [POINTER(c_float)]
GetTemperatureStatus = _stdcall_libraries['ATMCD32D'].GetTemperatureStatus
GetTemperatureStatus.restype = c_uint
GetTemperatureStatus.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float)]
GetTemperatureRange = _stdcall_libraries['ATMCD32D'].GetTemperatureRange
GetTemperatureRange.restype = c_uint
GetTemperatureRange.argtypes = [POINTER(c_int), POINTER(c_int)]
GetTotalNumberImagesAcquired = _stdcall_libraries['ATMCD32D'].GetTotalNumberImagesAcquired
GetTotalNumberImagesAcquired.restype = c_uint
GetTotalNumberImagesAcquired.argtypes = [POINTER(c_long)]
GetVerticalSpeed = _stdcall_libraries['ATMCD32D'].GetVerticalSpeed
GetVerticalSpeed.restype = c_uint
GetVerticalSpeed.argtypes = [c_int, POINTER(c_int)]
GetVSSpeed = _stdcall_libraries['ATMCD32D'].GetVSSpeed
GetVSSpeed.restype = c_uint
GetVSSpeed.argtypes = [c_int, POINTER(c_float)]
GPIBReceive = _stdcall_libraries['ATMCD32D'].GPIBReceive
GPIBReceive.restype = c_uint
GPIBReceive.argtypes = [c_int, c_short, STRING, c_int]
GPIBSend = _stdcall_libraries['ATMCD32D'].GPIBSend
GPIBSend.restype = c_uint
GPIBSend.argtypes = [c_int, c_short, STRING]
I2CBurstRead = _stdcall_libraries['ATMCD32D'].I2CBurstRead
I2CBurstRead.restype = c_uint
I2CBurstRead.argtypes = [BYTE, c_long, POINTER(BYTE)]
I2CBurstWrite = _stdcall_libraries['ATMCD32D'].I2CBurstWrite
I2CBurstWrite.restype = c_uint
I2CBurstWrite.argtypes = [BYTE, c_long, POINTER(BYTE)]
I2CRead = _stdcall_libraries['ATMCD32D'].I2CRead
I2CRead.restype = c_uint
I2CRead.argtypes = [BYTE, BYTE, POINTER(BYTE)]
I2CReset = _stdcall_libraries['ATMCD32D'].I2CReset
I2CReset.restype = c_uint
I2CReset.argtypes = []
I2CWrite = _stdcall_libraries['ATMCD32D'].I2CWrite
I2CWrite.restype = c_uint
I2CWrite.argtypes = [BYTE, BYTE, BYTE]
IdAndorDll = _stdcall_libraries['ATMCD32D'].IdAndorDll
IdAndorDll.restype = c_uint
IdAndorDll.argtypes = []
InAuxPort = _stdcall_libraries['ATMCD32D'].InAuxPort
InAuxPort.restype = c_uint
InAuxPort.argtypes = [c_int, POINTER(c_int)]
Initialize = _stdcall_libraries['ATMCD32D'].Initialize
Initialize.restype = c_uint
Initialize.argtypes = [STRING]
InitializeDevice = _stdcall_libraries['ATMCD32D'].InitializeDevice
InitializeDevice.restype = c_uint
InitializeDevice.argtypes = [STRING]
IsInternalMechanicalShutter = _stdcall_libraries['ATMCD32D'].IsInternalMechanicalShutter
IsInternalMechanicalShutter.restype = c_uint
IsInternalMechanicalShutter.argtypes = [POINTER(c_int)]
IsPreAmpGainAvailable = _stdcall_libraries['ATMCD32D'].IsPreAmpGainAvailable
IsPreAmpGainAvailable.restype = c_uint
IsPreAmpGainAvailable.argtypes = [c_int, c_int, c_int, c_int, POINTER(c_int)]
Merge = _stdcall_libraries['ATMCD32D'].Merge
Merge.restype = c_uint
Merge.argtypes = [POINTER(c_long), c_long, c_long, c_long, POINTER(c_float), c_long, c_long, POINTER(c_long), POINTER(c_float), POINTER(c_float)]
OutAuxPort = _stdcall_libraries['ATMCD32D'].OutAuxPort
OutAuxPort.restype = c_uint
OutAuxPort.argtypes = [c_int, c_int]
#OWInit = _stdcall_libraries['ATMCD32D'].OWInit
#OWInit.restype = c_uint
#OWInit.argtypes = [c_uint]
#OWGetNbDevices = _stdcall_libraries['ATMCD32D'].OWGetNbDevices
#OWGetNbDevices.restype = c_uint
#OWGetNbDevices.argtypes = []
#OWGetDeviceID = _stdcall_libraries['ATMCD32D'].OWGetDeviceID
#OWGetDeviceID.restype = c_uint
#OWGetDeviceID.argtypes = [c_uint, POINTER(BYTE)]
#OWPulsePIO = _stdcall_libraries['ATMCD32D'].OWPulsePIO
#OWPulsePIO.restype = c_uint
#OWPulsePIO.argtypes = [c_uint, BYTE]
#OWReadMem = _stdcall_libraries['ATMCD32D'].OWReadMem
#OWReadMem.restype = c_uint
#OWReadMem.argtypes = [c_uint, WORD, c_ulong, POINTER(BYTE)]
#OWReadPIO = _stdcall_libraries['ATMCD32D'].OWReadPIO
#OWReadPIO.restype = c_uint
#OWReadPIO.argtypes = [c_uint, POINTER(BYTE)]
#OWResetPIOActivityLatch = _stdcall_libraries['ATMCD32D'].OWResetPIOActivityLatch
#OWResetPIOActivityLatch.restype = c_uint
#OWResetPIOActivityLatch.argtypes = [c_uint]
#OWSetMemPageRegister = _stdcall_libraries['ATMCD32D'].OWSetMemPageRegister
#OWSetMemPageRegister.restype = c_uint
#OWSetMemPageRegister.argtypes = [c_uint, c_ushort, BYTE]
#OWSetPIO = _stdcall_libraries['ATMCD32D'].OWSetPIO
#OWSetPIO.restype = c_uint
#OWSetPIO.argtypes = [c_uint, BYTE, c_uint]
#OWLockMemPageRegister = _stdcall_libraries['ATMCD32D'].OWLockMemPageRegister
#OWLockMemPageRegister.restype = c_uint
#OWLockMemPageRegister.argtypes = [c_uint, c_ushort, BYTE]
#OWWriteMem = _stdcall_libraries['ATMCD32D'].OWWriteMem
#OWWriteMem.restype = c_uint
#OWWriteMem.argtypes = [c_uint, WORD, c_ulong, POINTER(BYTE)]
#OWWritePIORegister = _stdcall_libraries['ATMCD32D'].OWWritePIORegister
#OWWritePIORegister.restype = c_uint
#OWWritePIORegister.argtypes = [c_uint, c_ushort, BYTE]
#OWReadPIORegister = _stdcall_libraries['ATMCD32D'].OWReadPIORegister
#OWReadPIORegister.restype = c_uint
#OWReadPIORegister.argtypes = [c_uint, c_ushort, POINTER(BYTE)]
#OWSetTransmissionMode = _stdcall_libraries['ATMCD32D'].OWSetTransmissionMode
#OWSetTransmissionMode.restype = c_uint
#OWSetTransmissionMode.argtypes = [c_ushort]
PrepareAcquisition = _stdcall_libraries['ATMCD32D'].PrepareAcquisition
PrepareAcquisition.restype = c_uint
PrepareAcquisition.argtypes = []
SaveAsBmp = _stdcall_libraries['ATMCD32D'].SaveAsBmp
SaveAsBmp.restype = c_uint
SaveAsBmp.argtypes = [STRING, STRING, c_long, c_long]
SaveAsCommentedSif = _stdcall_libraries['ATMCD32D'].SaveAsCommentedSif
SaveAsCommentedSif.restype = c_uint
SaveAsCommentedSif.argtypes = [STRING, STRING]
SaveAsEDF = _stdcall_libraries['ATMCD32D'].SaveAsEDF
SaveAsEDF.restype = c_uint
SaveAsEDF.argtypes = [STRING, c_int]
SaveAsSif = _stdcall_libraries['ATMCD32D'].SaveAsSif
SaveAsSif.restype = c_uint
SaveAsSif.argtypes = [STRING]
SaveAsFITS = _stdcall_libraries['ATMCD32D'].SaveAsFITS
SaveAsFITS.restype = c_uint
SaveAsFITS.argtypes = [STRING, c_int]
SaveAsTiff = _stdcall_libraries['ATMCD32D'].SaveAsTiff
SaveAsTiff.restype = c_uint
SaveAsTiff.argtypes = [STRING, STRING, c_int, c_int]
SaveEEPROMToFile = _stdcall_libraries['ATMCD32D'].SaveEEPROMToFile
SaveEEPROMToFile.restype = c_uint
SaveEEPROMToFile.argtypes = [STRING]
SaveToClipBoard = _stdcall_libraries['ATMCD32D'].SaveToClipBoard
SaveToClipBoard.restype = c_uint
SaveToClipBoard.argtypes = [STRING]
SelectDevice = _stdcall_libraries['ATMCD32D'].SelectDevice
SelectDevice.restype = c_uint
SelectDevice.argtypes = [c_int]
SetAccumulationCycleTime = _stdcall_libraries['ATMCD32D'].SetAccumulationCycleTime
SetAccumulationCycleTime.restype = c_uint
SetAccumulationCycleTime.argtypes = [c_float]
SetAcquisitionMode = _stdcall_libraries['ATMCD32D'].SetAcquisitionMode
SetAcquisitionMode.restype = c_uint
SetAcquisitionMode.argtypes = [c_int]
SetAcquisitionType = _stdcall_libraries['ATMCD32D'].SetAcquisitionType
SetAcquisitionType.restype = c_uint
SetAcquisitionType.argtypes = [c_int]
SetADChannel = _stdcall_libraries['ATMCD32D'].SetADChannel
SetADChannel.restype = c_uint
SetADChannel.argtypes = [c_int]
SetBackground = _stdcall_libraries['ATMCD32D'].SetBackground
SetBackground.restype = c_uint
SetBackground.argtypes = [POINTER(c_long), c_ulong]
SetBaselineClamp = _stdcall_libraries['ATMCD32D'].SetBaselineClamp
SetBaselineClamp.restype = c_uint
SetBaselineClamp.argtypes = [c_int]
SetBaselineOffset = _stdcall_libraries['ATMCD32D'].SetBaselineOffset
SetBaselineOffset.restype = c_uint
SetBaselineOffset.argtypes = [c_int]
GetBaselineClamp = _stdcall_libraries['ATMCD32D'].GetBaselineClamp
GetBaselineClamp.restype = c_uint
GetBaselineClamp.argtypes = [POINTER(c_int)]
SetComplexImage = _stdcall_libraries['ATMCD32D'].SetComplexImage
SetComplexImage.restype = c_uint
SetComplexImage.argtypes = [c_int, POINTER(c_int)]
SetCoolerMode = _stdcall_libraries['ATMCD32D'].SetCoolerMode
SetCoolerMode.restype = c_uint
SetCoolerMode.argtypes = [c_int]
SetCurrentCamera = _stdcall_libraries['ATMCD32D'].SetCurrentCamera
SetCurrentCamera.restype = c_uint
SetCurrentCamera.argtypes = [c_long]
SetCustomTrackHBin = _stdcall_libraries['ATMCD32D'].SetCustomTrackHBin
SetCustomTrackHBin.restype = c_uint
SetCustomTrackHBin.argtypes = [c_int]
SetCropMode = _stdcall_libraries['ATMCD32D'].SetCropMode
SetCropMode.restype = c_uint
SetCropMode.argtypes = [c_int, c_int, c_int]
SetDataType = _stdcall_libraries['ATMCD32D'].SetDataType
SetDataType.restype = c_uint
SetDataType.argtypes = [c_int]
SetDDGAddress = _stdcall_libraries['ATMCD32D'].SetDDGAddress
SetDDGAddress.restype = c_uint
SetDDGAddress.argtypes = [BYTE, BYTE, BYTE, BYTE, BYTE]
SetDDGGain = _stdcall_libraries['ATMCD32D'].SetDDGGain
SetDDGGain.restype = c_uint
SetDDGGain.argtypes = [c_int]
SetDDGGateStep = _stdcall_libraries['ATMCD32D'].SetDDGGateStep
SetDDGGateStep.restype = c_uint
SetDDGGateStep.argtypes = [c_double]
SetDDGInsertionDelay = _stdcall_libraries['ATMCD32D'].SetDDGInsertionDelay
SetDDGInsertionDelay.restype = c_uint
SetDDGInsertionDelay.argtypes = [c_int]
SetDDGIntelligate = _stdcall_libraries['ATMCD32D'].SetDDGIntelligate
SetDDGIntelligate.restype = c_uint
SetDDGIntelligate.argtypes = [c_int]
SetDDGIOC = _stdcall_libraries['ATMCD32D'].SetDDGIOC
SetDDGIOC.restype = c_uint
SetDDGIOC.argtypes = [c_int]
SetDDGIOCFrequency = _stdcall_libraries['ATMCD32D'].SetDDGIOCFrequency
SetDDGIOCFrequency.restype = c_uint
SetDDGIOCFrequency.argtypes = [c_double]
SetDDGTimes = _stdcall_libraries['ATMCD32D'].SetDDGTimes
SetDDGTimes.restype = c_uint
SetDDGTimes.argtypes = [c_double, c_double, c_double]
SetDDGTriggerMode = _stdcall_libraries['ATMCD32D'].SetDDGTriggerMode
SetDDGTriggerMode.restype = c_uint
SetDDGTriggerMode.argtypes = [c_int]
SetDDGVariableGateStep = _stdcall_libraries['ATMCD32D'].SetDDGVariableGateStep
SetDDGVariableGateStep.restype = c_uint
SetDDGVariableGateStep.argtypes = [c_int, c_double, c_double]
SetDelayGenerator = _stdcall_libraries['ATMCD32D'].SetDelayGenerator
SetDelayGenerator.restype = c_uint
SetDelayGenerator.argtypes = [c_int, c_short, c_int]
SetDMAParameters = _stdcall_libraries['ATMCD32D'].SetDMAParameters
SetDMAParameters.restype = c_uint
SetDMAParameters.argtypes = [c_int, c_float]
#SetDriverEvent = _stdcall_libraries['ATMCD32D'].SetDriverEvent
#SetDriverEvent.restype = c_uint
#SetDriverEvent.argtypes = [HANDLE]
SetFPDP = _stdcall_libraries['ATMCD32D'].SetFPDP
SetFPDP.restype = c_uint
SetFPDP.argtypes = [c_int]
#SetAcqStatusEvent = _stdcall_libraries['ATMCD32D'].SetAcqStatusEvent
#SetAcqStatusEvent.restype = c_uint
#SetAcqStatusEvent.argtypes = [HANDLE]
#SetPCIMode = _stdcall_libraries['ATMCD32D'].SetPCIMode
#SetPCIMode.restype = c_uint
#SetPCIMode.argtypes = [c_int, c_int]
SetEMAdvanced = _stdcall_libraries['ATMCD32D'].SetEMAdvanced
SetEMAdvanced.restype = c_uint
SetEMAdvanced.argtypes = [c_int]
SetEMCCDGain = _stdcall_libraries['ATMCD32D'].SetEMCCDGain
SetEMCCDGain.restype = c_uint
SetEMCCDGain.argtypes = [c_int]
SetEMGainMode = _stdcall_libraries['ATMCD32D'].SetEMGainMode
SetEMGainMode.restype = c_uint
SetEMGainMode.argtypes = [c_int]
SetEMClockCompensation = _stdcall_libraries['ATMCD32D'].SetEMClockCompensation
SetEMClockCompensation.restype = c_uint
SetEMClockCompensation.argtypes = [c_int]
SetExposureTime = _stdcall_libraries['ATMCD32D'].SetExposureTime
SetExposureTime.restype = c_uint
SetExposureTime.argtypes = [c_float]
SetFanMode = _stdcall_libraries['ATMCD32D'].SetFanMode
SetFanMode.restype = c_uint
SetFanMode.argtypes = [c_int]
SetFastExtTrigger = _stdcall_libraries['ATMCD32D'].SetFastExtTrigger
SetFastExtTrigger.restype = c_uint
SetFastExtTrigger.argtypes = [c_int]
SetFastKinetics = _stdcall_libraries['ATMCD32D'].SetFastKinetics
SetFastKinetics.restype = c_uint
SetFastKinetics.argtypes = [c_int, c_int, c_float, c_int, c_int, c_int]
SetFastKineticsEx = _stdcall_libraries['ATMCD32D'].SetFastKineticsEx
SetFastKineticsEx.restype = c_uint
SetFastKineticsEx.argtypes = [c_int, c_int, c_float, c_int, c_int, c_int, c_int]
SetFilterMode = _stdcall_libraries['ATMCD32D'].SetFilterMode
SetFilterMode.restype = c_uint
SetFilterMode.argtypes = [c_int]
SetFilterParameters = _stdcall_libraries['ATMCD32D'].SetFilterParameters
SetFilterParameters.restype = c_uint
SetFilterParameters.argtypes = [c_int, c_float, c_int, c_float, c_int, c_int]
SetFKVShiftSpeed = _stdcall_libraries['ATMCD32D'].SetFKVShiftSpeed
SetFKVShiftSpeed.restype = c_uint
SetFKVShiftSpeed.argtypes = [c_int]
SetFrameTransferMode = _stdcall_libraries['ATMCD32D'].SetFrameTransferMode
SetFrameTransferMode.restype = c_uint
SetFrameTransferMode.argtypes = [c_int]
SetFullImage = _stdcall_libraries['ATMCD32D'].SetFullImage
SetFullImage.restype = c_uint
SetFullImage.argtypes = [c_int, c_int]
SetFVBHBin = _stdcall_libraries['ATMCD32D'].SetFVBHBin
SetFVBHBin.restype = c_uint
SetFVBHBin.argtypes = [c_int]
SetGain = _stdcall_libraries['ATMCD32D'].SetGain
SetGain.restype = c_uint
SetGain.argtypes = [c_int]
SetGate = _stdcall_libraries['ATMCD32D'].SetGate
SetGate.restype = c_uint
SetGate.argtypes = [c_float, c_float, c_float]
SetGateMode = _stdcall_libraries['ATMCD32D'].SetGateMode
SetGateMode.restype = c_uint
SetGateMode.argtypes = [c_int]
SetHighCapacity = _stdcall_libraries['ATMCD32D'].SetHighCapacity
SetHighCapacity.restype = c_uint
SetHighCapacity.argtypes = [c_int]
SetHorizontalSpeed = _stdcall_libraries['ATMCD32D'].SetHorizontalSpeed
SetHorizontalSpeed.restype = c_uint
SetHorizontalSpeed.argtypes = [c_int]
SetHSSpeed = _stdcall_libraries['ATMCD32D'].SetHSSpeed
SetHSSpeed.restype = c_uint
SetHSSpeed.argtypes = [c_int, c_int]
SetImage = _stdcall_libraries['ATMCD32D'].SetImage
SetImage.restype = c_uint
SetImage.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int]
SetKineticCycleTime = _stdcall_libraries['ATMCD32D'].SetKineticCycleTime
SetKineticCycleTime.restype = c_uint
SetKineticCycleTime.argtypes = [c_float]
SetMCPGating = _stdcall_libraries['ATMCD32D'].SetMCPGating
SetMCPGating.restype = c_uint
SetMCPGating.argtypes = [c_int]
SetMessageWindow = _stdcall_libraries['ATMCD32D'].SetMessageWindow
SetMessageWindow.restype = c_uint
SetMessageWindow.argtypes = [HWND]
SetMultiTrack = _stdcall_libraries['ATMCD32D'].SetMultiTrack
SetMultiTrack.restype = c_uint
SetMultiTrack.argtypes = [c_int, c_int, c_int, POINTER(c_int), POINTER(c_int)]
SetMultiTrackHBin = _stdcall_libraries['ATMCD32D'].SetMultiTrackHBin
SetMultiTrackHBin.restype = c_uint
SetMultiTrackHBin.argtypes = [c_int]
SetNextAddress = _stdcall_libraries['ATMCD32D'].SetNextAddress
SetNextAddress.restype = c_uint
SetNextAddress.argtypes = [POINTER(c_long), c_long, c_long, c_long, c_long]
#SetNextAddress16 = _stdcall_libraries['ATMCD32D'].SetNextAddress16
#SetNextAddress16.restype = c_uint
#SetNextAddress16.argtypes = [POINTER(c_long), c_long, c_long, c_long, c_long]
SetNumberAccumulations = _stdcall_libraries['ATMCD32D'].SetNumberAccumulations
SetNumberAccumulations.restype = c_uint
SetNumberAccumulations.argtypes = [c_int]
SetNumberKinetics = _stdcall_libraries['ATMCD32D'].SetNumberKinetics
SetNumberKinetics.restype = c_uint
SetNumberKinetics.argtypes = [c_int]
SetOutputAmplifier = _stdcall_libraries['ATMCD32D'].SetOutputAmplifier
SetOutputAmplifier.restype = c_uint
SetOutputAmplifier.argtypes = [c_int]
SetPhotonCounting = _stdcall_libraries['ATMCD32D'].SetPhotonCounting
SetPhotonCounting.restype = c_uint
SetPhotonCounting.argtypes = [c_int]
SetPhotonCountingThreshold = _stdcall_libraries['ATMCD32D'].SetPhotonCountingThreshold
SetPhotonCountingThreshold.restype = c_uint
SetPhotonCountingThreshold.argtypes = [c_long, c_long]
SetPixelMode = _stdcall_libraries['ATMCD32D'].SetPixelMode
SetPixelMode.restype = c_uint
SetPixelMode.argtypes = [c_int, c_int]
SetPreAmpGain = _stdcall_libraries['ATMCD32D'].SetPreAmpGain
SetPreAmpGain.restype = c_uint
SetPreAmpGain.argtypes = [c_int]
SetRandomTracks = _stdcall_libraries['ATMCD32D'].SetRandomTracks
SetRandomTracks.restype = c_uint
SetRandomTracks.argtypes = [c_int, POINTER(c_int)]
SetReadMode = _stdcall_libraries['ATMCD32D'].SetReadMode
SetReadMode.restype = c_uint
SetReadMode.argtypes = [c_int]
SetRegisterDump = _stdcall_libraries['ATMCD32D'].SetRegisterDump
SetRegisterDump.restype = c_uint
SetRegisterDump.argtypes = [c_int]
#SetSaturationEvent = _stdcall_libraries['ATMCD32D'].SetSaturationEvent
#SetSaturationEvent.restype = c_uint
#SetSaturationEvent.argtypes = [HANDLE]
SetShutter = _stdcall_libraries['ATMCD32D'].SetShutter
SetShutter.restype = c_uint
SetShutter.argtypes = [c_int, c_int, c_int, c_int]
SetShutterEx = _stdcall_libraries['ATMCD32D'].SetShutterEx
SetShutterEx.restype = c_uint
SetShutterEx.argtypes = [c_int, c_int, c_int, c_int, c_int]
SetSifComment = _stdcall_libraries['ATMCD32D'].SetSifComment
SetSifComment.restype = c_uint
SetSifComment.argtypes = [STRING]
SetSingleTrack = _stdcall_libraries['ATMCD32D'].SetSingleTrack
SetSingleTrack.restype = c_uint
SetSingleTrack.argtypes = [c_int, c_int]
SetSingleTrackHBin = _stdcall_libraries['ATMCD32D'].SetSingleTrackHBin
SetSingleTrackHBin.restype = c_uint
SetSingleTrackHBin.argtypes = [c_int]
SetSpool = _stdcall_libraries['ATMCD32D'].SetSpool
SetSpool.restype = c_uint
SetSpool.argtypes = [c_int, c_int, STRING, c_int]
SetStorageMode = _stdcall_libraries['ATMCD32D'].SetStorageMode
SetStorageMode.restype = c_uint
SetStorageMode.argtypes = [c_long]
SetTemperature = _stdcall_libraries['ATMCD32D'].SetTemperature
SetTemperature.restype = c_uint
SetTemperature.argtypes = [c_int]
SetTriggerMode = _stdcall_libraries['ATMCD32D'].SetTriggerMode
SetTriggerMode.restype = c_uint
SetTriggerMode.argtypes = [c_int]
#SetUserEvent = _stdcall_libraries['ATMCD32D'].SetUserEvent
#SetUserEvent.restype = c_uint
#SetUserEvent.argtypes = [HANDLE]
SetUSGenomics = _stdcall_libraries['ATMCD32D'].SetUSGenomics
SetUSGenomics.restype = c_uint
SetUSGenomics.argtypes = [c_long, c_long]
SetVerticalRowBuffer = _stdcall_libraries['ATMCD32D'].SetVerticalRowBuffer
SetVerticalRowBuffer.restype = c_uint
SetVerticalRowBuffer.argtypes = [c_int]
SetVerticalSpeed = _stdcall_libraries['ATMCD32D'].SetVerticalSpeed
SetVerticalSpeed.restype = c_uint
SetVerticalSpeed.argtypes = [c_int]
SetVirtualChip = _stdcall_libraries['ATMCD32D'].SetVirtualChip
SetVirtualChip.restype = c_uint
SetVirtualChip.argtypes = [c_int]
SetVSAmplitude = _stdcall_libraries['ATMCD32D'].SetVSAmplitude
SetVSAmplitude.restype = c_uint
SetVSAmplitude.argtypes = [c_int]
SetVSSpeed = _stdcall_libraries['ATMCD32D'].SetVSSpeed
SetVSSpeed.restype = c_uint
SetVSSpeed.argtypes = [c_int]
ShutDown = _stdcall_libraries['ATMCD32D'].ShutDown
ShutDown.restype = c_uint
ShutDown.argtypes = []
StartAcquisition = _stdcall_libraries['ATMCD32D'].StartAcquisition
StartAcquisition.restype = c_uint
StartAcquisition.argtypes = []
#UnMapPhysicalAddress = _stdcall_libraries['ATMCD32D'].UnMapPhysicalAddress
#UnMapPhysicalAddress.restype = c_uint
#UnMapPhysicalAddress.argtypes = []
WaitForAcquisition = _stdcall_libraries['ATMCD32D'].WaitForAcquisition
WaitForAcquisition.restype = c_uint
WaitForAcquisition.argtypes = []
WaitForAcquisitionTimeOut = _stdcall_libraries['ATMCD32D'].WaitForAcquisitionTimeOut
WaitForAcquisitionTimeOut.restype = c_uint
WaitForAcquisitionTimeOut.argtypes = [c_int]
WaitForAcquisitionByHandle = _stdcall_libraries['ATMCD32D'].WaitForAcquisitionByHandle
WaitForAcquisitionByHandle.restype = c_uint
WaitForAcquisitionByHandle.argtypes = [c_long]
WaitForAcquisitionByHandleTimeOut = _stdcall_libraries['ATMCD32D'].WaitForAcquisitionByHandleTimeOut
WaitForAcquisitionByHandleTimeOut.restype = c_uint
WaitForAcquisitionByHandleTimeOut.argtypes = [c_long, c_int]
AC_ACQMODE_ACCUMULATE = 4 # Variable c_int
AC_READMODE_FVB = 8 # Variable c_int
AC_TRIGGERMODE_INTERNAL = 1 # Variable c_int
DRV_TEMPERATURE_NOT_STABILIZED = 20035 # Variable c_int
AC_ACQMODE_FRAMETRANSFER = 16 # Variable c_int
AC_FEATURES_POLLING = 1 # Variable c_int
AC_SETFUNCTION_TEMPERATURE = 4 # Variable c_int
DRV_ERROR_BOARDTEST = 20012 # Variable c_int
DRV_USB_INTERRUPT_ENDPOINT_ERROR = 20093 # Variable c_int
DRV_SPOOLSETUPERROR = 20027 # Variable c_int
AC_SETFUNCTION_PREAMPGAIN = 512 # Variable c_int
DRV_P6INVALID = 20077 # Variable c_int
AC_SETFUNCTION_HIGHCAPACITY = 128 # Variable c_int
DRV_I2CERRORS = 20080 # Variable c_int
DRV_INIERROR = 20070 # Variable c_int
AC_FEATURES_EXTERNAL_I2C = 32 # Variable c_int
DRV_USBERROR = 20089 # Variable c_int
DRV_VXDNOTINSTALLED = 20003 # Variable c_int
AC_CAMERATYPE_USBISTAR = 10 # Variable c_int
DRV_P5INVALID = 20076 # Variable c_int
AC_CAMERATYPE_PDA = 0 # Variable c_int
DRV_TEMPERATURE_OUT_RANGE = 20038 # Variable c_int
AC_SETFUNCTION_ICCDGAIN = 8 # Variable c_int
AC_FEATURES_EVENTS = 2 # Variable c_int
DRV_VRMVERSIONERROR = 20091 # Variable c_int
AC_CAMERATYPE_IDUS = 7 # Variable c_int
AC_SETFUNCTION_EMCCDGAIN = 16 # Variable c_int
DRV_PROC_UNKONWN_INSTRUCTION = 20020 # Variable c_int
DRV_P3INVALID = 20068 # Variable c_int
DRV_ERROR_MDL = 20117 # Variable c_int
DRV_OW_NOT_INITIALIZED = 20154 # Variable c_int
DRV_TEMPCYCLE = 20074 # Variable c_int
DRV_INVALID_TRIGGER_MODE = 20095 # Variable c_int
DRV_RANDOM_TRACK_ERROR = 20094 # Variable c_int
DRV_ACQ_DOWNFIFO_FULL = 20019 # Variable c_int
AC_CAMERATYPE_CCD = 4 # Variable c_int
DRV_P2INVALID = 20067 # Variable c_int
AC_FEATURES_SPOOLING = 4 # Variable c_int
DRV_ERROR_PAGELOCK = 20010 # Variable c_int
AC_SETFUNCTION_GAIN = 8 # Variable c_int
DRV_UNKNOWN_FUNCTION = 20007 # Variable c_int
DRV_TEMPERATURE_DRIFT = 20040 # Variable c_int
DRV_DATATYPE = 20064 # Variable c_int
DRV_TEMP_NOT_STABILIZED = 20035 # Variable c_int
DRV_TEMP_DRIFT = 20040 # Variable c_int
DRV_ERROR_MAP = 20115 # Variable c_int
AC_SETFUNCTION_HREADOUT = 2 # Variable c_int
DRV_TEMP_OUT_RANGE = 20038 # Variable c_int
DRV_ERROR_FILELOAD = 20006 # Variable c_int
DRV_ERROR_CODES = 20001 # Variable c_int
AC_ACQMODE_VIDEO = 2 # Variable c_int
DRV_NO_NEW_DATA = 20024 # Variable c_int
DRV_I2CDEVNOTFOUND = 20081 # Variable c_int
AC_EMGAIN_REAL12 = 8 # Variable c_int
DRV_INVALID_AUX = 20050 # Variable c_int
DRV_IOCERROR = 20090 # Variable c_int
DRV_ERROR_CHECK_SUM = 20005 # Variable c_int
AC_FEATURES_SHUTTEREX = 16 # Variable c_int
DRV_ERROR_VXD_INIT = 20008 # Variable c_int
DRV_COF_NOTLOADED = 20051 # Variable c_int
AC_EMGAIN_8BIT = 1 # Variable c_int
AC_PIXELMODE_32BIT = 8 # Variable c_int
AC_CAMERATYPE_NEWTON = 8 # Variable c_int
AC_PIXELMODE_14BIT = 2 # Variable c_int
AC_GETFUNCTION_DETECTORSIZE = 8 # Variable c_int
DRV_ERROR_NOHANDLE = 20121 # Variable c_int
DRV_TEMPERATURE_NOT_REACHED = 20037 # Variable c_int
AC_PIXELMODE_16BIT = 4 # Variable c_int
AC_READMODE_MULTITRACK = 16 # Variable c_int
AC_CAMERATYPE_VIDEO = 6 # Variable c_int
DRV_OW_CMD_FAIL = 20150 # Variable c_int
DRV_TEMPERATURE_OFF = 20034 # Variable c_int
DRV_INVALID_FILTER = 20079 # Variable c_int
DRV_ERROR_UNMAP = 20116 # Variable c_int
DRV_P4INVALID = 20069 # Variable c_int
DRV_I2CTIMEOUT = 20082 # Variable c_int
AC_CAMERATYPE_ICCD = 2 # Variable c_int
AC_ACQMODE_SINGLE = 1 # Variable c_int
DRV_ERROR_BUFFSIZE = 20119 # Variable c_int
DRV_GPIBERROR = 20054 # Variable c_int
AC_CAMERATYPE_ISTAR = 5 # Variable c_int
AC_TRIGGERMODE_EXTERNAL = 2 # Variable c_int
DRV_ACQUISITION_ERRORS = 20017 # Variable c_int
DRV_IDLE = 20073 # Variable c_int
DRV_P7INVALID = 20083 # Variable c_int
AC_SETFUNCTION_VREADOUT = 1 # Variable c_int
DRV_ERROR_UNMDL = 20118 # Variable c_int
DRV_GENERAL_ERRORS = 20049 # Variable c_int
AC_EMGAIN_LINEAR12 = 4 # Variable c_int
AC_GETFUNCTION_TEMPERATURE = 1 # Variable c_int
AC_GETFUNCTION_ICCDGAIN = 16 # Variable c_int
DRV_ERROR_NOCAMERA = 20990 # Variable c_int
DRV_COFERROR = 20071 # Variable c_int
AC_GETFUNCTION_GAIN = 16 # Variable c_int
AC_GETFUNCTION_EMCCDGAIN = 32 # Variable c_int
AC_FEATURES_SHUTTER = 8 # Variable c_int
AC_READMODE_SUBIMAGE = 2 # Variable c_int
DRV_FILESIZELIMITERROR = 20028 # Variable c_int
AC_PIXELMODE_CMY = 131072 # Variable c_int
AC_READMODE_SINGLETRACK = 4 # Variable c_int
DRV_SUCCESS = 20002 # Variable c_int
AC_CAMERATYPE_LUCA = 11 # Variable c_int
DRV_LOAD_FIRMWARE_ERROR = 20096 # Variable c_int
AC_EMGAIN_12BIT = 2 # Variable c_int
AC_READMODE_RANDOMTRACK = 32 # Variable c_int
DRV_ACCUM_TIME_NOT_MET = 20023 # Variable c_int
AC_READMODE_FULLIMAGE = 1 # Variable c_int
DRV_KINETIC_TIME_NOT_MET = 20022 # Variable c_int
AC_CAMERATYPE_EMCCD = 3 # Variable c_int
DRV_ERROR_SCAN = 20004 # Variable c_int
DRV_NOT_SUPPORTED = 20991 # Variable c_int
DRV_TEMP_OFF = 20034 # Variable c_int
DRV_DRIVER_ERRORS = 20065 # Variable c_int
DRV_ERROR_ADDRESS = 20009 # Variable c_int
DRV_OW_ERROR_SLAVE_NUM = 20155 # Variable c_int
AC_PIXELMODE_MONO = 0 # Variable c_int
AC_PIXELMODE_RGB = 65536 # Variable c_int
DRV_GATING_NOT_AVAILABLE = 20130 # Variable c_int
AC_GETFUNCTION_TARGETTEMPERATURE = 2 # Variable c_int
DRV_INVALID_MODE = 20078 # Variable c_int
DRV_ERROR_UP_FIFO = 20014 # Variable c_int
AC_GETFUNCTION_TEMPERATURERANGE = 4 # Variable c_int
AC_CAMERATYPE_SURCAM = 9 # Variable c_int
DRV_ACQ_BUFFER = 20018 # Variable c_int
AC_ACQMODE_KINETIC = 8 # Variable c_int
DRV_SPOOLERROR = 20026 # Variable c_int
DRV_TEMP_NOT_REACHED = 20037 # Variable c_int
DRV_ERROR_PAGEUNLOCK = 20011 # Variable c_int
DRV_TEMP_CODES = 20033 # Variable c_int
AC_PIXELMODE_8BIT = 1 # Variable c_int
DRV_P1INVALID = 20066 # Variable c_int
AC_SETFUNCTION_BASELINECLAMP = 32 # Variable c_int
DRV_TEMP_STABILIZED = 20036 # Variable c_int
DRV_TEMPERATURE_STABILIZED = 20036 # Variable c_int
DRV_OWCMD_NOT_AVAILABLE = 20152 # Variable c_int
DRV_EEPROMVERSIONERROR = 20055 # Variable c_int
DRV_NOT_INITIALIZED = 20075 # Variable c_int
DRV_FPGAPROG = 20052 # Variable c_int
AC_SETFUNCTION_VSAMPLITUDE = 64 # Variable c_int
DRV_ACQUIRING = 20072 # Variable c_int
AC_ACQMODE_FASTKINETICS = 32 # Variable c_int
AC_CAMERATYPE_IXON = 1 # Variable c_int
DRV_TEMPERATURE_NOT_SUPPORTED = 20039 # Variable c_int
DRV_OW_NO_SLAVES = 20153 # Variable c_int
DRV_OWMEMORY_BAD_ADDR = 20151 # Variable c_int
DRV_ERROR_PATTERN = 20015 # Variable c_int
DRV_TEMPERATURE_CODES = 20033 # Variable c_int
AC_SETFUNCTION_CROPMODE = 1024 # Variable c_int
DRV_NOT_AVAILABLE = 20992 # Variable c_int
DRV_FLEXERROR = 20053 # Variable c_int
AC_SETFUNCTION_BASELINEOFFSET = 256 # Variable c_int
DRV_TEMP_NOT_SUPPORTED = 20039 # Variable c_int
DRV_FPGA_VOLTAGE_ERROR = 20131 # Variable c_int
DRV_ILLEGAL_OP_CODE = 20021 # Variable c_int
DRV_ERROR_ACK = 20013 # Variable c_int
AC_GETFUNCTION_BASELINECLAMP = 32768 # Variable c_int

__all__ = ['SetEMClockCompensation', 'GetImages', 'SetSifComment',
           'AC_FEATURES_POLLING', 'SetFilterParameters',
           'GetPreAmpGain', 'DRV_DATATYPE', 'DRV_VXDNOTINSTALLED',
           'SaveAsTiff', 'DRV_VRMVERSIONERROR',
           'DRV_PROC_UNKONWN_INSTRUCTION', 'GetAmpDesc', 'SaveAsBmp',
           'SetDDGAddress', 'GetFKExposureTime', 'OWReadPIO',
           'SetHorizontalSpeed', 'SetAccumulationCycleTime',
           'SetAcqStatusEvent', 'OWWritePIORegister',
           'GetRegisterDump', 'DRV_ERROR_ADDRESS', 'GetNewFloatData',
           'DRV_TEMP_OUT_RANGE', 'GetCameraSerialNumber',
           'SetReadMode', 'DRV_IOCERROR', 'OWGetNbDevices',
           'AndorCapabilities', 'AC_CAMERATYPE_NEWTON',
           'GetFKVShiftSpeedF', 'INT', 'GetAcquisitionTimings',
           'IsInternalMechanicalShutter', 'GetEMCCDGain',
           'AC_SETFUNCTION_BASELINEOFFSET', 'SetNextAddress16',
           'DRV_ACQUISITION_ERRORS', 'ShutDown', 'SetGateMode',
           'SetPhotonCounting', 'SetHSSpeed',
           'AC_GETFUNCTION_ICCDGAIN', 'DRV_COFERROR',
           'GetNumberVSSpeeds', 'SetExposureTime',
           'AC_CAMERATYPE_LUCA', 'GetPhysicalDMAAddress',
           'DRV_LOAD_FIRMWARE_ERROR', 'Merge', 'SetUserEvent',
           'AC_READMODE_FULLIMAGE', 'DRV_ERROR_SCAN',
           'DRV_DRIVER_ERRORS', 'InitializeDevice',
           'DRV_TEMPERATURE_OUT_RANGE', 'DRV_INVALID_MODE',
           'DRV_ERROR_UP_FIFO', 'AC_GETFUNCTION_TEMPERATURERANGE',
           'OWSetPIO', 'SetFPDP', 'DRV_SPOOLERROR',
           'DRV_TEMP_NOT_REACHED', 'GPIBSend', 'FreeInternalMemory',
           'AC_PIXELMODE_8BIT', 'DRV_P1INVALID', 'SetTriggerMode',
           'DRV_TEMPERATURE_STABILIZED', 'LPINT',
           'DRV_EEPROMVERSIONERROR', 'SetPixelMode',
           'AC_CAMERATYPE_VIDEO', 'SetGain', 'DRV_OW_NO_SLAVES',
           'DRV_OWMEMORY_BAD_ADDR', 'SetDDGTimes', 'SetEMAdvanced',
           'GetFilterMode', 'SetADChannel', 'I2CReset',
           'SetSaturationEvent', 'DRV_USB_INTERRUPT_ENDPOINT_ERROR',
           'I2CRead', 'SetMultiTrackHBin', 'DRV_USBERROR',
           'SetUSGenomics', 'GetNumberPreAmpGains',
           'GetFastestRecommendedVSSpeed', 'AC_FEATURES_EVENTS',
           'SetBaselineClamp', 'DRV_P3INVALID', 'SetDelayGenerator',
           'DRV_ERROR_MDL', 'SetFastExtTrigger',
           'DRV_OW_NOT_INITIALIZED', 'SetCropMode', 'InAuxPort',
           'CoolerON', 'LPVOID', 'DRV_P2INVALID',
           'SetDDGVariableGateStep', 'SetMCPGating',
           'SaveAsCommentedSif', 'AC_SETFUNCTION_GAIN', 'GetVSSpeed',
           'SetKineticCycleTime', 'DRV_ERROR_FILELOAD',
           'DRV_NOT_INITIALIZED', 'SaveEEPROMToFile', 'GetDDGPulse',
           'PBYTE', 'GetCurrentCamera', 'DRV_ERROR_VXD_INIT',
           'SetSingleTrack', 'SetPhotonCountingThreshold',
           'GetStatus', 'DRV_COF_NOTLOADED', 'AC_EMGAIN_8BIT',
           'SetFastKineticsEx', 'GetImagesPerDMA',
           'DRV_TEMPERATURE_NOT_REACHED', 'AC_ACQMODE_FASTKINETICS',
           'DRV_OW_CMD_FAIL', 'GetCameraHandle', 'DRV_INVALID_FILTER',
           'SetDDGGateStep', 'SetAcquisitionMode', 'DRV_I2CTIMEOUT',
           'SetDDGInsertionDelay', 'LPWORD', 'AC_CAMERATYPE_ISTAR',
           'DRV_IDLE', 'AC_SETFUNCTION_VREADOUT', 'AC_PIXELMODE_MONO',
           'DRV_GENERAL_ERRORS', 'WaitForAcquisitionByHandleTimeOut',
           'AC_EMGAIN_LINEAR12', 'AC_GETFUNCTION_TEMPERATURE',
           'SetHighCapacity', 'PrepareAcquisition', 'OWPulsePIO',
           'GetNewData16', 'GetFKVShiftSpeed', 'I2CBurstRead',
           'OWSetMemPageRegister', 'CoolerOFF', 'SetEMCCDGain',
           'DRV_ACQ_BUFFER', 'SetDataType', 'I2CBurstWrite',
           'SetBackground', 'GetMinimumImageLength',
           'GetControllerCardModel', 'AC_EMGAIN_12BIT',
           'DRV_ACCUM_TIME_NOT_MET', 'GetCameraEventStatus',
           'GetHSSpeed', 'GetAmpMaxSpeed', 'AC_CAMERATYPE_EMCCD',
           'ColorDemosaicInfo', 'GetNumberHSSpeeds', 'GetNewData',
           'SetImage', 'GetAcquiredData', 'IdAndorDll',
           'SetDMAParameters', 'AC_GETFUNCTION_TARGETTEMPERATURE',
           'LPLONG', 'LPBOOL', 'SetVerticalSpeed',
           'AC_ACQMODE_KINETIC', 'DRV_TEMP_STABILIZED',
           'GetNumberNewImages', 'OWWriteMem', 'SetRegisterDump',
           'SetShutterEx', 'GetAcquiredFloatData', 'GetHVflag',
           'OWInit', 'GetTemperatureStatus', 'DRV_FILESIZELIMITERROR',
           'DRV_TEMPERATURE_CODES', 'AC_EMGAIN_REAL12', 'FLOAT',
           'GetTemperatureRange', 'GetID', 'CancelWait',
           'DRV_TEMP_NOT_SUPPORTED', 'PWORD', 'DRV_ILLEGAL_OP_CODE',
           'SetDDGIOCFrequency', 'AC_ACQMODE_ACCUMULATE',
           'AC_TRIGGERMODE_INTERNAL',
           'DRV_TEMPERATURE_NOT_STABILIZED',
           'AC_ACQMODE_FRAMETRANSFER', 'DRV_SPOOLSETUPERROR',
           'AC_SETFUNCTION_PREAMPGAIN', 'DRV_I2CERRORS',
           'DRV_INIERROR', 'AC_FEATURES_EXTERNAL_I2C',
           'GetNumberHorizontalSpeeds', 'SetGate', 'PUCHAR',
           'AC_READMODE_FVB', 'SaveToClipBoard', 'GetVerticalSpeed',
           'AC_CAMERATYPE_IDUS', 'AC_SETFUNCTION_EMCCDGAIN',
           'GetAcquiredData16', 'GetAvailableCameras',
           'DRV_INVALID_TRIGGER_MODE', 'OutAuxPort',
           'DRV_ACQ_DOWNFIFO_FULL', 'AC_CAMERATYPE_CCD',
           'WaitForAcquisition', 'AC_FEATURES_SPOOLING',
           'SetBaselineOffset', 'USHORT', 'DRV_ERROR_PAGELOCK',
           'DRV_UNKNOWN_FUNCTION', 'DRV_TEMPERATURE_DRIFT',
           'AC_SETFUNCTION_TEMPERATURE', 'DRV_TEMP_DRIFT',
           'GetTotalNumberImagesAcquired', 'AC_SETFUNCTION_HREADOUT',
           'DRV_ERROR_CODES', 'AC_ACQMODE_VIDEO', 'PUINT',
           'AbortAcquisition', 'DRV_INVALID_AUX', 'SetDDGIOC',
           'AC_FEATURES_SHUTTEREX', 'DRV_RANDOM_TRACK_ERROR',
           'SetFrameTransferMode', 'AC_PIXELMODE_32BIT',
           'SetDDGIntelligate', 'SetMessageWindow', 'SetPreAmpGain',
           'GetHeadModel', 'PSZ', 'DRV_GPIBERROR', 'DRV_P7INVALID',
           'AC_PIXELMODE_14BIT', 'AC_TRIGGERMODE_EXTERNAL',
           'DRV_ERROR_NOCAMERA', 'SetVSAmplitude', 'SaveAsEDF',
           'GetMostRecentColorImage16', 'AC_FEATURES_SHUTTER',
           'AC_READMODE_SUBIMAGE', 'AC_READMODE_SINGLETRACK',
           'SetComplexImage', 'SetDDGGain', 'SetFilterMode',
           'LPDWORD', 'WaitForAcquisitionTimeOut',
           'GetHorizontalSpeed', 'DRV_GATING_NOT_AVAILABLE',
           'GetTemperatureF', 'DRV_TEMPERATURE_OFF',
           'SetOutputAmplifier', 'GetEMGainRange',
           'AC_CAMERATYPE_SURCAM', 'GetMostRecentImage',
           'AC_SETFUNCTION_VSAMPLITUDE', 'DRV_ERROR_PAGEUNLOCK',
           'AC_CAMERATYPE_USBISTAR', 'DRV_OWCMD_NOT_AVAILABLE',
           'DRV_TEMP_NOT_STABILIZED', 'LPCVOID', 'SetFVBHBin',
           'GetTemperature', 'DRV_ACQUIRING', 'AC_CAMERATYPE_IXON',
           'SetNextAddress', 'SetSpool', 'AC_SETFUNCTION_CROPMODE',
           'DRV_NOT_AVAILABLE', 'SetFKVShiftSpeed', 'GetMCPGain',
           'DRV_FLEXERROR', 'GetImages16', 'GetNumberDevices',
           'DRV_ERROR_ACK', 'PINT', 'SetFullImage', 'SetVirtualChip',
           'OWGetDeviceID', 'PDWORD', 'SetPCIMode',
           'SetSingleTrackHBin', 'GetOldestImage16', 'OWReadMem',
           'DRV_ERROR_BOARDTEST', 'SetRandomTracks', 'DemosaicImage',
           'GetSizeOfCircularBuffer', 'AC_SETFUNCTION_HIGHCAPACITY',
           'GetAllDMAData', 'GetCameraInformation', 'GetBackground',
           'GetDICameraInfo', 'Initialize', 'PUSHORT',
           'DRV_P5INVALID', 'AC_CAMERATYPE_PDA',
           'AC_SETFUNCTION_ICCDGAIN', 'GetCapabilities',
           'OWReadPIORegister', 'OWLockMemPageRegister',
           'DRV_TEMPCYCLE', 'SetNumberKinetics', 'GPIBReceive',
           'SetFastKinetics', 'StartAcquisition',
           'OWSetTransmissionMode', 'SetDDGTriggerMode',
           'GetMCPVoltage', 'GetFIFOUsage', 'DRV_ERROR_MAP',
           'SetAcquisitionType', 'DRV_NO_NEW_DATA',
           'DRV_I2CDEVNOTFOUND', 'DRV_ERROR_CHECK_SUM',
           'SetCustomTrackHBin', 'DRV_FPGAPROG',
           'GetNumberVerticalSpeeds', 'SetVSSpeed',
           'GetHardwareVersion', 'PBOOL', 'UCHAR',
           'WaitForAcquisitionByHandle', 'GetNumberVSAmplitudes',
           'AC_GETFUNCTION_DETECTORSIZE', 'DRV_ERROR_NOHANDLE',
           'AC_PIXELMODE_16BIT', 'AC_READMODE_MULTITRACK',
           'SetFanMode', 'COLORDEMOSAICINFO',
           'OWResetPIOActivityLatch', 'PULONG', 'GetMaximumExposure',
           'DRV_ERROR_UNMAP', 'DRV_P4INVALID', 'AC_CAMERATYPE_ICCD',
           'AC_ACQMODE_SINGLE', 'DRV_ERROR_BUFFSIZE',
           'GetAcquisitionProgress', 'LPBYTE', 'DRV_ERROR_UNMDL',
           'DRV_P6INVALID', 'SetTemperature', 'GetOldestImage',
           'AC_GETFUNCTION_GAIN', 'SetVerticalRowBuffer',
           'AC_GETFUNCTION_EMCCDGAIN', 'AC_PIXELMODE_CMY',
           'GetMostRecentImage16', 'DRV_SUCCESS',
           'AC_SETFUNCTION_BASELINECLAMP', 'ANDORCAPS',
           'SetNumberAccumulations', 'AC_READMODE_RANDOMTRACK',
           'DRV_KINETIC_TIME_NOT_MET', 'DRV_NOT_SUPPORTED',
           'GetPixelSize', 'SetEMGainMode', 'SaveAsFITS',
           'GetMaximumBinning', 'GetDDGIOCPulses',
           'DRV_OW_ERROR_SLAVE_NUM', 'GetSlotBusDeviceFunction',
           'GetNumberADChannels', 'AC_PIXELMODE_RGB', 'I2CWrite',
           'SetMultiTrack', 'SetCurrentCamera',
           'UnMapPhysicalAddress', 'GetBitDepth', 'DRV_TEMP_CODES',
           'SetCoolerMode', 'GetNumberFKVShiftSpeeds', 'GetNumberAmp',
           'GetSpoolProgress', 'SelectDevice',
           'DRV_TEMPERATURE_NOT_SUPPORTED', 'GetDetector',
           'GetSoftwareVersion', 'DRV_ERROR_PATTERN',
           'SetDriverEvent', 'IsPreAmpGainAvailable',
           'SetStorageMode', 'DRV_TEMP_OFF', 'DRV_FPGA_VOLTAGE_ERROR',
           'PFLOAT', 'SetShutter', 'SaveAsSif',
           'AC_GETFUNCTION_BASELINECLAMP', 'GetBaselineClamp']


errorCodes = {}
for symb in __all__:
    if (symb[0:4] == 'DRV_'):
        errorCodes[eval(symb)] = symb
