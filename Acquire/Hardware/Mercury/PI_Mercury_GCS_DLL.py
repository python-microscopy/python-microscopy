from ctypes import *

STRING = c_char_p
_stdcall_libraries = {}
_stdcall_libraries['PI_Mercury_GCS_DLL.dll'] = WinDLL('PI_Mercury_GCS_DLL.dll')
from ctypes.wintypes import BOOL


InterfaceSetupDlg = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_InterfaceSetupDlg
InterfaceSetupDlg.restype = c_int
InterfaceSetupDlg.argtypes = [STRING]
ConnectRS232 = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_ConnectRS232
ConnectRS232.restype = c_int
ConnectRS232.argtypes = [c_int, c_int]
IsConnected = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsConnected
IsConnected.restype = BOOL
IsConnected.argtypes = [c_int]
CloseConnection = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_CloseConnection
CloseConnection.restype = None
CloseConnection.argtypes = [c_int]
GetError = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetError
GetError.restype = c_int
GetError.argtypes = [c_int]
SetErrorCheck = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_SetErrorCheck
SetErrorCheck.restype = BOOL
SetErrorCheck.argtypes = [c_int, BOOL]
TranslateError = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_TranslateError
TranslateError.restype = BOOL
TranslateError.argtypes = [c_int, STRING, c_int]
qERR = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qERR
qERR.restype = BOOL
qERR.argtypes = [c_int, POINTER(c_int)]
qIDN = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qIDN
qIDN.restype = BOOL
qIDN.argtypes = [c_int, STRING, c_int]
qVER = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qVER
qVER.restype = BOOL
qVER.argtypes = [c_int, STRING, c_int]
INI = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_INI
INI.restype = BOOL
INI.argtypes = [c_int, STRING]
MOV = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MOV
MOV.restype = BOOL
MOV.argtypes = [c_int, STRING, POINTER(c_double)]
qMOV = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qMOV
qMOV.restype = BOOL
qMOV.argtypes = [c_int, STRING, POINTER(c_double)]
MVR = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MVR
MVR.restype = BOOL
MVR.argtypes = [c_int, STRING, POINTER(c_double)]
IsMoving = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsMoving
IsMoving.restype = BOOL
IsMoving.argtypes = [c_int, STRING, POINTER(BOOL)]
qONT = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qONT
qONT.restype = BOOL
qONT.argtypes = [c_int, STRING, POINTER(BOOL)]
DFF = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_DFF
DFF.restype = BOOL
DFF.argtypes = [c_int, STRING, POINTER(c_double)]
qDFF = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qDFF
qDFF.restype = BOOL
qDFF.argtypes = [c_int, STRING, POINTER(c_double)]
DFH = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_DFH
DFH.restype = BOOL
DFH.argtypes = [c_int, STRING]
qDFH = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qDFH
qDFH.restype = BOOL
qDFH.argtypes = [c_int, STRING, POINTER(c_double)]
GOH = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GOH
GOH.restype = BOOL
GOH.argtypes = [c_int, STRING]
qPOS = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qPOS
qPOS.restype = BOOL
qPOS.argtypes = [c_int, STRING, POINTER(c_double)]
POS = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_POS
POS.restype = BOOL
POS.argtypes = [c_int, STRING, POINTER(c_double)]
HLT = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_HLT
HLT.restype = BOOL
HLT.argtypes = [c_int, STRING]
STP = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_STP
STP.restype = BOOL
STP.argtypes = [c_int]
qCST = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qCST
qCST.restype = BOOL
qCST.argtypes = [c_int, STRING, STRING, c_int]
CST = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_CST
CST.restype = BOOL
CST.argtypes = [c_int, STRING, STRING]
qVST = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qVST
qVST.restype = BOOL
qVST.argtypes = [c_int, STRING, c_int]
qTVI = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTVI
qTVI.restype = BOOL
qTVI.argtypes = [c_int, STRING, c_int]
SAI = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_SAI
SAI.restype = BOOL
SAI.argtypes = [c_int, STRING, STRING]
qSAI = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qSAI
qSAI.restype = BOOL
qSAI.argtypes = [c_int, STRING, c_int]
qSAI_ALL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qSAI_ALL
qSAI_ALL.restype = BOOL
qSAI_ALL.argtypes = [c_int, STRING, c_int]
SVO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_SVO
SVO.restype = BOOL
SVO.argtypes = [c_int, STRING, POINTER(BOOL)]
qSVO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qSVO
qSVO.restype = BOOL
qSVO.argtypes = [c_int, STRING, POINTER(BOOL)]
VEL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_VEL
VEL.restype = BOOL
VEL.argtypes = [c_int, STRING, POINTER(c_double)]
qVEL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qVEL
qVEL.restype = BOOL
qVEL.argtypes = [c_int, STRING, POINTER(c_double)]
SPA = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_SPA
SPA.restype = BOOL
SPA.argtypes = [c_int, STRING, POINTER(c_int), POINTER(c_double), STRING]
qSPA = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qSPA
qSPA.restype = BOOL
qSPA.argtypes = [c_int, STRING, POINTER(c_int), POINTER(c_double), STRING, c_int]
qSRG = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qSRG
qSRG.restype = BOOL
qSRG.argtypes = [c_int, STRING, POINTER(c_int), POINTER(c_int)]
GetInputChannelNames = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetInputChannelNames
GetInputChannelNames.restype = BOOL
GetInputChannelNames.argtypes = [c_int, STRING, c_int]
GetOutputChannelNames = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetOutputChannelNames
GetOutputChannelNames.restype = BOOL
GetOutputChannelNames.argtypes = [c_int, STRING, c_int]
DIO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_DIO
DIO.restype = BOOL
DIO.argtypes = [c_int, STRING, POINTER(BOOL)]
qDIO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qDIO
qDIO.restype = BOOL
qDIO.argtypes = [c_int, STRING, POINTER(BOOL)]
qTIO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTIO
qTIO.restype = BOOL
qTIO.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
qTAC = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTAC
qTAC.restype = BOOL
qTAC.argtypes = [c_int, POINTER(c_int)]
qTAV = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTAV
qTAV.restype = BOOL
qTAV.argtypes = [c_int, c_int, POINTER(c_double)]
BRA = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_BRA
BRA.restype = BOOL
BRA.argtypes = [c_int, STRING, POINTER(BOOL)]
qBRA = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qBRA
qBRA.restype = BOOL
qBRA.argtypes = [c_int, STRING, c_int]
qHLP = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qHLP
qHLP.restype = BOOL
qHLP.argtypes = [c_int, STRING, c_int]
qHPA = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qHPA
qHPA.restype = BOOL
qHPA.argtypes = [c_long, STRING, c_long]
SendNonGCSString = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_SendNonGCSString
SendNonGCSString.restype = BOOL
SendNonGCSString.argtypes = [c_int, STRING]
ReceiveNonGCSString = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_ReceiveNonGCSString
ReceiveNonGCSString.restype = BOOL
ReceiveNonGCSString.argtypes = [c_int, STRING, c_int]
GcsCommandset = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GcsCommandset
GcsCommandset.restype = BOOL
GcsCommandset.argtypes = [c_int, STRING]
GcsGetAnswer = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GcsGetAnswer
GcsGetAnswer.restype = BOOL
GcsGetAnswer.argtypes = [c_int, STRING, c_int]
GcsGetAnswerSize = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GcsGetAnswerSize
GcsGetAnswerSize.restype = BOOL
GcsGetAnswerSize.argtypes = [c_int, POINTER(c_int)]
MNL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MNL
MNL.restype = BOOL
MNL.argtypes = [c_int, STRING]
MPL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MPL
MPL.restype = BOOL
MPL.argtypes = [c_int, STRING]
REF = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_REF
REF.restype = BOOL
REF.argtypes = [c_int, STRING]
qREF = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qREF
qREF.restype = BOOL
qREF.argtypes = [c_int, STRING, POINTER(BOOL)]
qLIM = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qLIM
qLIM.restype = BOOL
qLIM.argtypes = [c_int, STRING, POINTER(BOOL)]
IsReferencing = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsReferencing
IsReferencing.restype = BOOL
IsReferencing.argtypes = [c_int, STRING, POINTER(BOOL)]
GetRefResult = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetRefResult
GetRefResult.restype = BOOL
GetRefResult.argtypes = [c_int, STRING, POINTER(c_int)]
IsReferenceOK = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsReferenceOK
IsReferenceOK.restype = BOOL
IsReferenceOK.argtypes = [c_int, STRING, POINTER(BOOL)]
qTMN = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTMN
qTMN.restype = BOOL
qTMN.argtypes = [c_int, STRING, POINTER(c_double)]
qTMX = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTMX
qTMX.restype = BOOL
qTMX.argtypes = [c_int, STRING, POINTER(c_double)]
RON = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_RON
RON.restype = BOOL
RON.argtypes = [c_int, STRING, POINTER(BOOL)]
qRON = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qRON
qRON.restype = BOOL
qRON.argtypes = [c_int, STRING, POINTER(BOOL)]
IsRecordingMacro = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsRecordingMacro
IsRecordingMacro.restype = BOOL
IsRecordingMacro.argtypes = [c_int, POINTER(BOOL)]
IsRunningMacro = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsRunningMacro
IsRunningMacro.restype = BOOL
IsRunningMacro.argtypes = [c_int, POINTER(BOOL)]
DEL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_DEL
DEL.restype = BOOL
DEL.argtypes = [c_int, c_int]
MAC_BEG = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MAC_BEG
MAC_BEG.restype = BOOL
MAC_BEG.argtypes = [c_int, STRING]
MAC_START = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MAC_START
MAC_START.restype = BOOL
MAC_START.argtypes = [c_int, STRING]
MAC_NSTART = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MAC_NSTART
MAC_NSTART.restype = BOOL
MAC_NSTART.argtypes = [c_int, STRING, c_int]
MAC_END = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MAC_END
MAC_END.restype = BOOL
MAC_END.argtypes = [c_int]
MEX = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MEX
MEX.restype = BOOL
MEX.argtypes = [c_int, STRING]
MAC_DEL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MAC_DEL
MAC_DEL.restype = BOOL
MAC_DEL.argtypes = [c_int, STRING]
qMAC = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qMAC
qMAC.restype = BOOL
qMAC.argtypes = [c_int, STRING, STRING, c_int]
WAC = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_WAC
WAC.restype = BOOL
WAC.argtypes = [c_int, STRING]
JON = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_JON
JON.restype = BOOL
JON.argtypes = [c_int, POINTER(c_int), POINTER(BOOL), c_int]
qJON = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qJON
qJON.restype = BOOL
qJON.argtypes = [c_int, POINTER(c_int), POINTER(BOOL), c_int]
qTNJ = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTNJ
qTNJ.restype = BOOL
qTNJ.argtypes = [c_int, POINTER(c_int)]
qJAX = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qJAX
qJAX.restype = BOOL
qJAX.argtypes = [c_int, POINTER(c_int), POINTER(c_int), c_int, STRING, c_int]
JDT = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_JDT
JDT.restype = BOOL
JDT.argtypes = [c_int, POINTER(c_int), POINTER(c_int), c_int]
qJBS = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qJBS
qJBS.restype = BOOL
qJBS.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(BOOL), c_int]
qJAS = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qJAS
qJAS.restype = BOOL
qJAS.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(c_double), c_int]
JLT = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_JLT
JLT.restype = BOOL
JLT.argtypes = [c_int, c_int, c_int, c_int, POINTER(c_double), c_int]
qJLT = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qJLT
qJLT.restype = BOOL
qJLT.argtypes = [c_int, POINTER(c_int), POINTER(c_int), c_int, c_int, c_int, POINTER(POINTER(c_double)), STRING, c_int]
CTO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_CTO
CTO.restype = BOOL
CTO.argtypes = [c_int, POINTER(c_long), POINTER(c_long), STRING, c_int]
qCTO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qCTO
qCTO.restype = BOOL
qCTO.argtypes = [c_int, POINTER(c_long), POINTER(c_long), STRING, c_int, c_int]
TRO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_TRO
TRO.restype = BOOL
TRO.argtypes = [c_int, POINTER(c_long), POINTER(BOOL), c_int]
qTRO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTRO
qTRO.restype = BOOL
qTRO.argtypes = [c_int, POINTER(c_long), POINTER(BOOL), c_int]
AddStage = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_AddStage
AddStage.restype = BOOL
AddStage.argtypes = [c_int, STRING]
RemoveStage = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_RemoveStage
RemoveStage.restype = BOOL
RemoveStage.argtypes = [c_int, STRING]
OpenUserStagesEditDialog = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_OpenUserStagesEditDialog
OpenUserStagesEditDialog.restype = BOOL
OpenUserStagesEditDialog.argtypes = [c_int]
OpenPiStagesEditDialog = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_OpenPiStagesEditDialog
OpenPiStagesEditDialog.restype = BOOL
OpenPiStagesEditDialog.argtypes = [c_int]
GetStatus = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetStatus
GetStatus.restype = BOOL
GetStatus.argtypes = [c_int, STRING, POINTER(c_int), POINTER(c_int)]
GetAsyncBufferIndex = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetAsyncBufferIndex
GetAsyncBufferIndex.restype = c_int
GetAsyncBufferIndex.argtypes = [c_long]
__all__ = ['qTNJ', 'GcsGetAnswerSize', 'qSRG',
           'DFF', 'qPOS', 'qJBS',
           'DFH', 'qLIM', 'MAC_BEG',
           'qDIO', 'GcsCommandset', 'MOV',
           'qERR', 'JLT', 'VEL',
           'DIO', 'CST', 'IsReferencing',
           'GetAsyncBufferIndex', 'REF',
           'SPA', 'qRON', 'JDT',
           'TranslateError', 'qCTO', 'RON',
           'qHPA', 'BRA', 'qSPA',
           'RemoveStage', 'JON',
           'ReceiveNonGCSString', 'qCST',
           'HLT', 'OpenUserStagesEditDialog',
           'MPL', 'ConnectRS232', 'qTMX',
           'qTMN', 'qTVI', 'qJAS',
           'MVR', 'SVO', 'qIDN',
           'OpenPiStagesEditDialog', 'MNL',
           'CloseConnection', 'qBRA',
           'MAC_END', 'POS',
           'GetOutputChannelNames', 'qDFH',
           'qSAI', 'IsMoving', 'qDFF',
           'MAC_NSTART', 'INI', 'qSVO',
           'qJAX', 'GetInputChannelNames',
           'qMOV', 'qONT', 'STP',
           'GcsGetAnswer', 'IsReferenceOK',
           'GetRefResult', 'qJON',
           'SendNonGCSString', 'GOH', 'qVST',
           'TRO', 'DEL', 'GetStatus',
           'qJLT', 'qREF', 'qMAC',
           'CTO', 'IsRecordingMacro',
           'IsConnected', 'InterfaceSetupDlg',
           'MEX', 'GetError', 'qTAV',
           'qTIO', 'WAC', 'MAC_START',
           'qTRO', 'qTAC', 'qHLP',
           'qSAI_ALL', 'SetErrorCheck',
           'MAC_DEL', 'qVER', 'AddStage',
           'SAI', 'IsRunningMacro', 'qVEL']
