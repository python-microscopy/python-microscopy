#!/usr/bin/python

###############
# PI_Mercury_GCS_DLL.py
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
from ctypes import *
import platform

STRING = c_char_p
_stdcall_libraries = {}

#_stdcall_libraries['PI_Mercury_GCS_DLL.dll'] = WinDLL('PI_Mercury_GCS_DLL.dll')

arch, plat = platform.architecture()

if plat.startswith('Windows'):
    if arch == '32bit':
        _stdcall_libraries['PI_Mercury_GCS_DLL.dll'] = WinDLL('PI_Mercury_GCS_DLL.dll')
    else:
        _stdcall_libraries['PI_Mercury_GCS_DLL.dll'] = WinDLL('PI_Mercury_GCS_DLL_x64.dll')
    from ctypes.wintypes import ULONG, DWORD, BOOL, BYTE, WORD, UINT, HANDLE, HWND
else:
    raise RuntimeError('Not a supported platform: %s' % plat)

from ctypes.wintypes import BOOL


InterfaceSetupDlg = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_InterfaceSetupDlg
InterfaceSetupDlg.restype = c_int
InterfaceSetupDlg.argtypes = [STRING]
InterfaceSetupDlg.argnames = ['szRegKeyName']
InterfaceSetupDlg.__doc__ = \
"""int Mercury_InterfaceSetupDlg(unknown * szRegKeyName)
PI_Mercury_GCS_DLL.h:40"""

ConnectRS232 = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_ConnectRS232
ConnectRS232.restype = c_int
ConnectRS232.argtypes = [c_int, c_int]
ConnectRS232.argnames = ['port', 'baudrate']
ConnectRS232.__doc__ = \
"""int Mercury_ConnectRS232(int port, int baudrate)
PI_Mercury_GCS_DLL.h:42"""

IsConnected = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsConnected
IsConnected.restype = BOOL
IsConnected.argtypes = [c_int]
IsConnected.argnames = ['ID']
IsConnected.__doc__ = \
"""BOOL Mercury_IsConnected(int ID)
PI_Mercury_GCS_DLL.h:46"""

CloseConnection = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_CloseConnection
CloseConnection.restype = None
CloseConnection.argtypes = [c_int]
CloseConnection.argnames = ['ID']
CloseConnection.__doc__ = \
"""void Mercury_CloseConnection(int ID)
PI_Mercury_GCS_DLL.h:47"""

GetError = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetError
GetError.restype = c_int
GetError.argtypes = [c_int]
GetError.argnames = ['ID']
GetError.__doc__ = \
"""int Mercury_GetError(int ID)
PI_Mercury_GCS_DLL.h:48"""

SetErrorCheck = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_SetErrorCheck
SetErrorCheck.restype = BOOL
SetErrorCheck.argtypes = [c_int, BOOL]
SetErrorCheck.argnames = ['ID', 'bErrorCheck']
SetErrorCheck.__doc__ = \
"""BOOL Mercury_SetErrorCheck(int ID, BOOL bErrorCheck)
PI_Mercury_GCS_DLL.h:49"""

TranslateError = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_TranslateError
TranslateError.restype = BOOL
TranslateError.argtypes = [c_int, STRING, c_int]
TranslateError.argnames = ['errNr', 'szBuffer', 'maxlen']
TranslateError.__doc__ = \
"""BOOL Mercury_TranslateError(int errNr, char * szBuffer, int maxlen)
PI_Mercury_GCS_DLL.h:50"""

qERR = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qERR
qERR.restype = BOOL
qERR.argtypes = [c_int, POINTER(c_int)]
qERR.argnames = ['ID', 'pnError']
qERR.__doc__ = \
"""BOOL Mercury_qERR(int ID, int * pnError)
PI_Mercury_GCS_DLL.h:55"""

qIDN = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qIDN
qIDN.restype = BOOL
qIDN.argtypes = [c_int, STRING, c_int]
qIDN.argnames = ['ID', 'buffer', 'maxlen']
qIDN.__doc__ = \
"""BOOL Mercury_qIDN(int ID, char * buffer, int maxlen)
PI_Mercury_GCS_DLL.h:56"""

qVER = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qVER
qVER.restype = BOOL
qVER.argtypes = [c_int, STRING, c_int]
qVER.argnames = ['ID', 'buffer', 'maxlen']
qVER.__doc__ = \
"""BOOL Mercury_qVER(int ID, char * buffer, int maxlen)
PI_Mercury_GCS_DLL.h:57"""

INI = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_INI
INI.restype = BOOL
INI.argtypes = [c_int, STRING]
INI.argnames = ['ID', 'szAxes']
INI.__doc__ = \
"""BOOL Mercury_INI(int ID, unknown * szAxes)
PI_Mercury_GCS_DLL.h:58"""

MOV = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MOV
MOV.restype = BOOL
MOV.argtypes = [c_int, STRING, POINTER(c_double)]
MOV.argnames = ['ID', 'szAxes', 'pdValarray']
MOV.__doc__ = \
"""BOOL Mercury_MOV(int ID, unknown * szAxes, unknown * pdValarray)
PI_Mercury_GCS_DLL.h:60"""

qMOV = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qMOV
qMOV.restype = BOOL
qMOV.argtypes = [c_int, STRING, POINTER(c_double)]
qMOV.argnames = ['ID', 'szAxes', 'pdValarray']
qMOV.__doc__ = \
"""BOOL Mercury_qMOV(int ID, unknown * szAxes, double * pdValarray)
PI_Mercury_GCS_DLL.h:61"""

MVR = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MVR
MVR.restype = BOOL
MVR.argtypes = [c_int, STRING, POINTER(c_double)]
MVR.argnames = ['ID', 'szAxes', 'pdValarray']
MVR.__doc__ = \
"""BOOL Mercury_MVR(int ID, unknown * szAxes, unknown * pdValarray)
PI_Mercury_GCS_DLL.h:62"""

IsMoving = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsMoving
IsMoving.restype = BOOL
IsMoving.argtypes = [c_int, STRING, POINTER(BOOL)]
IsMoving.argnames = ['ID', 'szAxes', 'pbValarray']
IsMoving.__doc__ = \
"""BOOL Mercury_IsMoving(int ID, unknown * szAxes, BOOL * pbValarray)
PI_Mercury_GCS_DLL.h:63"""

qONT = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qONT
qONT.restype = BOOL
qONT.argtypes = [c_int, STRING, POINTER(BOOL)]
qONT.argnames = ['ID', 'szAxes', 'pbValarray']
qONT.__doc__ = \
"""BOOL Mercury_qONT(int ID, unknown * szAxes, BOOL * pbValarray)
PI_Mercury_GCS_DLL.h:64"""

DFF = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_DFF
DFF.restype = BOOL
DFF.argtypes = [c_int, STRING, POINTER(c_double)]
DFF.argnames = ['ID', 'szAxes', 'pdValarray']
DFF.__doc__ = \
"""BOOL Mercury_DFF(int ID, unknown * szAxes, unknown * pdValarray)
PI_Mercury_GCS_DLL.h:66"""

qDFF = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qDFF
qDFF.restype = BOOL
qDFF.argtypes = [c_int, STRING, POINTER(c_double)]
qDFF.argnames = ['ID', 'szAxes', 'pdValarray']
qDFF.__doc__ = \
"""BOOL Mercury_qDFF(int ID, unknown * szAxes, double * pdValarray)
PI_Mercury_GCS_DLL.h:67"""

DFH = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_DFH
DFH.restype = BOOL
DFH.argtypes = [c_int, STRING]
DFH.argnames = ['ID', 'szAxes']
DFH.__doc__ = \
"""BOOL Mercury_DFH(int ID, unknown * szAxes)
PI_Mercury_GCS_DLL.h:69"""

qDFH = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qDFH
qDFH.restype = BOOL
qDFH.argtypes = [c_int, STRING, POINTER(c_double)]
qDFH.argnames = ['ID', 'szAxes', 'pdValarray']
qDFH.__doc__ = \
"""BOOL Mercury_qDFH(int ID, unknown * szAxes, double * pdValarray)
PI_Mercury_GCS_DLL.h:70"""

GOH = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GOH
GOH.restype = BOOL
GOH.argtypes = [c_int, STRING]
GOH.argnames = ['ID', 'szAxes']
GOH.__doc__ = \
"""BOOL Mercury_GOH(int ID, unknown * szAxes)
PI_Mercury_GCS_DLL.h:71"""

qPOS = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qPOS
qPOS.restype = BOOL
qPOS.argtypes = [c_int, STRING, POINTER(c_double)]
qPOS.argnames = ['ID', 'szAxes', 'pdValarray']
qPOS.__doc__ = \
"""BOOL Mercury_qPOS(int ID, unknown * szAxes, double * pdValarray)
PI_Mercury_GCS_DLL.h:73"""

POS = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_POS
POS.restype = BOOL
POS.argtypes = [c_int, STRING, POINTER(c_double)]
POS.argnames = ['ID', 'szAxes', 'pdValarray']
POS.__doc__ = \
"""BOOL Mercury_POS(int ID, unknown * szAxes, unknown * pdValarray)
PI_Mercury_GCS_DLL.h:74"""

HLT = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_HLT
HLT.restype = BOOL
HLT.argtypes = [c_int, STRING]
HLT.argnames = ['ID', 'szAxes']
HLT.__doc__ = \
"""BOOL Mercury_HLT(int ID, unknown * szAxes)
PI_Mercury_GCS_DLL.h:76"""

STP = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_STP
STP.restype = BOOL
STP.argtypes = [c_int]
STP.argnames = ['ID']
STP.__doc__ = \
"""BOOL Mercury_STP(int ID)
PI_Mercury_GCS_DLL.h:77"""

qCST = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qCST
qCST.restype = BOOL
qCST.argtypes = [c_int, STRING, STRING, c_int]
qCST.argnames = ['ID', 'szAxes', 'names', 'maxlen']
qCST.__doc__ = \
"""BOOL Mercury_qCST(int ID, unknown * szAxes, char * names, int maxlen)
PI_Mercury_GCS_DLL.h:79"""

CST = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_CST
CST.restype = BOOL
CST.argtypes = [c_int, STRING, STRING]
CST.argnames = ['ID', 'szAxes', 'names']
CST.__doc__ = \
"""BOOL Mercury_CST(int ID, unknown * szAxes, unknown * names)
PI_Mercury_GCS_DLL.h:80"""

qVST = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qVST
qVST.restype = BOOL
qVST.argtypes = [c_int, STRING, c_int]
qVST.argnames = ['ID', 'buffer', 'maxlen']
qVST.__doc__ = \
"""BOOL Mercury_qVST(int ID, char * buffer, int maxlen)
PI_Mercury_GCS_DLL.h:81"""

qTVI = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTVI
qTVI.restype = BOOL
qTVI.argtypes = [c_int, STRING, c_int]
qTVI.argnames = ['ID', 'axes', 'maxlen']
qTVI.__doc__ = \
"""BOOL Mercury_qTVI(int ID, char * axes, int maxlen)
PI_Mercury_GCS_DLL.h:82"""

SAI = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_SAI
SAI.restype = BOOL
SAI.argtypes = [c_int, STRING, STRING]
SAI.argnames = ['ID', 'szOldAxes', 'szNewAxes']
SAI.__doc__ = \
"""BOOL Mercury_SAI(int ID, unknown * szOldAxes, unknown * szNewAxes)
PI_Mercury_GCS_DLL.h:83"""

qSAI = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qSAI
qSAI.restype = BOOL
qSAI.argtypes = [c_int, STRING, c_int]
qSAI.argnames = ['ID', 'axes', 'maxlen']
qSAI.__doc__ = \
"""BOOL Mercury_qSAI(int ID, char * axes, int maxlen)
PI_Mercury_GCS_DLL.h:84"""

qSAI_ALL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qSAI_ALL
qSAI_ALL.restype = BOOL
qSAI_ALL.argtypes = [c_int, STRING, c_int]
qSAI_ALL.argnames = ['ID', 'axes', 'maxlen']
qSAI_ALL.__doc__ = \
"""BOOL Mercury_qSAI_ALL(int ID, char * axes, int maxlen)
PI_Mercury_GCS_DLL.h:85"""

SVO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_SVO
SVO.restype = BOOL
SVO.argtypes = [c_int, STRING, POINTER(BOOL)]
SVO.argnames = ['ID', 'szAxes', 'pbValarray']
SVO.__doc__ = \
"""BOOL Mercury_SVO(int ID, unknown * szAxes, unknown * pbValarray)
PI_Mercury_GCS_DLL.h:87"""

qSVO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qSVO
qSVO.restype = BOOL
qSVO.argtypes = [c_int, STRING, POINTER(BOOL)]
qSVO.argnames = ['ID', 'szAxes', 'pbValarray']
qSVO.__doc__ = \
"""BOOL Mercury_qSVO(int ID, unknown * szAxes, BOOL * pbValarray)
PI_Mercury_GCS_DLL.h:88"""

VEL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_VEL
VEL.restype = BOOL
VEL.argtypes = [c_int, STRING, POINTER(c_double)]
VEL.argnames = ['ID', 'szAxes', 'pdValarray']
VEL.__doc__ = \
"""BOOL Mercury_VEL(int ID, unknown * szAxes, unknown * pdValarray)
PI_Mercury_GCS_DLL.h:90"""

qVEL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qVEL
qVEL.restype = BOOL
qVEL.argtypes = [c_int, STRING, POINTER(c_double)]
qVEL.argnames = ['ID', 'szAxes', 'pdValarray']
qVEL.__doc__ = \
"""BOOL Mercury_qVEL(int ID, unknown * szAxes, double * pdValarray)
PI_Mercury_GCS_DLL.h:91"""

SPA = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_SPA
SPA.restype = BOOL
SPA.argtypes = [c_int, STRING, POINTER(c_int), POINTER(c_double), STRING]
SPA.argnames = ['ID', 'szAxes', 'iCmdarray', 'dValarray', 'szStageNames']
SPA.__doc__ = \
"""BOOL Mercury_SPA(int ID, unknown * szAxes, unknown * iCmdarray, unknown * dValarray, unknown * szStageNames)
PI_Mercury_GCS_DLL.h:93"""

qSPA = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qSPA
qSPA.restype = BOOL
qSPA.argtypes = [c_int, STRING, POINTER(c_int), POINTER(c_double), STRING, c_int]
qSPA.argnames = ['ID', 'szAxes', 'iCmdarray', 'dValarray', 'szStageNames', 'iMaxNameSize']
qSPA.__doc__ = \
"""BOOL Mercury_qSPA(int ID, unknown * szAxes, unknown * iCmdarray, double * dValarray, char * szStageNames, int iMaxNameSize)
PI_Mercury_GCS_DLL.h:94"""

qSRG = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qSRG
qSRG.restype = BOOL
qSRG.argtypes = [c_int, STRING, POINTER(c_int), POINTER(c_int)]
qSRG.argnames = ['ID', 'szAxes', 'iCmdarray', 'iValarray']
qSRG.__doc__ = \
"""BOOL Mercury_qSRG(int ID, unknown * szAxes, unknown * iCmdarray, int * iValarray)
PI_Mercury_GCS_DLL.h:96"""

GetInputChannelNames = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetInputChannelNames
GetInputChannelNames.restype = BOOL
GetInputChannelNames.argtypes = [c_int, STRING, c_int]
GetInputChannelNames.argnames = ['ID', 'szBuffer', 'maxlen']
GetInputChannelNames.__doc__ = \
"""BOOL Mercury_GetInputChannelNames(int ID, char * szBuffer, int maxlen)
PI_Mercury_GCS_DLL.h:98"""

GetOutputChannelNames = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetOutputChannelNames
GetOutputChannelNames.restype = BOOL
GetOutputChannelNames.argtypes = [c_int, STRING, c_int]
GetOutputChannelNames.argnames = ['ID', 'szBuffer', 'maxlen']
GetOutputChannelNames.__doc__ = \
"""BOOL Mercury_GetOutputChannelNames(int ID, char * szBuffer, int maxlen)
PI_Mercury_GCS_DLL.h:99"""

DIO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_DIO
DIO.restype = BOOL
DIO.argtypes = [c_int, STRING, POINTER(BOOL)]
DIO.argnames = ['ID', 'szChannels', 'pbValarray']
DIO.__doc__ = \
"""BOOL Mercury_DIO(int ID, unknown * szChannels, unknown * pbValarray)
PI_Mercury_GCS_DLL.h:100"""

qDIO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qDIO
qDIO.restype = BOOL
qDIO.argtypes = [c_int, STRING, POINTER(BOOL)]
qDIO.argnames = ['ID', 'szChannels', 'pbValarray']
qDIO.__doc__ = \
"""BOOL Mercury_qDIO(int ID, unknown * szChannels, BOOL * pbValarray)
PI_Mercury_GCS_DLL.h:101"""

qTIO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTIO
qTIO.restype = BOOL
qTIO.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
qTIO.argnames = ['ID', 'pINr', 'pONr']
qTIO.__doc__ = \
"""BOOL Mercury_qTIO(int ID, int * pINr, int * pONr)
PI_Mercury_GCS_DLL.h:102"""

qTAC = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTAC
qTAC.restype = BOOL
qTAC.argtypes = [c_int, POINTER(c_int)]
qTAC.argnames = ['ID', 'pnNr']
qTAC.__doc__ = \
"""BOOL Mercury_qTAC(int ID, int * pnNr)
PI_Mercury_GCS_DLL.h:104"""

qTAV = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTAV
qTAV.restype = BOOL
qTAV.argtypes = [c_int, c_int, POINTER(c_double)]
qTAV.argnames = ['ID', 'nChannel', 'pdValue']
qTAV.__doc__ = \
"""BOOL Mercury_qTAV(int ID, int nChannel, double * pdValue)
PI_Mercury_GCS_DLL.h:105"""

BRA = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_BRA
BRA.restype = BOOL
BRA.argtypes = [c_int, STRING, POINTER(BOOL)]
BRA.argnames = ['ID', 'szAxes', 'pbValarray']
BRA.__doc__ = \
"""BOOL Mercury_BRA(int ID, unknown * szAxes, unknown * pbValarray)
PI_Mercury_GCS_DLL.h:107"""

qBRA = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qBRA
qBRA.restype = BOOL
qBRA.argtypes = [c_int, STRING, c_int]
qBRA.argnames = ['ID', 'szBuffer', 'maxlen']
qBRA.__doc__ = \
"""BOOL Mercury_qBRA(int ID, char * szBuffer, int maxlen)
PI_Mercury_GCS_DLL.h:108"""

qHLP = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qHLP
qHLP.restype = BOOL
qHLP.argtypes = [c_int, STRING, c_int]
qHLP.argnames = ['ID', 'buffer', 'maxlen']
qHLP.__doc__ = \
"""BOOL Mercury_qHLP(int ID, char * buffer, int maxlen)
PI_Mercury_GCS_DLL.h:110"""

qHPA = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qHPA
qHPA.restype = BOOL
qHPA.argtypes = [c_long, STRING, c_long]
qHPA.argnames = ['ID', 'szBuffer', 'iBufferSize']
qHPA.__doc__ = \
"""BOOL Mercury_qHPA(long int ID, char * szBuffer, long int iBufferSize)
PI_Mercury_GCS_DLL.h:111"""

SendNonGCSString = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_SendNonGCSString
SendNonGCSString.restype = BOOL
SendNonGCSString.argtypes = [c_int, STRING]
SendNonGCSString.argnames = ['ID', 'szString']
SendNonGCSString.__doc__ = \
"""BOOL Mercury_SendNonGCSString(int ID, unknown * szString)
PI_Mercury_GCS_DLL.h:115"""

ReceiveNonGCSString = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_ReceiveNonGCSString
ReceiveNonGCSString.restype = BOOL
ReceiveNonGCSString.argtypes = [c_int, STRING, c_int]
ReceiveNonGCSString.argnames = ['ID', 'szString', 'iMaxSize']
ReceiveNonGCSString.__doc__ = \
"""BOOL Mercury_ReceiveNonGCSString(int ID, char * szString, int iMaxSize)
PI_Mercury_GCS_DLL.h:116"""

GcsCommandset = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GcsCommandset
GcsCommandset.restype = BOOL
GcsCommandset.argtypes = [c_int, STRING]
GcsCommandset.argnames = ['ID', 'szCommand']
GcsCommandset.__doc__ = \
"""BOOL Mercury_GcsCommandset(int ID, unknown * szCommand)
PI_Mercury_GCS_DLL.h:118"""

GcsGetAnswer = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GcsGetAnswer
GcsGetAnswer.restype = BOOL
GcsGetAnswer.argtypes = [c_int, STRING, c_int]
GcsGetAnswer.argnames = ['ID', 'szAnswer', 'bufsize']
GcsGetAnswer.__doc__ = \
"""BOOL Mercury_GcsGetAnswer(int ID, char * szAnswer, int bufsize)
PI_Mercury_GCS_DLL.h:119"""

GcsGetAnswerSize = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GcsGetAnswerSize
GcsGetAnswerSize.restype = BOOL
GcsGetAnswerSize.argtypes = [c_int, POINTER(c_int)]
GcsGetAnswerSize.argnames = ['ID', 'iAnswerSize']
GcsGetAnswerSize.__doc__ = \
"""BOOL Mercury_GcsGetAnswerSize(int ID, int * iAnswerSize)
PI_Mercury_GCS_DLL.h:120"""

MNL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MNL
MNL.restype = BOOL
MNL.argtypes = [c_int, STRING]
MNL.argnames = ['ID', 'szAxes']
MNL.__doc__ = \
"""BOOL Mercury_MNL(int ID, unknown * szAxes)
PI_Mercury_GCS_DLL.h:126"""

MPL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MPL
MPL.restype = BOOL
MPL.argtypes = [c_int, STRING]
MPL.argnames = ['ID', 'szAxes']
MPL.__doc__ = \
"""BOOL Mercury_MPL(int ID, unknown * szAxes)
PI_Mercury_GCS_DLL.h:127"""

REF = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_REF
REF.restype = BOOL
REF.argtypes = [c_int, STRING]
REF.argnames = ['ID', 'szAxes']
REF.__doc__ = \
"""BOOL Mercury_REF(int ID, unknown * szAxes)
PI_Mercury_GCS_DLL.h:128"""

qREF = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qREF
qREF.restype = BOOL
qREF.argtypes = [c_int, STRING, POINTER(BOOL)]
qREF.argnames = ['ID', 'szAxes', 'pbValarray']
qREF.__doc__ = \
"""BOOL Mercury_qREF(int ID, unknown * szAxes, BOOL * pbValarray)
PI_Mercury_GCS_DLL.h:129"""

qLIM = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qLIM
qLIM.restype = BOOL
qLIM.argtypes = [c_int, STRING, POINTER(BOOL)]
qLIM.argnames = ['ID', 'szAxes', 'pbValarray']
qLIM.__doc__ = \
"""BOOL Mercury_qLIM(int ID, unknown * szAxes, BOOL * pbValarray)
PI_Mercury_GCS_DLL.h:130"""

IsReferencing = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsReferencing
IsReferencing.restype = BOOL
IsReferencing.argtypes = [c_int, STRING, POINTER(BOOL)]
IsReferencing.argnames = ['ID', 'szAxes', 'pbIsReferencing']
IsReferencing.__doc__ = \
"""BOOL Mercury_IsReferencing(int ID, unknown * szAxes, BOOL * pbIsReferencing)
PI_Mercury_GCS_DLL.h:131"""

GetRefResult = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetRefResult
GetRefResult.restype = BOOL
GetRefResult.argtypes = [c_int, STRING, POINTER(c_int)]
GetRefResult.argnames = ['ID', 'szAxes', 'pnResult']
GetRefResult.__doc__ = \
"""BOOL Mercury_GetRefResult(int ID, unknown * szAxes, int * pnResult)
PI_Mercury_GCS_DLL.h:132"""

IsReferenceOK = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsReferenceOK
IsReferenceOK.restype = BOOL
IsReferenceOK.argtypes = [c_int, STRING, POINTER(BOOL)]
IsReferenceOK.argnames = ['ID', 'szAxes', 'pbValarray']
IsReferenceOK.__doc__ = \
"""BOOL Mercury_IsReferenceOK(int ID, unknown * szAxes, BOOL * pbValarray)
PI_Mercury_GCS_DLL.h:133"""

qTMN = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTMN
qTMN.restype = BOOL
qTMN.argtypes = [c_int, STRING, POINTER(c_double)]
qTMN.argnames = ['ID', 'szAxes', 'pdValarray']
qTMN.__doc__ = \
"""BOOL Mercury_qTMN(int ID, unknown * szAxes, double * pdValarray)
PI_Mercury_GCS_DLL.h:134"""

qTMX = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTMX
qTMX.restype = BOOL
qTMX.argtypes = [c_int, STRING, POINTER(c_double)]
qTMX.argnames = ['ID', 'szAxes', 'pdValarray']
qTMX.__doc__ = \
"""BOOL Mercury_qTMX(int ID, unknown * szAxes, double * pdValarray)
PI_Mercury_GCS_DLL.h:135"""

RON = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_RON
RON.restype = BOOL
RON.argtypes = [c_int, STRING, POINTER(BOOL)]
RON.argnames = ['ID', 'szAxes', 'pbValarray']
RON.__doc__ = \
"""BOOL Mercury_RON(int ID, unknown * szAxes, unknown * pbValarray)
PI_Mercury_GCS_DLL.h:136"""

qRON = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qRON
qRON.restype = BOOL
qRON.argtypes = [c_int, STRING, POINTER(BOOL)]
qRON.argnames = ['ID', 'szAxes', 'pbValarray']
qRON.__doc__ = \
"""BOOL Mercury_qRON(int ID, unknown * szAxes, BOOL * pbValarray)
PI_Mercury_GCS_DLL.h:137"""

IsRecordingMacro = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsRecordingMacro
IsRecordingMacro.restype = BOOL
IsRecordingMacro.argtypes = [c_int, POINTER(BOOL)]
IsRecordingMacro.argnames = ['ID', 'pbRecordingMacro']
IsRecordingMacro.__doc__ = \
"""BOOL Mercury_IsRecordingMacro(int ID, BOOL * pbRecordingMacro)
PI_Mercury_GCS_DLL.h:141"""

IsRunningMacro = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_IsRunningMacro
IsRunningMacro.restype = BOOL
IsRunningMacro.argtypes = [c_int, POINTER(BOOL)]
IsRunningMacro.argnames = ['ID', 'pbRunningMacro']
IsRunningMacro.__doc__ = \
"""BOOL Mercury_IsRunningMacro(int ID, BOOL * pbRunningMacro)
PI_Mercury_GCS_DLL.h:142"""

DEL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_DEL
DEL.restype = BOOL
DEL.argtypes = [c_int, c_int]
DEL.argnames = ['ID', 'nMilliSeconds']
DEL.__doc__ = \
"""BOOL Mercury_DEL(int ID, int nMilliSeconds)
PI_Mercury_GCS_DLL.h:144"""

MAC_BEG = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MAC_BEG
MAC_BEG.restype = BOOL
MAC_BEG.argtypes = [c_int, STRING]
MAC_BEG.argnames = ['ID', 'szMacroName']
MAC_BEG.__doc__ = \
"""BOOL Mercury_MAC_BEG(int ID, unknown * szMacroName)
PI_Mercury_GCS_DLL.h:145"""

MAC_START = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MAC_START
MAC_START.restype = BOOL
MAC_START.argtypes = [c_int, STRING]
MAC_START.argnames = ['ID', 'szMacroName']
MAC_START.__doc__ = \
"""BOOL Mercury_MAC_START(int ID, unknown * szMacroName)
PI_Mercury_GCS_DLL.h:146"""

MAC_NSTART = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MAC_NSTART
MAC_NSTART.restype = BOOL
MAC_NSTART.argtypes = [c_int, STRING, c_int]
MAC_NSTART.argnames = ['ID', 'szMacroName', 'nrRuns']
MAC_NSTART.__doc__ = \
"""BOOL Mercury_MAC_NSTART(int ID, unknown * szMacroName, int nrRuns)
PI_Mercury_GCS_DLL.h:147"""

MAC_END = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MAC_END
MAC_END.restype = BOOL
MAC_END.argtypes = [c_int]
MAC_END.argnames = ['ID']
MAC_END.__doc__ = \
"""BOOL Mercury_MAC_END(int ID)
PI_Mercury_GCS_DLL.h:148"""

MEX = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MEX
MEX.restype = BOOL
MEX.argtypes = [c_int, STRING]
MEX.argnames = ['ID', 'szCondition']
MEX.__doc__ = \
"""BOOL Mercury_MEX(int ID, unknown * szCondition)
PI_Mercury_GCS_DLL.h:149"""

MAC_DEL = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_MAC_DEL
MAC_DEL.restype = BOOL
MAC_DEL.argtypes = [c_int, STRING]
MAC_DEL.argnames = ['ID', 'szMacroName']
MAC_DEL.__doc__ = \
"""BOOL Mercury_MAC_DEL(int ID, unknown * szMacroName)
PI_Mercury_GCS_DLL.h:150"""

qMAC = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qMAC
qMAC.restype = BOOL
qMAC.argtypes = [c_int, STRING, STRING, c_int]
qMAC.argnames = ['ID', 'szMacroName', 'szBuffer', 'maxlen']
qMAC.__doc__ = \
"""BOOL Mercury_qMAC(int ID, unknown * szMacroName, char * szBuffer, int maxlen)
PI_Mercury_GCS_DLL.h:151"""

WAC = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_WAC
WAC.restype = BOOL
WAC.argtypes = [c_int, STRING]
WAC.argnames = ['ID', 'szCondition']
WAC.__doc__ = \
"""BOOL Mercury_WAC(int ID, unknown * szCondition)
PI_Mercury_GCS_DLL.h:153"""

JON = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_JON
JON.restype = BOOL
JON.argtypes = [c_int, POINTER(c_int), POINTER(BOOL), c_int]
JON.argnames = ['ID', 'iJoystickIDs', 'pbValarray', 'iArraySize']
JON.__doc__ = \
"""BOOL Mercury_JON(int ID, unknown * iJoystickIDs, unknown * pbValarray, int iArraySize)
PI_Mercury_GCS_DLL.h:155"""

qJON = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qJON
qJON.restype = BOOL
qJON.argtypes = [c_int, POINTER(c_int), POINTER(BOOL), c_int]
qJON.argnames = ['ID', 'iJoystickIDs', 'pbValarray', 'iArraySize']
qJON.__doc__ = \
"""BOOL Mercury_qJON(int ID, unknown * iJoystickIDs, BOOL * pbValarray, int iArraySize)
PI_Mercury_GCS_DLL.h:156"""

qTNJ = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTNJ
qTNJ.restype = BOOL
qTNJ.argtypes = [c_int, POINTER(c_int)]
qTNJ.argnames = ['ID', 'pnNr']
qTNJ.__doc__ = \
"""BOOL Mercury_qTNJ(int ID, int * pnNr)
PI_Mercury_GCS_DLL.h:157"""

qJAX = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qJAX
qJAX.restype = BOOL
qJAX.argtypes = [c_int, POINTER(c_int), POINTER(c_int), c_int, STRING, c_int]
qJAX.argnames = ['ID', 'iJoystickIDs', 'iAxesIDs', 'iArraySize', 'szAxesBuffer', 'iBufferSize']
qJAX.__doc__ = \
"""BOOL Mercury_qJAX(int ID, unknown * iJoystickIDs, unknown * iAxesIDs, int iArraySize, char * szAxesBuffer, int iBufferSize)
PI_Mercury_GCS_DLL.h:158"""

JDT = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_JDT
JDT.restype = BOOL
JDT.argtypes = [c_int, POINTER(c_int), POINTER(c_int), c_int]
JDT.argnames = ['ID', 'iJoystickIDs', 'iValarray', 'iArraySize']
JDT.__doc__ = \
"""BOOL Mercury_JDT(int ID, unknown * iJoystickIDs, unknown * iValarray, int iArraySize)
PI_Mercury_GCS_DLL.h:159"""

qJBS = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qJBS
qJBS.restype = BOOL
qJBS.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(BOOL), c_int]
qJBS.argnames = ['ID', 'iJoystickIDs', 'iAxesIDs', 'pbValueArray', 'iArraySize']
qJBS.__doc__ = \
"""BOOL Mercury_qJBS(int ID, unknown * iJoystickIDs, unknown * iAxesIDs, BOOL * pbValueArray, int iArraySize)
PI_Mercury_GCS_DLL.h:160"""

qJAS = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qJAS
qJAS.restype = BOOL
qJAS.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(c_double), c_int]
qJAS.argnames = ['ID', 'iJoystickIDs', 'iAxesIDs', 'pdValueArray', 'iArraySize']
qJAS.__doc__ = \
"""BOOL Mercury_qJAS(int ID, unknown * iJoystickIDs, unknown * iAxesIDs, double * pdValueArray, int iArraySize)
PI_Mercury_GCS_DLL.h:161"""

JLT = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_JLT
JLT.restype = BOOL
JLT.argtypes = [c_int, c_int, c_int, c_int, POINTER(c_double), c_int]
JLT.argnames = ['ID', 'iJoystickID', 'iAxisID', 'iStartAdress', 'pdValueArray', 'iArraySize']
JLT.__doc__ = \
"""BOOL Mercury_JLT(int ID, int iJoystickID, int iAxisID, int iStartAdress, unknown * pdValueArray, int iArraySize)
PI_Mercury_GCS_DLL.h:162"""

qJLT = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qJLT
qJLT.restype = BOOL
qJLT.argtypes = [c_int, POINTER(c_int), POINTER(c_int), c_int, c_int, c_int, POINTER(POINTER(c_double)), STRING, c_int]
qJLT.argnames = ['ID', 'iJoystickIDsArray', 'iAxisIDsArray', 'iNumberOfTables', 'iOffsetOfFirstPointInTable', 'iNumberOfValues', 'pdValueArray', 'szGcsArrayHeader', 'iGcsArrayHeaderMaxSize']
qJLT.__doc__ = \
"""BOOL Mercury_qJLT(int ID, unknown * iJoystickIDsArray, unknown * iAxisIDsArray, int iNumberOfTables, int iOffsetOfFirstPointInTable, int iNumberOfValues, double * * pdValueArray, char * szGcsArrayHeader, int iGcsArrayHeaderMaxSize)
PI_Mercury_GCS_DLL.h:163"""

CTO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_CTO
CTO.restype = BOOL
CTO.argtypes = [c_int, POINTER(c_long), POINTER(c_long), STRING, c_int]
CTO.argnames = ['ID', 'iTriggerLines', 'iParamID', 'szValues', 'iArraySize']
CTO.__doc__ = \
"""BOOL Mercury_CTO(int ID, unknown * iTriggerLines, unknown * iParamID, unknown * szValues, int iArraySize)
PI_Mercury_GCS_DLL.h:165"""

qCTO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qCTO
qCTO.restype = BOOL
qCTO.argtypes = [c_int, POINTER(c_long), POINTER(c_long), STRING, c_int, c_int]
qCTO.argnames = ['ID', 'iTriggerLines', 'pParamID', 'szBuffer', 'iArraySize', 'iBufferMaxlen']
qCTO.__doc__ = \
"""BOOL Mercury_qCTO(int ID, unknown * iTriggerLines, unknown * pParamID, char * szBuffer, int iArraySize, int iBufferMaxlen)
PI_Mercury_GCS_DLL.h:166"""

TRO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_TRO
TRO.restype = BOOL
TRO.argtypes = [c_int, POINTER(c_long), POINTER(BOOL), c_int]
TRO.argnames = ['ID', 'iTriggerLines', 'pbValarray', 'iArraySize']
TRO.__doc__ = \
"""BOOL Mercury_TRO(int ID, unknown * iTriggerLines, unknown * pbValarray, int iArraySize)
PI_Mercury_GCS_DLL.h:168"""

qTRO = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_qTRO
qTRO.restype = BOOL
qTRO.argtypes = [c_int, POINTER(c_long), POINTER(BOOL), c_int]
qTRO.argnames = ['ID', 'iTriggerLines', 'pbValarray', 'iArraySize']
qTRO.__doc__ = \
"""BOOL Mercury_qTRO(int ID, unknown * iTriggerLines, BOOL * pbValarray, int iArraySize)
PI_Mercury_GCS_DLL.h:169"""

AddStage = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_AddStage
AddStage.restype = BOOL
AddStage.argtypes = [c_int, STRING]
AddStage.argnames = ['ID', 'szAxes']
AddStage.__doc__ = \
"""BOOL Mercury_AddStage(int ID, unknown * szAxes)
PI_Mercury_GCS_DLL.h:174"""

RemoveStage = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_RemoveStage
RemoveStage.restype = BOOL
RemoveStage.argtypes = [c_int, STRING]
RemoveStage.argnames = ['ID', 'szStageName']
RemoveStage.__doc__ = \
"""BOOL Mercury_RemoveStage(int ID, unknown * szStageName)
PI_Mercury_GCS_DLL.h:175"""

OpenUserStagesEditDialog = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_OpenUserStagesEditDialog
OpenUserStagesEditDialog.restype = BOOL
OpenUserStagesEditDialog.argtypes = [c_int]
OpenUserStagesEditDialog.argnames = ['ID']
OpenUserStagesEditDialog.__doc__ = \
"""BOOL Mercury_OpenUserStagesEditDialog(int ID)
PI_Mercury_GCS_DLL.h:177"""

OpenPiStagesEditDialog = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_OpenPiStagesEditDialog
OpenPiStagesEditDialog.restype = BOOL
OpenPiStagesEditDialog.argtypes = [c_int]
OpenPiStagesEditDialog.argnames = ['ID']
OpenPiStagesEditDialog.__doc__ = \
"""BOOL Mercury_OpenPiStagesEditDialog(int ID)
PI_Mercury_GCS_DLL.h:178"""

GetStatus = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetStatus
GetStatus.restype = BOOL
GetStatus.argtypes = [c_int, STRING, POINTER(c_int), POINTER(c_int)]
GetStatus.argnames = ['ID', 'szAxes', 'icmdarry', 'piValarray']
GetStatus.__doc__ = \
"""BOOL Mercury_GetStatus(int ID, unknown * szAxes, unknown * icmdarry, int * piValarray)
PI_Mercury_GCS_DLL.h:180"""

GetAsyncBufferIndex = _stdcall_libraries['PI_Mercury_GCS_DLL.dll'].Mercury_GetAsyncBufferIndex
GetAsyncBufferIndex.restype = c_int
GetAsyncBufferIndex.argtypes = [c_long]
GetAsyncBufferIndex.argnames = ['ID']
GetAsyncBufferIndex.__doc__ = \
"""int Mercury_GetAsyncBufferIndex(long int ID)
PI_Mercury_GCS_DLL.h:181"""

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
