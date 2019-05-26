#!/usr/bin/python

###############
# GCS_DLL.py
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
"""
Adapted from Mercury GCS controller, initially for E816. Ideally these should be refactored so that both
can use the same code.

"""
from ctypes import *
import ctypes
import platform

STRING = c_char_p
_stdcall_libraries = {}

#gcs_dll = WinDLL('PI_Mercury_GCS_DLL.dll')

arch, plat = platform.architecture()



dll_prefix = 'E816_'
prefix =  'E816_'

if plat.startswith('Windows'):
    if arch == '32bit':
        gcs_dll = WinDLL(dll_prefix + 'DLL.dll')
    else:
        gcs_dll = WinDLL(dll_prefix + 'DLL_x64.dll')
    from ctypes.wintypes import ULONG, DWORD, BOOL, BYTE, WORD, UINT, HANDLE, HWND
else:
    raise RuntimeError('Not a supported platform: %s' % plat)




from ctypes.wintypes import BOOL


InterfaceSetupDlg = getattr(gcs_dll, prefix + 'InterfaceSetupDlg')
InterfaceSetupDlg.restype = c_int
InterfaceSetupDlg.argtypes = [STRING]
InterfaceSetupDlg.argnames = ['szRegKeyName']
InterfaceSetupDlg.__doc__ = \
"""int Mercury_InterfaceSetupDlg(unknown * szRegKeyName)
PI_Mercury_GCS_DLL.h:40"""

ConnectRS232 = getattr(gcs_dll, prefix + 'ConnectRS232')
ConnectRS232.restype = c_int
ConnectRS232.argtypes = [c_int, c_int]
ConnectRS232.argnames = ['port', 'baudrate']
ConnectRS232.__doc__ = \
"""int Mercury_ConnectRS232(int port, int baudrate)
PI_Mercury_GCS_DLL.h:42"""

ConnectUSB = getattr(gcs_dll, prefix + 'ConnectUSB')
ConnectUSB.restype = c_int
ConnectUSB.argtypes = [STRING]
ConnectUSB.argnames = ['szDescription']
ConnectUSB.__doc__ = \
"""int Mercury_ConnectUSB(const char *szDescription)
"""

EnumerateUSB = getattr(gcs_dll, prefix + 'EnumerateUSB')
EnumerateUSB.restype = c_int
EnumerateUSB.argtypes = [STRING, c_int, STRING]
EnumerateUSB.argnames = ['szBuffer', 'maxlen', 'szFilter']
EnumerateUSB.__doc__ = \
"""int Mercury_EnumerateUSB(char * szBuffer, int iBufferSize, const char *szFilter)
"""

IsConnected = getattr(gcs_dll, prefix + 'IsConnected')
IsConnected.restype = BOOL
IsConnected.argtypes = [c_int]
IsConnected.argnames = ['ID']
IsConnected.__doc__ = \
"""BOOL Mercury_IsConnected(int ID)
PI_Mercury_GCS_DLL.h:46"""

CloseConnection = getattr(gcs_dll, prefix + 'CloseConnection')
CloseConnection.restype = None
CloseConnection.argtypes = [c_int]
CloseConnection.argnames = ['ID']
CloseConnection.__doc__ = \
"""void Mercury_CloseConnection(int ID)
PI_Mercury_GCS_DLL.h:47"""

GetError = getattr(gcs_dll, prefix + 'GetError')
GetError.restype = c_int
GetError.argtypes = [c_int]
GetError.argnames = ['ID']
GetError.__doc__ = \
"""int Mercury_GetError(int ID)
PI_Mercury_GCS_DLL.h:48"""

# SetErrorCheck = getattr(gcs_dll, prefix + 'SetErrorCheck')
# SetErrorCheck.restype = BOOL
# SetErrorCheck.argtypes = [c_int, BOOL]
# SetErrorCheck.argnames = ['ID', 'bErrorCheck']
# SetErrorCheck.__doc__ = \
# """BOOL Mercury_SetErrorCheck(int ID, BOOL bErrorCheck)
# PI_Mercury_GCS_DLL.h:49"""

TranslateError = getattr(gcs_dll, prefix + 'TranslateError')
TranslateError.restype = BOOL
TranslateError.argtypes = [c_int, STRING, c_int]
TranslateError.argnames = ['errNr', 'szBuffer', 'maxlen']
TranslateError.__doc__ = \
"""BOOL Mercury_TranslateError(int errNr, char * szBuffer, int maxlen)
PI_Mercury_GCS_DLL.h:50"""

qERR = getattr(gcs_dll, prefix + 'qERR')
qERR.restype = BOOL
qERR.argtypes = [c_int, POINTER(c_int)]
qERR.argnames = ['ID', 'pnError']
qERR.__doc__ = \
"""BOOL Mercury_qERR(int ID, int * pnError)
PI_Mercury_GCS_DLL.h:55"""

qIDN = getattr(gcs_dll, prefix + 'qIDN')
qIDN.restype = BOOL
qIDN.argtypes = [c_int, STRING, c_int]
qIDN.argnames = ['ID', 'buffer', 'maxlen']
qIDN.__doc__ = \
"""BOOL Mercury_qIDN(int ID, char * buffer, int maxlen)
PI_Mercury_GCS_DLL.h:56"""




MOV = getattr(gcs_dll, prefix + 'MOV')
MOV.restype = BOOL
MOV.argtypes = [c_int, STRING, POINTER(c_double)]
MOV.argnames = ['ID', 'szAxes', 'pdValarray']
MOV.__doc__ = \
"""BOOL Mercury_MOV(int ID, unknown * szAxes, unknown * pdValarray)
PI_Mercury_GCS_DLL.h:60"""

qMOV = getattr(gcs_dll, prefix + 'qMOV')
qMOV.restype = BOOL
qMOV.argtypes = [c_int, STRING, POINTER(c_double)]
qMOV.argnames = ['ID', 'szAxes', 'pdValarray']
qMOV.__doc__ = \
"""BOOL Mercury_qMOV(int ID, unknown * szAxes, double * pdValarray)
PI_Mercury_GCS_DLL.h:61"""

MVR = getattr(gcs_dll, prefix + 'MVR')
MVR.restype = BOOL
MVR.argtypes = [c_int, STRING, POINTER(c_double)]
MVR.argnames = ['ID', 'szAxes', 'pdValarray']
MVR.__doc__ = \
"""BOOL Mercury_MVR(int ID, unknown * szAxes, unknown * pdValarray)
PI_Mercury_GCS_DLL.h:62"""



qONT = getattr(gcs_dll, prefix + 'qONT')
qONT.restype = BOOL
qONT.argtypes = [c_int, STRING, POINTER(BOOL)]
qONT.argnames = ['ID', 'szAxes', 'pbValarray']
qONT.__doc__ = \
"""BOOL Mercury_qONT(int ID, unknown * szAxes, BOOL * pbValarray)
PI_Mercury_GCS_DLL.h:64"""



qPOS = getattr(gcs_dll, prefix + 'qPOS')
qPOS.restype = BOOL
qPOS.argtypes = [c_int, STRING, POINTER(c_double)]
qPOS.argnames = ['ID', 'szAxes', 'pdValarray']
qPOS.__doc__ = \
"""BOOL Mercury_qPOS(int ID, unknown * szAxes, double * pdValarray)
PI_Mercury_GCS_DLL.h:73"""

# POS = getattr(gcs_dll, prefix + 'POS')
# POS.restype = BOOL
# POS.argtypes = [c_int, STRING, POINTER(c_double)]
# POS.argnames = ['ID', 'szAxes', 'pdValarray']
# POS.__doc__ = \
# """BOOL Mercury_POS(int ID, unknown * szAxes, unknown * pdValarray)
# PI_Mercury_GCS_DLL.h:74"""





# SAI = getattr(gcs_dll, prefix + 'SAI')
# SAI.restype = BOOL
# SAI.argtypes = [c_int, STRING, STRING]
# SAI.argnames = ['ID', 'szOldAxes', 'szNewAxes']
# SAI.__doc__ = \
# """BOOL Mercury_SAI(int ID, unknown * szOldAxes, unknown * szNewAxes)
# PI_Mercury_GCS_DLL.h:83"""
#
# qSAI = getattr(gcs_dll, prefix + 'qSAI')
# qSAI.restype = BOOL
# qSAI.argtypes = [c_int, STRING, c_int]
# qSAI.argnames = ['ID', 'axes', 'maxlen']
# qSAI.__doc__ = \
# """BOOL Mercury_qSAI(int ID, char * axes, int maxlen)
# PI_Mercury_GCS_DLL.h:84"""


SVO = getattr(gcs_dll, prefix + 'SVO')
SVO.restype = BOOL
SVO.argtypes = [c_int, STRING, POINTER(BOOL)]
SVO.argnames = ['ID', 'szAxes', 'pbValarray']
SVO.__doc__ = \
"""BOOL Mercury_SVO(int ID, unknown * szAxes, unknown * pbValarray)
PI_Mercury_GCS_DLL.h:87"""

qSVO = getattr(gcs_dll, prefix + 'qSVO')
qSVO.restype = BOOL
qSVO.argtypes = [c_int, STRING, POINTER(BOOL)]
qSVO.argnames = ['ID', 'szAxes', 'pbValarray']
qSVO.__doc__ = \
"""BOOL Mercury_qSVO(int ID, unknown * szAxes, BOOL * pbValarray)
PI_Mercury_GCS_DLL.h:88"""


SPA = getattr(gcs_dll, prefix + 'SPA')
SPA.restype = BOOL
SPA.argtypes = [c_int, STRING, POINTER(c_int), POINTER(c_double), STRING]
SPA.argnames = ['ID', 'szAxes', 'iCmdarray', 'dValarray', 'szStageNames']
SPA.__doc__ = \
"""BOOL Mercury_SPA(int ID, unknown * szAxes, unknown * iCmdarray, unknown * dValarray, unknown * szStageNames)
PI_Mercury_GCS_DLL.h:93"""

qSPA = getattr(gcs_dll, prefix + 'qSPA')
qSPA.restype = BOOL
qSPA.argtypes = [c_int, STRING, POINTER(c_int), POINTER(c_double), STRING, c_int]
qSPA.argnames = ['ID', 'szAxes', 'iCmdarray', 'dValarray', 'szStageNames', 'iMaxNameSize']
qSPA.__doc__ = \
"""BOOL Mercury_qSPA(int ID, unknown * szAxes, unknown * iCmdarray, double * dValarray, char * szStageNames, int iMaxNameSize)
PI_Mercury_GCS_DLL.h:94"""


qHLP = getattr(gcs_dll, prefix + 'qHLP')
qHLP.restype = BOOL
qHLP.argtypes = [c_int, STRING, c_int]
qHLP.argnames = ['ID', 'buffer', 'maxlen']
qHLP.__doc__ = \
"""BOOL Mercury_qHLP(int ID, char * buffer, int maxlen)
PI_Mercury_GCS_DLL.h:110"""


GcsCommandset = getattr(gcs_dll, prefix + 'GcsCommandset')
GcsCommandset.restype = BOOL
GcsCommandset.argtypes = [c_int, STRING]
GcsCommandset.argnames = ['ID', 'szCommand']
GcsCommandset.__doc__ = \
"""BOOL Mercury_GcsCommandset(int ID, unknown * szCommand)
PI_Mercury_GCS_DLL.h:118"""

GcsGetAnswer = getattr(gcs_dll, prefix + 'GcsGetAnswer')
GcsGetAnswer.restype = BOOL
GcsGetAnswer.argtypes = [c_int, STRING, c_int]
GcsGetAnswer.argnames = ['ID', 'szAnswer', 'bufsize']
GcsGetAnswer.__doc__ = \
"""BOOL Mercury_GcsGetAnswer(int ID, char * szAnswer, int bufsize)
PI_Mercury_GCS_DLL.h:119"""

GcsGetAnswerSize = getattr(gcs_dll, prefix + 'GcsGetAnswerSize')
GcsGetAnswerSize.restype = BOOL
GcsGetAnswerSize.argtypes = [c_int, POINTER(c_int)]
GcsGetAnswerSize.argnames = ['ID', 'iAnswerSize']
GcsGetAnswerSize.__doc__ = \
"""BOOL Mercury_GcsGetAnswerSize(int ID, int * iAnswerSize)
PI_Mercury_GCS_DLL.h:120"""


IsRunningMacro = getattr(gcs_dll, prefix + 'IsRunningMacro')
IsRunningMacro.restype = BOOL
IsRunningMacro.argtypes = [c_int, POINTER(BOOL)]
IsRunningMacro.argnames = ['ID', 'pbRunningMacro']
IsRunningMacro.__doc__ = \
"""BOOL Mercury_IsRunningMacro(int ID, BOOL * pbRunningMacro)
PI_Mercury_GCS_DLL.h:142"""

# DEL = getattr(gcs_dll, prefix + 'DEL')
# DEL.restype = BOOL
# DEL.argtypes = [c_int, c_int]
# DEL.argnames = ['ID', 'nMilliSeconds']
# DEL.__doc__ = \
# """BOOL Mercury_DEL(int ID, int nMilliSeconds)
# PI_Mercury_GCS_DLL.h:144"""

MAC_BEG = getattr(gcs_dll, prefix + 'MAC_BEG')
MAC_BEG.restype = BOOL
MAC_BEG.argtypes = [c_int, STRING]
MAC_BEG.argnames = ['ID', 'szMacroName']
MAC_BEG.__doc__ = \
"""BOOL Mercury_MAC_BEG(int ID, unknown * szMacroName)
PI_Mercury_GCS_DLL.h:145"""

MAC_START = getattr(gcs_dll, prefix + 'MAC_START')
MAC_START.restype = BOOL
MAC_START.argtypes = [c_int, STRING]
MAC_START.argnames = ['ID', 'szMacroName']
MAC_START.__doc__ = \
"""BOOL Mercury_MAC_START(int ID, unknown * szMacroName)
PI_Mercury_GCS_DLL.h:146"""

MAC_NSTART = getattr(gcs_dll, prefix + 'MAC_NSTART')
MAC_NSTART.restype = BOOL
MAC_NSTART.argtypes = [c_int, STRING, c_int]
MAC_NSTART.argnames = ['ID', 'szMacroName', 'nrRuns']
MAC_NSTART.__doc__ = \
"""BOOL Mercury_MAC_NSTART(int ID, unknown * szMacroName, int nrRuns)
PI_Mercury_GCS_DLL.h:147"""



MAC_DEL = getattr(gcs_dll, prefix + 'MAC_DEL')
MAC_DEL.restype = BOOL
MAC_DEL.argtypes = [c_int, STRING]
MAC_DEL.argnames = ['ID', 'szMacroName']
MAC_DEL.__doc__ = \
"""BOOL Mercury_MAC_DEL(int ID, unknown * szMacroName)
PI_Mercury_GCS_DLL.h:150"""



__all__ = ['GcsGetAnswerSize',
           'qPOS',
           'MAC_BEG',
           'GcsCommandset', 'MOV',
           'qERR',
           'SPA',
           'TranslateError',
           'qSPA',
           'ConnectRS232',
           'MVR', 'SVO', 'qIDN',
           'CloseConnection',
           #'MAC_END',
           #'POS',
           #'qSAI',
           'MAC_NSTART', 'qSVO',
           'qMOV', 'qONT',
           'GcsGetAnswer',
           #'DEL', #'qMAC',
           'IsConnected', 'InterfaceSetupDlg',
           'GetError',
            'MAC_START',
            'qHLP',
           #'SetErrorCheck',
           'MAC_DEL',

           #'SAI',
           'IsRunningMacro',
           'ConnectUSB', 'EnumerateUSB']
