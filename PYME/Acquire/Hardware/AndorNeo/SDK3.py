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

import ctypes
import platform
#import ctypes
from ctypes import POINTER, c_int, c_uint, c_double, c_void_p

#### typedefs
AT_H = ctypes.c_int
AT_BOOL = ctypes.c_int
AT_64 = ctypes.c_int64
AT_U8 = ctypes.c_uint8
AT_WC = ctypes.c_wchar

_stdcall_libraries = {}

arch, plat = platform.architecture()

#if 'WinDLL' in dir():
if plat.startswith('Windows'):
    if arch == '32bit':
        _stdcall_libraries['ATCORE'] = ctypes.WinDLL('atcore')
        _stdcall_libraries['ATUTIL'] = ctypes.WinDLL('atutility')
    else:
        try:
            _stdcall_libraries['ATCORE'] = ctypes.WinDLL('atcore')
            _stdcall_libraries['ATUTIL'] = ctypes.WinDLL('atutility')
        except OSError:
            # see https://stackoverflow.com/questions/59330863/cant-import-dll-module-in-python
            # winmode=0 enforces windows default dll search mechanism including searching the path set
            # necessary since python 3.8.x
            _stdcall_libraries['ATCORE'] = ctypes.WinDLL('atcore',winmode=0)
            _stdcall_libraries['ATUTIL'] = ctypes.WinDLL('atutility',winmode=0)
    CALLBACKTYPE = ctypes.WINFUNCTYPE(c_int, AT_H, POINTER(AT_WC), c_void_p)
else:
    _stdcall_libraries['ATCORE'] = ctypes.CDLL('atcore.so')
    _stdcall_libraries['ATUTIL'] = ctypes.CDLL('atutility.so')
    CALLBACKTYPE = ctypes.CFUNCTYPE(c_int, AT_H, POINTER(AT_WC), c_void_p)

#### Defines
errorCodes = {}
def errCode(name,value):
    errorCodes[value] = name
    
AT_INFINITE = 0xFFFFFFFF
AT_CALLBACK_SUCCESS  = 0

AT_TRUE  = 1
AT_FALSE  = 0

AT_SUCCESS = 0
errCode('AT_ERR_NOTINITIALISED', 1)
errCode('AT_ERR_NOTIMPLEMENTED', 2)
errCode('AT_ERR_READONLY', 3)
errCode('AT_ERR_NOTREADABLE', 4)
errCode('AT_ERR_NOTWRITABLE', 5)
errCode('AT_ERR_OUTOFRANGE', 6)
errCode('AT_ERR_INDEXNOTAVAILABLE', 7)
errCode('AT_ERR_INDEXNOTIMPLEMENTED', 8)
errCode('AT_ERR_EXCEEDEDMAXSTRINGLENGTH', 9)
errCode('AT_ERR_CONNECTION', 10)
errCode('AT_ERR_NODATA', 11)
errCode('AT_ERR_INVALIDHANDLE', 12)
errCode('AT_ERR_TIMEDOUT', 13)
errCode('AT_ERR_BUFFERFULL', 14)
errCode('AT_ERR_INVALIDSIZE', 15)
errCode('AT_ERR_INVALIDALIGNMENT', 16)
errCode('AT_ERR_COMM', 17)
errCode('AT_ERR_STRINGNOTAVAILABLE', 18)
errCode('AT_ERR_STRINGNOTIMPLEMENTED', 19)

errCode('AT_ERR_NULL_FEATURE', 20)
errCode('AT_ERR_NULL_HANDLE', 21)
errCode('AT_ERR_NULL_IMPLEMENTED_VAR', 22)
errCode('AT_ERR_NULL_READABLE_VAR', 23)
errCode('AT_ERR_NULL_READONLY_VAR', 24)
errCode('AT_ERR_NULL_WRITABLE_VAR', 25)
errCode('AT_ERR_NULL_MINVALUE', 26)
errCode('AT_ERR_NULL_MAXVALUE', 27)
errCode('AT_ERR_NULL_VALUE', 28)
errCode('AT_ERR_NULL_STRING', 29)
errCode('AT_ERR_NULL_COUNT_VAR', 30)
errCode('AT_ERR_NULL_ISAVAILABLE_VAR', 31)
errCode('AT_ERR_NULL_MAXSTRINGLENGTH', 32)
errCode('AT_ERR_NULL_EVCALLBACK', 33)
errCode('AT_ERR_NULL_QUEUE_PTR', 34)
errCode('AT_ERR_NULL_WAIT_PTR', 35)
errCode('AT_ERR_NULL_PTRSIZE', 36)
errCode('AT_ERR_NOMEMORY', 37)

errCode('AT_ERR_HARDWARE_OVERFLOW', 100)

class CameraError(Exception):
    def __init__(self, fcnName, errNo):
        self.errNo = errNo
        self.fcnName = fcnName
        
    def __str__(self):
        return 'when calling %s - %s' % (self.fcnName, errorCodes[self.errNo])
        

#special case for buffer timeout
AT_ERR_TIMEDOUT = 13
AT_ERR_NODATA = 11

class TimeoutError(CameraError):
    pass
        


AT_HANDLE_UNINITIALISED  = -1
AT_HANDLE_SYSTEM  = 1

### Functions ###
STRING = POINTER(AT_WC)

#classes so that we do some magic and automatically add byrefs etc ... can classify outputs
class _meta(object):
    pass

class OUTPUT(_meta):
    def __init__(self, val):
        self.type = val
        self.val = POINTER(val)
    
    def getVar(self, bufLen=0):
        v = self.type()
        return v, ctypes.byref(v)
        
class _OUTSTRING(OUTPUT):
    def __init__(self):
        self.val = STRING
        
    def getVar(self, bufLen):
        v = ctypes.create_unicode_buffer(bufLen)
        return v, v
        
OUTSTRING = _OUTSTRING()

class _OUTSTRLEN(_meta):
    def __init__(self):
        self.val = c_int
        
OUTSTRLEN = _OUTSTRLEN()
        

def stripMeta(val):
    if isinstance(val, _meta):
        return val.val
    else:
        return val
        
class dllFunction(object):
    def __init__(self, name, args = [], argnames = [], lib='ATCORE'):
        self.f = getattr(_stdcall_libraries[lib], name)
        self.f.restype = c_int
        self.f.argtypes = [stripMeta(a) for a in args]
        
        self.fargs = args
        self.fargnames = argnames
        self.name = name
        
        self.inp = [not isinstance(a, OUTPUT) for a in args]
        self.in_args = [a for a in args if not isinstance(a, OUTPUT)]
        self.out_args = [a for a in args if isinstance(a, OUTPUT)]
        
        self.buf_size_arg_pos = -1
        for i in range(len(self.in_args)):
            if isinstance(self.in_args[i], _OUTSTRLEN):
                self.buf_size_arg_pos = i
        
        ds = name + '\n\nArguments:\n===========\n'
        for i in range(len(args)):
            an = ''
            if i <len(argnames):
                an = argnames[i]
            ds += '\t%s\t%s\n' % (args[i], an)
        
        self.f.__doc__ = ds
        
    def __call__(self, *args):
        ars = []
        i = 0
        ret = []

        if self.buf_size_arg_pos >= 0:
            bs = args[self.buf_size_arg_pos]
        else:
            bs = 255
        
        for j in range(len(self.inp)):
            if self.inp[j]: #an input
                ars.append(args[i])
                i+=1
            else: #an output
                r, ar = self.fargs[j].getVar(bs)
                ars.append(ar)
                ret.append(r)
                #print r, r._type_
            
        #print ars
        res = self.f(*ars)
        #print res
        
        if not res == AT_SUCCESS:
            if res == AT_ERR_TIMEDOUT or res == AT_ERR_NODATA:
                #handle timeouts as a special case, as we expect to get them
                raise TimeoutError(self.name, res)
            else:
                raise CameraError(self.name, res)
        
        if len(ret) == 0:
            return None
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
        
    
        
        
def dllFunc(name, args = [], argnames = [], lib='ATCORE'):
    f = dllFunction(name, args, argnames, lib)    
    globals()[name[3:]] = f

dllFunc('AT_InitialiseLibrary')
dllFunc('AT_FinaliseLibrary')

dllFunc('AT_Open', [c_int, OUTPUT(AT_H)])
dllFunc('AT_Close', [AT_H])

dllFunc('AT_IsImplemented', [AT_H, STRING, OUTPUT(AT_BOOL)])
dllFunc('AT_IsReadable', [AT_H, STRING, OUTPUT(AT_BOOL)])
dllFunc('AT_IsWritable', [AT_H, STRING, OUTPUT(AT_BOOL)])
dllFunc('AT_IsReadOnly', [AT_H, STRING, OUTPUT(AT_BOOL)])

dllFunc('AT_SetInt', [AT_H, STRING, AT_64])
dllFunc('AT_GetInt', [AT_H, STRING, OUTPUT(AT_64)])
dllFunc('AT_GetIntMax', [AT_H, STRING, OUTPUT(AT_64)])
dllFunc('AT_GetIntMin', [AT_H, STRING, OUTPUT(AT_64)])

dllFunc('AT_SetFloat', [AT_H, STRING, c_double])
dllFunc('AT_GetFloat', [AT_H, STRING, OUTPUT(c_double)])
dllFunc('AT_GetFloatMax', [AT_H, STRING, OUTPUT(c_double)])
dllFunc('AT_GetFloatMin', [AT_H, STRING, OUTPUT(c_double)])

dllFunc('AT_SetBool', [AT_H, STRING, AT_BOOL])
dllFunc('AT_GetBool', [AT_H, STRING, OUTPUT(AT_BOOL)])

dllFunc('AT_SetEnumerated', [AT_H, STRING, c_int])
dllFunc('AT_SetEnumeratedString', [AT_H, STRING, STRING])
dllFunc('AT_GetEnumerated', [AT_H, STRING, OUTPUT(c_int)])
dllFunc('AT_GetEnumeratedCount', [AT_H, STRING, OUTPUT(c_int)])
dllFunc('AT_IsEnumeratedIndexAvailable', [AT_H, STRING, c_int, OUTPUT(AT_BOOL)])
dllFunc('AT_IsEnumeratedIndexImplemented', [AT_H, STRING, c_int, OUTPUT(AT_BOOL)])
dllFunc('AT_GetEnumeratedString', [AT_H, STRING, c_int, OUTSTRING, OUTSTRLEN])

dllFunc('AT_SetEnumIndex', [AT_H, STRING, c_int])
dllFunc('AT_SetEnumString', [AT_H, STRING, STRING])
dllFunc('AT_GetEnumIndex', [AT_H, STRING, OUTPUT(c_int)])
dllFunc('AT_GetEnumCount', [AT_H, STRING, OUTPUT(c_int)])
dllFunc('AT_IsEnumIndexAvailable', [AT_H, STRING, c_int, OUTPUT(AT_BOOL)])
dllFunc('AT_IsEnumIndexImplemented', [AT_H, STRING, c_int, OUTPUT(AT_BOOL)])
dllFunc('AT_GetEnumStringByIndex', [AT_H, STRING, c_int, OUTSTRING, OUTSTRLEN])

dllFunc('AT_Command', [AT_H, POINTER(AT_WC)])

dllFunc('AT_SetString', [AT_H, STRING, STRING])
dllFunc('AT_GetString', [AT_H, STRING, OUTSTRING, OUTSTRLEN])
dllFunc('AT_GetStringMaxLength', [AT_H, STRING, OUTPUT(c_int)])

dllFunc('AT_QueueBuffer', [AT_H, POINTER(AT_U8), c_int])
dllFunc('AT_WaitBuffer', [AT_H, OUTPUT(POINTER(AT_U8)), OUTPUT(c_int), c_uint])
dllFunc('AT_Flush', [AT_H])

dllFunc('AT_RegisterFeatureCallback', [AT_H, POINTER(AT_WC), CALLBACKTYPE, c_void_p])
dllFunc('AT_UnregisterFeatureCallback', [AT_H, POINTER(AT_WC), CALLBACKTYPE, c_void_p])
#####################################
##Utility library (for unpacking etc ...)
dllFunc('AT_InitialiseUtilityLibrary', lib='ATUTIL')
dllFunc('AT_FinaliseUtilityLibrary', lib='ATUTIL')
dllFunc('AT_ConvertBuffer', [POINTER(AT_U8), POINTER(AT_U8), AT_64, AT_64, AT_64, STRING, STRING], lib='ATUTIL')
dllFunc('AT_ConvertBufferUsingMetadata', [POINTER(AT_U8), POINTER(AT_U8), AT_64, STRING], lib='ATUTIL')

#Initialize the utility library
InitialiseUtilityLibrary()
