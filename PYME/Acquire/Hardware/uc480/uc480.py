__all__ = [ ]

import os
import sys
import textwrap
import numpy as np
from numpy import ctypeslib
import ctypes
import ctypes.util
import ctypes.wintypes
import warnings

from .uc480_h import *
from ctypes.wintypes import BYTE
from ctypes.wintypes import WORD
from ctypes.wintypes import DWORD
from ctypes.wintypes import BOOL
HCAM = ctypes.wintypes.HANDLE
from ctypes.wintypes import HDC
from ctypes.wintypes import HWND
from ctypes.wintypes import INT
from ctypes import Structure
c_char = ctypes.c_byte
c_char_p = ctypes.POINTER(ctypes.c_byte)
c_int_p = ctypes.POINTER(ctypes.c_int)
from ctypes import c_int
IS_CHAR = ctypes.c_byte 

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


if os.name=='nt':
    # UNTESTED: Please report results to http://code.google.com/p/pylibuc480/issues
    libname = 'uc480'
    include_uc480_h = os.environ['PROGRAMFILES']+'\\Thorlabs DCU camera\\Develop\\Include\\uc480.h'
    lib = ctypes.util.find_library(libname)
    if lib is None:
        print('uc480.dll not found')
        lib = libname

		
#libuc480 = ctypes.cdll.LoadLibrary(lib)
#libuc480 = ctypes.WinDLL('uc480_64')
libuc480 = ctypes.WinDLL('ueye_api_64')
if libuc480 is not None:
	uc480_h_name = 'uc480_h'
	try:
		uc480_h = "uc480_h"
		#from uc480_h import *
		#exec 'from %s import *' % (uc480_h_name)
	except ImportError:
		uc480_h = None
	if uc480_h is None:
		assert os.path.isfile(include_uc480_h), repr(include_uc480_h)
		d = {}
		l = ['# This file is auto-generated. Do not edit!']
		error_map = {}
		f = open (include_uc480_h, 'r')
		
		def is_number(s):
			try:
				float(s)
				return True
			except ValueError:
				return False
				
		for line in f.readlines():
			if not line.startswith('#define'): continue
			i = line.find('//')
			words = line[7:i].strip().split(None, 2)
			if len (words)!=2: continue
			name, value = words
			if value.startswith('0x'):
				exec '%s = %s' % (name, value)
				d[name] = eval(value)
				l.append('%s = %s' % (name, value))
			# elif name.startswith('DAQmxError') or name.startswith('DAQmxWarning'):
				# assert value[0]=='(' and value[-1]==')', `name, value`
				# value = int(value[1:-1])
				# error_map[value] = name[10:]
			# elif name.startswith('DAQmx_Val') or name[5:] in ['Success','_ReadWaitMode']:
				# d[name] = eval(value)
				# l.append('%s = %s' % (name, value))
			elif is_number(value):
				d[name] = eval(value)
				l.append('%s = %s' % (name, value))
			elif value.startswith('UC'):
				print(value)
				d[name] = unicode(value[3:-1])
				l.append('%s = unicode("%s")' % (name, value[3:-1]))
			elif d.has_key(value):
				d[name] = d[value]
				l.append('%s = %s' % (name, d[value]))
			else:
				d[name] = value
				l.append('%s = %s' % (name, value))
				pass
		l.append('error_map = %r' % (error_map))
		fn = os.path.join (os.path.dirname(os.path.abspath (__file__)), uc480_h_name+'.py')
		print(('Generating %r' % (fn)))
		f = open(fn, 'w')
		f.write ('\n'.join(l) + '\n')
		f.close()
		print(('Please upload generated file %r to http://code.google.com/p/pylibuc480/issues' % (fn)))
	else:
		pass
		#d = uc480_h.__dict__
	
#	for name, value in d.items():
#		if name.startswith ('_'): continue
#		exec '%s = %r' % (name, value)


# def CHK(return_code, funcname, *args):
    # """
    # Return ``return_code`` while handle any warnings and errors from
    # calling a libuc480 function ``funcname`` with arguments
    # ``args``.
    # """
    # if return_code==0: # call was succesful
        # pass
    # else:
        # buf_size = default_buf_size
        # while buf_size < 1000000:
            # buf = ctypes.create_string_buffer('\000' * buf_size)
            # try:
                # r = libuc480.DAQmxGetErrorString(return_code, ctypes.byref(buf), buf_size)
            # except RuntimeError, msg:
                # if 'Buffer is too small to fit the string' in str(msg):
                    # buf_size *= 2
                # else:
                    # raise
            # else:
                # break
        # if r:
            # if return_code < 0:
                # raise RuntimeError('%s%s failed with error %s=%d: %s'%\
                                       # (funcname, args, error_map[return_code], return_code, repr(buf.value)))
            # else:
                # warning = error_map.get(return_code, return_code)
                # sys.stderr.write('%s%s warning: %s\n' % (funcname, args, warning))                
        # else:
            # text = '\n  '.join(['']+textwrap.wrap(buf.value, 80)+['-'*10])
            # if return_code < 0:
                # raise RuntimeError('%s%s:%s' % (funcname,args, text))
            # else:
                # sys.stderr.write('%s%s warning:%s\n' % (funcname, args, text))
    # return return_code
	
def CALL(name, *args):
	"""
	Calls libuc480 function "name" and arguments "args".
	"""
	funcname = 'is_' + name
	#print name
	func = getattr(libuc480, funcname)
	new_args = []
	for a in args:		
		if isinstance (a, unicode):
			print((name, 'argument',a, 'is unicode'))
			new_args.append (str (a))
		else:
			new_args.append (a)
	r = func(*new_args)
	#print r
  # r = CHK(r, funcname, *new_args)
	return r

		
class camera(HCAM):
	def __init__(self,camera_id=0):
		#self.id = camera_id
		HCAM.__init__(self,0)
		self.h = CALL('InitCamera',ctypes.byref(self),HWND(0))
		self.width = 1024
		self.height = 768		
		self.data = np.zeros((self.height,self.width),dtype=np.int8)
		return None
		
	def ExitCamera(self):
		return CALL('ExitCamera', ctypes.byref(self)) == 0
		
	def SaveImage(self,file):
		return CALL('SaveImage',self,None)
		
	def AllocImageMem(self,width=1024,height=768,bitpixel=8):
#		self.image = np.zeros((height,width),dtype=np.int8)

		self.image = c_char_p()
		self.id = c_int()
		CALL('AllocImageMem',self,c_int(width),c_int(height),c_int(bitpixel),ctypes.byref(self.image),ctypes.byref(self.id))
		print((self.id))
#		CALL('AllocImageMem',self,c_int(width),c_int(height),c_int(bitpixel),self.image.data,ctypes.byref(self.id))
	
	def FreeImageMem (self):
		CALL("FreeImageMem",self,self.image,self.id)
		
	def FreezeVideo(self,wait=IS_WAIT):
		CALL("FreezeVideo",self,INT(wait))
		
	def CopyImageMem(self):

		r = CALL("CopyImageMem",self,self.image,self.id,self.data.ctypes.data)
		if r == -1:
			self.GetError()
			print((self.err))
			print((self.errMessage.value))
		return 

	def GetError(self):
		self.err = ctypes.c_int()
		self.errMessage = ctypes.c_char_p()
		CALL("GetError",self,ctypes.byref(self.err),ctypes.byref(self.errMessage))
		
	def SetImageMem (self):
		CALL("SetImageMem",self,self.image,self.id)
		
	def SetImageSize(self,x=IS_GET_IMAGE_SIZE_X_MAX,y=IS_GET_IMAGE_SIZE_X_MAX):
		print(IS_GET_IMAGE_SIZE_X_MAX)
		CALL("SetImageSize",self,c_int(x),c_int(y))
		
	def SetImagePos(self,x=0,y=0):
		CALL("SetImagePos",self,c_int(x),c_int(y))
		
	def CaptureVideo(self,wait=IS_DONT_WAIT):
		CALL("CaptureVideo",self,c_int(wait))
		
	def SetColorMode(self,color_mode=IS_SET_CM_Y8):
		CALL("SetColorMode",self,c_int(color_mode))
	
	def SetSubSampling(self,mode=IS_SUBSAMPLING_DISABLE):
		CALL("SetSubSampling",self,c_int(mode))
		
	def StopLiveVideo(self,wait=IS_WAIT):
		CALL("StopLiveVideo",self,c_int(wait))
		
	def ExitCamera (self):
		CALL("ExitCamera",self)