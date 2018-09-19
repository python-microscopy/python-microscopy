import numpy as np
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

from .uc480 import *

class camera(HCAM):
    def __init__(self,camera_id=0):
        #self.id = camera_id
        HCAM.__init__(self,0)
        self.h = CALL('InitCamera',ctypes.byref(self),HWND(0))
        self.width = 1024
        self.height = 768
        self.data = np.zeros((self.height,self.width),dtype=np.int8)
        #return None

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
