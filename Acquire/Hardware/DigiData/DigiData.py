from axDD132x import *
from ctypes import *
from ctypes.wintypes import *

class DigiData:
    def __init__(self):
        #find our device - N.B. will only work for first device in this simplistic implementation
        ifo = DD132X_Info()
        pnError = c_int()
        
        nDevs = DD132X_FindDevices(byref(ifo), 1, byref(pnError))
        if (nDevs < 1):
            raise('No DigiData Device found')
        
        #assuming we found something, open it
        self.hDev = DD132X_OpenDevice(ifo.byAdaptor, ifo.byTarget, byref(pnError))
        
        self.DOSet = 0
        self.AOSet = [0,0]
        
    def __del__(self):
        pnError = c_int()
        
        DD132X_CloseDevice(self.hDev, byref(pnError))
        
    def PutDOValues(self, DOVals):
        pnError = c_int()
        
        DD132X_PutDOValues(self.hDev, DWORD(DOVals), byref(pnError))
        self.DOSet = DOVals #keep a copy so we can do bit toggling
        
    def PutAOValue(self, chan, AOVal):
        pnError = c_int()
        
        DD132X_PutAOValue(self.hDev, UINT(chan), c_short(AOVal), byref(pnError))
        self.AOSet[chan] = AOVal
    def GetDIValues(self):
        pnError = c_int()
        DIVals = DWORD()
        
        DD132X_GetDIValues(self.hDev, byref(DIVals), byref(pnError))
        
        return DIVals.value
    
    def GetAIValue(self, chan):
        pnError = c_int()
        AIVal = c_short()
        
        DD132X_GetAIValue(self.hDev, UINT(chan), byref(AIVal), byref(pnError))
        
        return AIVal.value

    
    #extra fcns
    def GetAOValue(self,chan):
        return self.AOSet[chan]

    def GetDOValues(self):
        return self.DOSet

    def SetDOBit(self, bitNum):
        DOv = self.DOSet | (1 << bitNum)
        self.PutDOValues(DOv)

    def UnsetDOBit(self, bitNum):
        DOv = self.DOSet & ~(1 << bitNum)
        self.PutDOValues(DOv)

    def GetDOBit(self, bitNum):
        return self.DOSet & (1 << bitNum)
        
        
