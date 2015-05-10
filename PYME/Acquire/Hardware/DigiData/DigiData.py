#!/usr/bin/python

##################
# DigiData.py
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

from .axDD132x import *
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
        
        
