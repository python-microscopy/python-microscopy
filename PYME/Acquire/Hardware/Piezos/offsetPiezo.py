# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:54:15 2014

@author: David Baddeley
"""
import Pyro.core
import Pyro.naming
import threading

class piezoOffsetProxy(Pyro.core.ObjBase):    
    def __init__(self, basePiezo):
        Pyro.core.ObjBase.__init__(self)
        self.basePiezo = basePiezo
        self.offset = 0

    def ReInit(self):
        return self.basePiezo.ReInit()
        
    def SetServo(self,val = 1):
        return self.basePiezo.SetServo(val)
        
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        return self.basePiezo.MoveTo(iChannel, fPos + self.offset, bTimeOut)
            
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        return self.basePiezo.MoveRel(iChannel, incr, bTimeOut)

    def GetPos(self, iChannel=0):
        return self.basePiezo.GetPos(iChannel) - self.offset
        
    def GetControlReady(self):
        return self.basePiezo.GetControlReady()
         
    def GetChannelObject(self):
        return self.basePiezo.GetChannelObject()
        
    def GetChannelPhase(self):
        return self.basePiezo.GetChannelPhase()
        
    def GetMin(self,iChan=1):
        return self.basePiezo.GetMin(iChan)
        
    def GetMax(self, iChan=1):
        return self.basePiezo.GetMax(iChan)
        
    def GetFirmwareVersion(self):
        return self.basePiezo.GetFirmwareVersion()
        
    def GetOffset(self):
        return self.offset
        
    def SetOffset(self, val):
        self.offset = val
        
class ServerThread(threading.Thread):
    def __init__(self):
        Pyro.core.initServer()
        ns=Pyro.naming.NameServerLocator().getNS()
        daemon=Pyro.core.Daemon()
        daemon.useNameServer(ns)
        
        dd = RemoteDigiData()
        uri=daemon.connect(dd,"DigiData")
        
        try:
            daemon.requestLoop()
        finally:
            daemon.shutdown(True)