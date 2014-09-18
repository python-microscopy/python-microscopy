# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:54:15 2014

@author: David Baddeley
"""
import Pyro.core
import Pyro.naming
import threading
from PYME.misc.computerName import GetComputerName


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
        p = self.GetPos()
        self.offset = val
        self.MoveTo(0, p)
        
class ServerThread(threading.Thread):
    def __init__(self, basePiezo):
        threading.Thread.__init__(self)
        
        compName = GetComputerName()
        
        Pyro.core.initServer()
        ns=Pyro.naming.NameServerLocator().getNS()

        if not compName in [n[0] for n in ns.list('')]:
            ns.createGroup(compName)        
        
        self.daemon=Pyro.core.Daemon()
        self.daemon.useNameServer(ns)
        
        self.piezo = piezoOffsetProxy(basePiezo)
        
        pname = "%s.Piezo" % compName
        
        #get rid of any previous instance
        try:
            ns.unregister(pname)
        except Pyro.errors.NamingError:
            pass
        
        uri=self.daemon.connect(self.piezo,pname)
        
    def run(self):
        print 'foo'
        #try:
        self.daemon.requestLoop()
        #finally:
        #    daemon.shutdown(True)
                

def getClient(compName = GetComputerName()):
    return Pyro.core.getProxyForURI('PYRONAME://%s.Piezo'%compName)
    
def main():
    '''For testing only'''
    from PYME.Acquire.Hardware.Simulator import fakePiezo
    bp = fakePiezo.FakePiezo(100)
    st = ServerThread(bp)
    print 'foo'
    st.start()
    st.join()
    #st.run()
    #st.daemon.requestLoop()
    print 'bar'
    
if __name__ == '__main__':
    main()