# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:54:15 2014

@author: David Baddeley
"""
import Pyro.core
import Pyro.naming
import threading
from PYME.misc.computerName import GetComputerName

from PYME.Acquire import eventLog
import time

from .base_piezo import PiezoBase

class piezoOffsetProxy(PiezoBase, Pyro.core.ObjBase):
    def __init__(self, basePiezo):
        Pyro.core.ObjBase.__init__(self)
        self.basePiezo = basePiezo
        self.offset = 0
        
    @property
    def units_um(self):
        return self.basePiezo.units_um
        
    def SetServo(self,val = 1):
        return self.basePiezo.SetServo(val)
        
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        return self.basePiezo.MoveTo(iChannel, fPos + self.offset, bTimeOut)
            
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        return self.basePiezo.MoveRel(iChannel, incr, bTimeOut)

    def GetPos(self, iChannel=0):
        return self.basePiezo.GetPos(iChannel) - self.offset
        
    def GetTargetPos(self, iChannel=0):
        return self.basePiezo.GetTargetPos(iChannel) - self.offset
        
         
    def GetMin(self,iChan=1):
        return self.basePiezo.GetMin(iChan)
        
    def GetMax(self, iChan=1):
        return self.basePiezo.GetMax(iChan)
        
    def GetFirmwareVersion(self):
        return self.basePiezo.GetFirmwareVersion()
        
    def GetOffset(self):
        return self.offset
        
    def SetOffset(self, val):
        p = self.GetTargetPos()
        self.offset = val
        self.MoveTo(0, p)
        
    def LogShifts(self, dx, dy, dz, active=True):
        import wx
        #eventLog.logEvent('ShiftMeasure', '%3.4f, %3.4f, %3.4f' % (dx, dy, dz))
        wx.CallAfter(eventLog.logEvent, 'ShiftMeasure', '%3.4f, %3.4f, %3.4f' % (dx, dy, dz))
        wx.CallAfter(eventLog.logEvent, 'PiezoOffset', '%3.4f, %d' % (self.GetOffset(), active))
        
    def OnTarget(self):
        return self.basePiezo.OnTarget()

    def LogFocusCorrection(self,offset):
        import wx
        wx.CallAfter(eventLog.logEvent, 'PiezoOffsetUpdate', '%3.4f' %offset)

    # @property
    # def lastPos(self):
    #     return self.basePiezo.lastPos - self.offset

    # @lastPos.setter
    # def lastPos(self,val):
    #     self.basePiezo.lastPos = val
        
        
class ServerThread(threading.Thread):
    def __init__(self, basePiezo):
        threading.Thread.__init__(self)

        import socket
        ip_addr = socket.gethostbyname(socket.gethostname())
        
        compName = GetComputerName()
        
        Pyro.core.initServer()

        pname = "%s.Piezo" % compName
        
        try:
            from PYME.misc import pyme_zeroconf 
            ns = pyme_zeroconf.getNS()
        except:
            ns=Pyro.naming.NameServerLocator().getNS()

            if not compName in [n[0] for n in ns.list('')]:
                ns.createGroup(compName)

            #get rid of any previous instance
            try:
                ns.unregister(pname)
            except Pyro.errors.NamingError:
                pass        
        
        self.daemon=Pyro.core.Daemon(host = ip_addr)
        self.daemon.useNameServer(ns)
        
        #self.piezo = piezoOffsetProxy(basePiezo)
        self.piezo = basePiezo
        
        #pname = "%s.Piezo" % compName
        
        
        
        uri=self.daemon.connect(self.piezo,pname)
        
    def run(self):
        #print 'foo'
        #try:
        self.daemon.requestLoop()
        #finally:
        #    daemon.shutdown(True)
        
    def cleanup(self):
        print('Shutting down Offset Piezo Server')
        self.daemon.shutdown(True)
                

def getClient(compName = GetComputerName()):
    #try:
    from PYME.misc import pyme_zeroconf 
    ns = pyme_zeroconf.getNS()
    time.sleep(3)
    #print ns.list()
    URI = ns.resolve('%s.Piezo' % compName)
    #except:
    #    URI ='PYRONAME://%s.Piezo'%compName

    #print URI

    return Pyro.core.getProxyForURI(URI)
    
    
def main():
    """For testing only"""
    from PYME.Acquire.Hardware.Simulator import fakePiezo
    bp = fakePiezo.FakePiezo(100)
    st = ServerThread(bp)
    #print 'foo'
    st.start()
    st.join()
    #st.run()
    #st.daemon.requestLoop()
    #print 'bar'
    
if __name__ == '__main__':
    main()