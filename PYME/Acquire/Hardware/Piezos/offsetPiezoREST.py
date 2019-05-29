# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:54:15 2014

@author: David Baddeley
"""
import threading
from PYME.misc.computerName import GetComputerName

from PYME.Acquire import eventLog
import time

from PYME.util import webframework

from .base_piezo import PiezoBase

class OffsetPiezo(PiezoBase):
    def __init__(self, basePiezo):
        self.basePiezo = basePiezo
        self.offset = 0
        
    @property
    def units_um(self):
        return self.basePiezo.units_um
        
    def SetServo(self,val = 1):
        return self.basePiezo.SetServo(int(bool(val)))
        
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        return self.basePiezo.MoveTo(int(iChannel), float(fPos + self.offset), bool(bTimeOut))
            
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        return self.basePiezo.MoveRel(int(iChannel), float(incr), bool(bTimeOut))

    def GetPos(self, iChannel=0):
        return self.basePiezo.GetPos(int(iChannel)) - self.offset
        
    def GetTargetPos(self, iChannel=0):
        return self.basePiezo.GetTargetPos(int(iChannel)) - self.offset
        
         
    def GetMin(self,iChan=1):
        return self.basePiezo.GetMin(int(iChan))
        
    def GetMax(self, iChan=1):
        return self.basePiezo.GetMax(int(iChan))
        
    def GetFirmwareVersion(self):
        return self.basePiezo.GetFirmwareVersion()
        
    def GetOffset(self):
        return self.offset
        
    def SetOffset(self, val):
        p = self.GetTargetPos()
        self.offset = float(val)
        self.MoveTo(0, p)
        
    def LogShifts(self, dx, dy, dz, active=True):
        import wx
        #eventLog.logEvent('ShiftMeasure', '%3.4f, %3.4f, %3.4f' % (dx, dy, dz))
        wx.CallAfter(eventLog.logEvent, 'ShiftMeasure', '%3.4f, %3.4f, %3.4f' % (float(dx), float(dy), float(dz)))
        wx.CallAfter(eventLog.logEvent, 'PiezoOffset', '%3.4f, %d' % (self.GetOffset(), active))
        
    def OnTarget(self):
        return self.basePiezo.OnTarget()

    def LogFocusCorrection(self,offset):
        import wx
        wx.CallAfter(eventLog.logEvent, 'update offset', '%3.4f' % float(offset))


import requests
class OffsetPiezoClient(PiezoBase):
    def __init__(self, host='127.0.0.1', port=9797, name='offset_piezo'):
        self.host = host
        self.port = port
        self.name = 'offset_piezo'
        
        self.urlbase = 'http://%s:%d/%s' % (host, port,self.name)

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        return requests.get(self.urlbase + '/MoveTo?iChannel=%d&fPos=&3.3f' % (iChannel, fPos))

    def MoveRel(self, iChannel, incr, bTimeOut=True):
        return requests.get(self.urlbase + '/MoveRel?iChannel=%d&incr=&3.3f' % (iChannel, incr))

    def GetPos(self, iChannel=0):
        res = requests.get(self.urlbase + '/GetPos?iChannel=%d' % (iChannel, ))
        return self.basePiezo.GetPos(int(iChannel)) - self.offset

    def GetTargetPos(self, iChannel=0):
        return self.basePiezo.GetTargetPos(int(iChannel)) - self.offset

    def GetMin(self, iChan=1):
        return self.basePiezo.GetMin(int(iChan))

    def GetMax(self, iChan=1):
        return self.basePiezo.GetMax(int(iChan))

    def GetFirmwareVersion(self):
        return self.basePiezo.GetFirmwareVersion()

    def GetOffset(self):
        return self.offset

    def SetOffset(self, val):
        p = self.GetTargetPos()
        self.offset = float(val)
        self.MoveTo(0, p)

    def LogShifts(self, dx, dy, dz, active=True):
        import wx
        #eventLog.logEvent('ShiftMeasure', '%3.4f, %3.4f, %3.4f' % (dx, dy, dz))
        wx.CallAfter(eventLog.logEvent, 'ShiftMeasure', '%3.4f, %3.4f, %3.4f' % (float(dx), float(dy), float(dz)))
        wx.CallAfter(eventLog.logEvent, 'PiezoOffset', '%3.4f, %d' % (self.GetOffset(), active))

    def OnTarget(self):
        return self.basePiezo.OnTarget()

    def LogFocusCorrection(self, offset):
        import wx
        wx.CallAfter(eventLog.logEvent, 'update offset', '%3.4f' % float(offset))


class ServerThread(threading.Thread):
    def __init__(self, piezo, port=9797):
        threading.Thread.__init__(self)

        self.piezo = piezo
        self.port = port
        
    def run(self):
        import socket
    
        externalAddr = socket.gethostbyname(socket.gethostname())
        server = ScanServer(port=port)
    
        try:
            logger.info('Starting nodeserver on %s:%d' % (externalAddr, port))
            server.serve_forever()
        finally:
            logger.info('Shutting down ...')
            server.shutdown()
            server.server_close()
        
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