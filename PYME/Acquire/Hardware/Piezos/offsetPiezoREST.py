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

import logging
logger = logging.getLogger(__name__)

from PYME.Acquire.Hardware.Piezos.base_piezo import PiezoBase

class OffsetPiezo(PiezoBase):
    def __init__(self, basePiezo):
        self.basePiezo = basePiezo
        self.offset = 0
        
    @property
    def units_um(self):
        return self.basePiezo.units_um
        
    def SetServo(self,val = 1):
        return self.basePiezo.SetServo(int(bool(val)))
        
    @webframework.register_endpoint('/MoveTo', output_is_json=False)
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        return self.basePiezo.MoveTo(int(iChannel), float(fPos) + self.offset, bool(bTimeOut))

    @webframework.register_endpoint('/MoveRel', output_is_json=False)
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        return self.basePiezo.MoveRel(int(iChannel), float(incr), bool(bTimeOut))

    @webframework.register_endpoint('/GetPos', output_is_json=False)
    def GetPos(self, iChannel=0):
        return self.basePiezo.GetPos(int(iChannel)) - self.offset

    @webframework.register_endpoint('/GetTargetPos', output_is_json=False)
    def GetTargetPos(self, iChannel=0):
        return self.basePiezo.GetTargetPos(int(iChannel)) - self.offset

    @webframework.register_endpoint('/GetMin', output_is_json=False)
    def GetMin(self,iChan=1):
        return self.basePiezo.GetMin(int(iChan))

    @webframework.register_endpoint('/GetMax', output_is_json=False)
    def GetMax(self, iChan=1):
        return self.basePiezo.GetMax(int(iChan))
        
    def GetFirmwareVersion(self):
        return self.basePiezo.GetFirmwareVersion()

    @webframework.register_endpoint('/GetOffset', output_is_json=False)
    def GetOffset(self):
        return self.offset

    @webframework.register_endpoint('/SetOffset', output_is_json=False)
    def SetOffset(self, offset):
        p = self.GetTargetPos()
        self.offset = float(offset)
        self.MoveTo(0, p)

    @webframework.register_endpoint('/LogShifts', output_is_json=False)
    def LogShifts(self, dx, dy, dz, active=True):
        import wx
        #eventLog.logEvent('ShiftMeasure', '%3.4f, %3.4f, %3.4f' % (dx, dy, dz))
        wx.CallAfter(eventLog.logEvent, 'ShiftMeasure', '%3.4f, %3.4f, %3.4f' % (float(dx), float(dy), float(dz)))
        wx.CallAfter(eventLog.logEvent, 'PiezoOffset', '%3.4f, %d' % (self.GetOffset(), active))

    @webframework.register_endpoint('/OnTarget', output_is_json=False)
    def OnTarget(self):
        return self.basePiezo.OnTarget()

    @webframework.register_endpoint('/LogFocusCorrection', output_is_json=False)
    def LogFocusCorrection(self,offset):
        import wx
        wx.CallAfter(eventLog.logEvent, 'update offset', '%3.4f' % float(offset))


import requests
class OffsetPiezoClient(PiezoBase):
    def __init__(self, host='127.0.0.1', port=9797, name='offset_piezo'):
        self.host = host
        self.port = port
        self.name = name
        
        self.urlbase = 'http://%s:%d' % (host, port)#,self.name)

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        return requests.get(self.urlbase + '/MoveTo?iChannel=%d&fPos=%3.3f' % (iChannel, fPos))

    def MoveRel(self, iChannel, incr, bTimeOut=True):
        return requests.get(self.urlbase + '/MoveRel?iChannel=%d&incr=%3.3f' % (iChannel, incr))

    def GetPos(self, iChannel=0):
        res = requests.get(self.urlbase + '/GetPos?iChannel=%d' % (iChannel, ))
        return float(res.json())

    def GetTargetPos(self, iChannel=0):
        res = requests.get(self.urlbase + '/GetTargetPos?iChannel=%d' % (iChannel,))
        return float(res.json())

    def GetMin(self, iChan=1):
        res = requests.get(self.urlbase + '/GetMin?iChan=%d' % (iChan,))
        return float(res.json())

    def GetMax(self, iChan=1):
        res = requests.get(self.urlbase + '/GetMax?iChan=%d' % (iChan,))
        return float(res.json())

    def GetFirmwareVersion(self):
        res = requests.get(self.urlbase + '/GetFirmwareVersion')
        return str(res.json())

    def GetOffset(self):
        res = requests.get(self.urlbase + '/GetOffset')
        return float(res.json())

    def SetOffset(self, offset):
        return requests.get(self.urlbase + '/SetOffset?offset=%3.3f' % (offset))

    def LogShifts(self, dx, dy, dz, active=True):
        res = requests.get(self.urlbase + '/LogShifts?dx=%3.3f&dy=%3.3f&dz=%3.3f&active=%d'% (dx, dy, dz, active))
        return float(res.json())

    def OnTarget(self):
        res = requests.get(self.urlbase + '/OnTarget')
        return float(res.json())

    def LogFocusCorrection(self, offset):
        res = requests.get(self.urlbase + '/LogFocusCorrection?ofset=%3.3f' % (offset,))
        return float(res.json())

class OffsetPiezoServer(webframework.APIHTTPServer, OffsetPiezo):
    def __init__(self, basePiezo, port=9797):
        OffsetPiezo.__init__(self, basePiezo)
        
        server_address = ('127.0.0.1', port)
        self.port = port
        
        
        webframework.APIHTTPServer.__init__(self, server_address)
        self.daemon_threads = True
        
        self._server_thread = threading.Thread(target=self._thread_target)
        self._server_thread.daemon_threads = True
        
        self._server_thread.start()

    def _thread_target(self):
        try:
            logger.info('Starting piezo on 127.0.0.1:%d' % (self.port,))
            self.serve_forever()
        finally:
            logger.info('Shutting down ...')
            self.shutdown()
            self.server_close()

                
    
    
def main():
    """For testing only"""
    from PYME.Acquire.Hardware.Simulator import fakePiezo
    bp = fakePiezo.FakePiezo(100)
    st = OffsetPiezoServer(bp)
    st._server_thread.join()
    
if __name__ == '__main__':
    main()