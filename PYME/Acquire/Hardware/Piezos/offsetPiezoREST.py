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
import threading

class OffsetPiezo(PiezoBase):
    """
    basePiezo position - offset = OffsetPiezo position
    """
    def __init__(self, basePiezo):
        self.basePiezo = basePiezo
        self.offset = 0
        # webframework.APIHTTPServer handles requests in separate threads
        self._move_lock = threading.Lock()
        
    @property
    def units_um(self):
        return self.basePiezo.units_um
        
    def SetServo(self,val = 1):
        return self.basePiezo.SetServo(int(bool(val)))
        
    @webframework.register_endpoint('/MoveTo', output_is_json=False)
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        with self._move_lock:
            p = self.basePiezo.MoveTo(int(iChannel), float(fPos) + self.offset, 
                                      bool(bTimeOut))
        return p

    @webframework.register_endpoint('/MoveRel', output_is_json=False)
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        with self._move_lock:
            p = self.basePiezo.MoveRel(int(iChannel), float(incr), 
                                       bool(bTimeOut))
        return p

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
        # both gettarget and moveto account for offset, so make sure we only apply the change once
        with self._move_lock:
            pos = self.GetTargetPos(0)
            self.offset = float(offset)
            # self.MoveTo(0, pos)
            self.basePiezo.MoveTo(0, pos + self.offset, True)

    @webframework.register_endpoint('/CorrectOffset', output_is_json=False)
    def CorrectOffset(self, correction):
        # both gettarget and moveto account for offset, so make sure we only apply the change once
        correction = float(correction)
        with self._move_lock:
            target = self.GetTargetPos(0)
            # correct the offset; positive means push base pos higher than offsetpiezo pos
            correction = max(min(self.basePiezo.max_travel - (target + self.offset), 
                                 correction), -(target + self.offset))
            self.offset += correction
            # self.MoveTo(0, target)  # move the base piezo to correct position
            self.basePiezo.MoveTo(0, target + self.offset, True)

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
    def LogFocusCorrection(self, offset):
        import wx
        wx.CallAfter(eventLog.logEvent, 'PiezoOffsetUpdate', '%3.4f' % float(offset))
    
    @webframework.register_endpoint('/GetMaxOffset', output_is_json=False)
    def GetMaxOffset(self):
        return self.basePiezo.max_travel - self.GetTargetPos()
    
    @webframework.register_endpoint('/GetMinOffset', output_is_json=False)
    def GetMinOffset(self):
        return - self.GetTargetPos()


import requests
class OffsetPiezoClient(PiezoBase):
    def __init__(self, host='127.0.0.1', port=9797, name='offset_piezo'):
        self.host = host
        self.port = port
        self.name = name
        
        self.urlbase = 'http://%s:%d' % (host, port)#,self.name)
        self._session = requests.Session()

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        return self._session.get(self.urlbase + '/MoveTo?iChannel=%d&fPos=%3.3f' % (iChannel, fPos))

    def MoveRel(self, iChannel, incr, bTimeOut=True):
        return self._session.get(self.urlbase + '/MoveRel?iChannel=%d&incr=%3.3f' % (iChannel, incr))

    def GetPos(self, iChannel=0):
        res = self._session.get(self.urlbase + '/GetPos?iChannel=%d' % (iChannel, ))
        return float(res.json())

    def GetTargetPos(self, iChannel=0):
        res = self._session.get(self.urlbase + '/GetTargetPos?iChannel=%d' % (iChannel,))
        return float(res.json())

    def GetMin(self, iChan=1):
        res = self._session.get(self.urlbase + '/GetMin?iChan=%d' % (iChan,))
        return float(res.json())

    def GetMax(self, iChan=1):
        res = self._session.get(self.urlbase + '/GetMax?iChan=%d' % (iChan,))
        return float(res.json())

    def GetFirmwareVersion(self):
        res = self._session.get(self.urlbase + '/GetFirmwareVersion')
        return str(res.json())

    def GetOffset(self):
        res = self._session.get(self.urlbase + '/GetOffset')
        return float(res.json())

    def SetOffset(self, offset):
        return self._session.get(self.urlbase + '/SetOffset?offset=%3.3f' % (offset))

    def CorrectOffset(self, shim):
        return self._session.get(self.urlbase + '/CorrectOffset?correction=%3.3f' % (shim))

    def LogShifts(self, dx, dy, dz, active=True):
        res = self._session.get(self.urlbase + '/LogShifts?dx=%3.3f&dy=%3.3f&dz=%3.3f&active=%d'% (dx, dy, dz, active))
        return float(res.json())

    def OnTarget(self):
        res = self._session.get(self.urlbase + '/OnTarget')
        return bool(res.json())

    def LogFocusCorrection(self, offset):
        self._session.get(self.urlbase + '/LogFocusCorrection?offset=%3.3f' % (offset,))
    
    def GetMaxOffset(self):
        res = self._session.get(self.urlbase + '/GetMaxOffset')
        return float(res.json())
    
    def GetMinOffset(self):
        res = self._session.get(self.urlbase + '/GetMinOffset')
        return float(res.json())

def generate_offset_piezo_server(offset_piezo_base_class):
    """
    Class factory to return class which inherits from the desired style of OffsetPiezo

    Parameters
    ----------
    offset_piezo_base_class: class
        class object for desired offset piezo, either OffsetPiezo or TargetOwningOffsetPiezo

    Returns
    -------
    OffsetPiezoServer:
        class which can be instantiated to provide a server managing the desired type of offset piezo.

    """
    class OffsetPiezoServer(webframework.APIHTTPServer, offset_piezo_base_class):
        def __init__(self, basePiezo, port=9797):
            offset_piezo_base_class.__init__(self, basePiezo)

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

    return OffsetPiezoServer

class TargetOwningOffsetPiezo(OffsetPiezo):
    """
    The standard OffsetPiezo maintains a target position by adding an offset to the base_piezo target position. This is
    problematic because the base_piezo target position gets changed everytime it moves, so race conditions are possible
    and the target will eventually drift over time even if you'd like it to stay constant.

    This offset piezo owns its target position, so it changes with self.MoveTo and self.MoveRel, not with
    self.basePiezo.MoveTo and self.basePiezo.MoveRel.

    basePiezo position - offset = OffsetPiezo position
    """
    def __init__(self, base_piezo):
        OffsetPiezo.__init__(self, base_piezo)
        try:
            self._target_position = self.basePiezo.GetTargetPos(0)
        except (AttributeError, NotImplementedError):
            self._target_position = self.basePiezo.GetPos(0)

    @webframework.register_endpoint('/MoveTo', output_is_json=False)
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        with self._move_lock:
            self._target_position = float(fPos)
            p = self.basePiezo.MoveTo(int(iChannel), 
                                      self._target_position + self.offset, 
                                      bool(bTimeOut))
            return p

    @webframework.register_endpoint('/MoveRel', output_is_json=False)
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        with self._move_lock:
            self._target_position += float(incr)
            print('here - moving to %f' % self._target_position)
            p = self.basePiezo.MoveTo(int(iChannel), 
                                      self._target_position + self.offset, 
                                      bool(bTimeOut))
            return p
    
    @webframework.register_endpoint('/GetPos', output_is_json=False)
    def GetPos(self, iChannel=0):
        return self.basePiezo.GetPos(int(iChannel)) - self.offset

    @webframework.register_endpoint('/GetTargetPos', output_is_json=False)
    def GetTargetPos(self, iChannel=0):
        return self._target_position

                
    
    
def main():
    """For testing only"""
    from PYME.Acquire.Hardware.Simulator import fakePiezo
    bp = fakePiezo.FakePiezo(100)
    st = generate_offset_piezo_server(OffsetPiezo)(bp)
    st._server_thread.join()
    
if __name__ == '__main__':
    main()