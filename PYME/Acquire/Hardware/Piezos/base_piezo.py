class PiezoBase(object):
    """ Base class for piezos"""
    gui_description = '%s-piezo'
    units_um = 1
    
    def SetServo(self, val=1):
        raise NotImplementedError
    
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        raise NotImplementedError
    
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        raise NotImplementedError
    
    def GetPos(self, iChannel=0):
        raise NotImplementedError
    
    def GetTargetPos(self, iChannel=0):
        raise NotImplementedError
    
    def GetMin(self, iChan=1):
        raise NotImplementedError
    
    def GetMax(self, iChan=1):
        raise NotImplementedError
    
    def GetFirmwareVersion(self):
        raise NotImplementedError
    
    def OnTarget(self):
        raise NotImplementedError
    