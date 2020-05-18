class PiezoBase(object):
    """
    Base class for all stages

    Parameters
    ----------
    units_um: float
        multiplier to get from piezo units to micrometers. For a stage internally using millimeters, this would be
        1000, while a stage internally using nanometers would use 0.001, etc..
    gui_description: str
        [optional/deprecated] abbreviated description of stage axis, e.g. 'x-piezo'. This is typically handled
        downstream in PYME.Acquire.microscope.register_piezo

    """
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
        # assume that target pos = current pos. Over-ride in derived class if possible
        return self.GetPos(iChannel)
    
    def GetMin(self, iChan=1):
        raise NotImplementedError
    
    def GetMax(self, iChan=1):
        raise NotImplementedError
    
    def GetFirmwareVersion(self):
        raise NotImplementedError
    
    def OnTarget(self):
        raise NotImplementedError
    