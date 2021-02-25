class PiezoBase(object):
    """
    Base class for all stages

    Attributes (can be over-ridden in derived classes)
    --------------------------------------------------
    units_um: float
        multiplier to get from piezo units to micrometers. For a stage internally using millimeters, this would be
        1000, while a stage internally using nanometers would use 0.001, etc ... Exists for compatibility purposes for
        old stages which had variable units. New stage implementations should try and stick to um and leave this as 1.
    gui_description: str
        Template for a display name. Used together with the `axis` argument to PYME.Acquire.microscope.register_piezo to 
        generate a suitable display name for the stage. Can be over-ridden for asthetic purposes in e.g. stepper motors which
        use PiezoBase as a base, so their controls do not say piezo.

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
    
