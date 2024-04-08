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
    
    Notes
    -----
    When recording asynchronous sequences (as are used for, e.g. single molecule localisation imaging), the microscope software logs 
    an event and timestamp when the command to move a stage is given. These events are then used in processing to reconstruct the stage
    position over time and assign positions to individual frames. If the frame rate is faster than (or on the same order as) the
    stage motion, this can lead to frames between the command being issued and the stage reaching it's destination being assigned an
    incorrect position. This can be partially mitigated if the stage is capable of detecting when it is on-target  vs. moving. Stages 
    which are aware of this can log a 'PiezoOnTarget' event using PYME.Acquire.eventLog.logEvent with the actual settling position (in
    micrometers) as the event description. If `PiezoOnTarget` events are detected in a series, frames between the commanded move and the
    on-target event can be excluded from analysis. Note that this is somewhat fragile and there are several caveats with it's use - most notably that 
    this will only work if there is only one piezo class emitting OnTarget events, and if this is the z (focus) piezo. If x and y (or, e.g., phase)
    piezos emit OnTarget events, it will likely preclude sensible analysis of the aquired data. The other major caveat is thread-safety - `logEvent` is
    not gauranteed to be thread safe so calling it from, e.g. a polling thread could cause issues depending on which event backend is in use. 
    
    As a consequence, `PiezoOnTarget` events should really be deprecated and replaced by events where the direction (x,y,z, etc) of the piezo issuing
    the event is discernable from either the event name or payload. In the meantime they should be used with caution.

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
    
    @property
    def effective_pos(self):
        """ Return the effective position of the piezo. This is exclusively for use in simulation,
        and allows for derived classes to simulate drifty piezos, etc. """
        return self.GetPos()
    

class SingleAxisWrapper(PiezoBase):
    """
    Allows use of an axis on a multiaxis stage as a single-axis 
    PiezoBase object for compatibility with e.g. focus-lock code.
    """
    def __init__(self, multiaxis_stage, axis=1):
        super().__init__()
        self.multiaxis_stage = multiaxis_stage
        self.units_um = self.multiaxis_stage.units_um
        self.axis = axis

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        self.multiaxis_stage.MoveTo(self.axis, fPos, bTimeOut)
    
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        self.multiaxis_stage.MoveRel(self.axis, incr, bTimeOut)
    
    def GetPos(self, iChannel=0):
        return self.multiaxis_stage.GetPos(self.axis)
    
    def GetTargetPos(self, iChannel=0):
        return self.multiaxis_stage.GetTargetPos(self.axis)
    
    def GetMin(self, iChan=1):
        return self.multiaxis_stage.GetMin(self.axis)
    
    def GetMax(self, iChan=1):
        return self.multiaxis_stage.GetMax(self.axis)
    
    def GetFirmwareVersion(self):
        return self.multiaxis_stage.GetFirmwareVersion()
    
    def OnTarget(self):
        return self.multiaxis_stage.OnTarget()
    
    def SetServo(self, val=1):
        self.multiaxis_stage.SetServo(val)
