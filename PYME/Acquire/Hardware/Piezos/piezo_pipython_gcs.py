
# Defines a GCSPiezo class which uses the pipython package distributed 
# by Physik Instrumente to initialize and run their stages.
# Tested with a E-727 controller, on pipython 2.9.0.4 (pypi)
# see https://github.com/PI-PhysikInstrumente/PIPython

# GETTING STARTED:
# 1. Install https://github.com/PI-PhysikInstrumente/PIPython
# 2. In a Python shell, import this file and call get_gcs_usb()
# to find your stage (assuming it has the drivers it needs, is turned on, etc.)
# 3. Use the full description string returned by get_gcs_usb() to 
# find the axes names as used by PI by calling
# get_stage_axes(description). You'll want to use this description
# exactly in the next step, as PIPython can be sensitive to e.g.
# ['1', '2'] vs [1, 2], or '['A', 'B'] vs. ['a', 'b'].
# 4. Initialize the stage with GCSPiezo(description, axes) in your
# PYME init script.
# 5: Consider using GCSPiezoThreaded for performance improvements,
# particularly if you are using a multi-axis stage, or multiple GCSPiezos
# in your setup, as updating the position of stages happens in order to
# update the GUI and can slow down PYMEAcquire considerably.


from PYME.Acquire.Hardware.Piezos.base_piezo import PiezoBase
from pipython import GCSDevice, pitools
from pipython import PILogger, WARNING
import numpy as np
import logging
import time
PILogger.setLevel(WARNING)

logger = logging.getLogger(__name__)


def get_gcs_usb():
    """
    returns list of PI devices connected by USB and their serial numbers. Use
    to get str description suitable for initializing GCSPiezo, e.g.
    'PI E-727 Controller SN 0118035989'
    """
    with GCSDevice() as pidev:
        return pidev.EnumerateUSB()

def get_stage_axes(description):
    try:
        pi = GCSDevice()
        pi.ConnectUSB(description)
        return pitools.getaxeslist(pi, None)
    finally:
        pi.CloseConnection()


class GCSPiezo(PiezoBase):
    units_um = 1  # assumes controllers is configured in units of um.
    def __init__(self, description=None, axes=None):
        """
        Parameters
        ----------
        descriptions: str
            description as found by GCS enumerate, e.g. 
            PI E-727 Controller SN 0118035989.
        axes : list
            list of axes if this is a multi-axis controller. e.g. [1, 2, 3]
            for a 3-axis stage. Note some PI firmwares assume ['a', 'b', 'c'],
            or ['X', 'Y', 'Z']. After initialization these will be indexed into
            for method calls, i.e. GetPos(iChannel=0) will use axes[0] for the
            GCS axis descriptor.

        """
        PiezoBase.__init__(self)
        self.pi = GCSDevice()
        self.pi.ConnectUSB(description)
        assert self.pi.IsConnected()

        if axes is None:
            logger.error('NO AXES SPECIFIED. Will try and run with all axes')
            self.axes = pitools.getaxeslist(self.pi, None)
        else:
            self.axes = axes
        
        self._min = {}  # key'd on iChan
        self._max = {}
        # try:
        #     units = self.pi.qPUN(self.axes)
        #     logger.debug('stage units: %s' % [units[axis] for axis in self.axes])
        # except:
        #     pass
        # PI appears to use unicoded um to set units to um, not funcitonal at the moment, but
        # ideally we would remove the unit check/log above and just force to um here.
        # self.pi.PUN(axes, values)

    def SetServo(self, val=1):
        # due to form of SetServo in the baseclass, just set all axes for now
        self.pi.SVO(self.axes, [val for axis in self.axes])
    
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        self.pi.MOV(self.axes[iChannel], fPos)
    
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        """ relative to current target position
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        self.pi.MVR(self.axes[iChannel], incr)
    
    def GetPos(self, iChannel=1):
        try:
            assert np.isscalar(iChannel)
        except:
            raise AssertionError('GetPos only supports single-axis query')
        axis = self.axes[iChannel]
        return self.pi.qPOS([axis])[axis]
    
    def GetTargetPos(self, iChannel=1):
        # assume that target pos = current pos. Over-ride in derived class if possible
        # axis = self.axes[iChannel]
        return self.GetPos(iChannel)
    
    def GetMin(self, iChan=1):
        """
        Get lower limits ("soft limits")
        """
        try:
            assert np.isscalar(iChan)
        except:
            raise AssertionError('GetMin only supports single-axis query')

        try:
            return self._min[iChan]
        except KeyError:
            logger.debug('Fetching %s axis min' % iChan)
            axis = self.axes[iChan]
            self._min[iChan] = self.pi.qTMN(axis)[axis]
            return self._min[iChan]
        # return self.pi.qNLM(axes=[iChan])[iChan]
        # qCMN min commandable closed-loop target
    
    def GetMax(self, iChan=1):
        """
        Get upper limits ("soft limits")
        """
        try:
            assert np.isscalar(iChan)
        except:
            raise AssertionError('GetMax only supports single-axis query')
        try:
            return self._max[iChan]
        except KeyError:
            logger.debug('Fetching %s axis max' % iChan)
            axis = self.axes[iChan]
            self._max[iChan] = self.pi.qTMX(axis)[axis]
            return self._max[iChan]
    
    def GetFirmwareVersion(self):
        raise NotImplementedError
    
    def OnTarget(self, axes=None):
        """
        For multiaxis stages, query with None reports for all of them
        """
        return all(self.pi.qONT(axes))
    
    def close(self):
        self.pi.CloseConnection()


import threading
from PYME.Acquire.eventLog import logEvent
class GCSPiezoThreaded(PiezoBase):
    units_um = 1  # assumes controllers is configured in units of um.
    def __init__(self, description=None, axes=None, update_rate=0.01, startup=True,
                 stages=None, refmodes=None, servostates=True, controlmodes=None, joystick=None):
        """
        Parameters
        ----------
        descriptions: str
            description as found by GCS enumerate, e.g. 
            PI E-727 Controller SN 0118035989.
        axes : list
            list of axes if this is a multi-axis controller. e.g. ['1', '2', '3']
            for a 3-axis stage. Note some PI firmwares assume ['a', 'b', 'c'],
            or ['X', 'Y', 'Z']. After initialization these will be indexed into
            for method calls, i.e. GetPos(iChannel=0) will use axes[0] for the
            GCS axis descriptor.
        update_rate : float
            number of seconds pause between threaded polling of position / 
            on-targets
        startup: boolean to use pitools.startup function to startup device

        next keyword parameters below are passed to the pitools.startup command:
        
        stages: Name of stage(s) to initialize as string or list (not tuple!) or 'None' to skip.
        refmodes: Referencing command(s) as string (for all stages) or list (not tuple!) or 'None' to skip.
                     Please see the manual of the controller for the valid reference procedure.
                     'refmodes' can be:
                         'FRF': References the stage using the reference position.
                         'FNL': References the stage using the negative limit switch.
                         'FPL': References the stage using the positive limit switch.
                         'POS': Sets the current position of the stage to 0.0.
                         'ATZ': Runs an auto zero procedure.
        servostates: Desired servo state(s) as Boolean (for all stages) or dict {axis : state} or 'None' to skip.
                        For controllers with GCS 3.0 syntax:
                            If True the axis is switched to control mode 0x2.
                            If False the axis is switched to control mode 0x1.
        controlmodes: Only valid for controllers with GCS 3.0 syntax!
                         Switch the axis to the given control mode:
                             int (for all axes) or dict {axis : controlmode} or 'None' to skip.
                         To skip any control mode switch make sure the servostate is also 'None'!
                         If 'controlmodes' is set (not 'None') the parameter 'servostates' is ignored.

        joystick: should be a joystick object or None if no joystick; needs to support a few standard methods
        """
        PiezoBase.__init__(self)
        self.pi = GCSDevice()
        self.pi.ConnectUSB(description)
        assert self.pi.IsConnected()
        self._lock = threading.Lock()

        if axes is None:
            logger.error('NO AXES SPECIFIED. Will try and run with all axes')
            self.axes = pitools.getaxeslist(self.pi, None)
        else:
            self.axes = axes

        # startup device with supplied per axes parameters
        if startup:
            pitools.startup(self.pi, stages=stages, refmodes=refmodes,
                            servostates=servostates, controlmodes=controlmodes)
            if not refmodes is None: # if any referencing is carried out wait to finish
                pitools.waitonreferencing(self.pi, None, timeout=120) # for now 2 min timeout

        self._min = [pitools.getmintravelrange(self.pi, axis)[axis] for axis in self.axes]
        self._max = [pitools.getmaxtravelrange(self.pi, axis)[axis] for axis in self.axes]

        # before the loop is active, need to query positions explicitly to make self.positions valid
        self.positions = np.array([self.pi.qPOS([axis])[axis] for axis in self.axes])
        self.joystick = joystick
        if self.joystick is not None:
            self.joystick.init(self) # the joystick object should have an init method
            self.disable_joystick() # this includes enabling updating_ontarget
        else:
            self.enable_updating_ontarget()

        self._update_rate = update_rate
        self._start_loop()

    def enable_joystick(self):
        if self.joystick is None:
            return
        self.disable_updating_ontarget()
        with self._lock:
            self.joystick.enablecommands()
        self._joystick_enabled = True

    def disable_joystick(self):
        if self.joystick is None:
            return
        with self._lock:
            self.joystick.disablecommands()
        self.enable_updating_ontarget()
        self._joystick_enabled = False

    def SetServo(self, val=1):
        with self._lock:
            # due to form of SetServo in the baseclass, just set all axes for now
            self.pi.SVO(self.axes, [val for axis in self.axes])
    
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        self.target_positions[iChannel] = fPos
        # self.pi.MOV(self.axes[iChannel], fPos)
    
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        """ relative to current target position
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        new_pos = self.target_positions[iChannel] + incr
        self.target_positions[iChannel] = new_pos
        # self.pi.MVR(self.axes[iChannel], incr)
    
    def GetPos(self, iChannel=1):
        try:
            assert np.isscalar(iChannel)
        except:
            raise AssertionError('GetPos only supports single-axis query')
        return self.positions[iChannel]
        # axis = self.axes[iChannel]
        # return self.pi.qPOS([axis])[axis]
    
    def GetTargetPos(self, iChannel=1):
        # assume that target pos = current pos. Over-ride in derived class if possible
        # axis = self.axes[iChannel]
        # return self.GetPos(iChannel)
        return self.target_positions[iChannel]
    
    def GetMin(self, iChan=1):
        """
        Get lower limits ("soft limits")
        """
        try:
            assert np.isscalar(iChan)
        except:
            raise AssertionError('GetMin only supports single-axis query')
        
        return self._min[iChan]
    
    def GetMax(self, iChan=1):
        """
        Get upper limits ("soft limits")
        """
        try:
            assert np.isscalar(iChan)
        except:
            raise AssertionError('GetMax only supports single-axis query')
        
        return self._max[iChan]
    
    def GetFirmwareVersion(self):
        if self.pi.HasqVER():
            ver = self.pi.qVER() # under lock?
            verinfo = ver.strip()
        else:
            verinfo = 'Controller has no version info'

        return verinfo
    
    def OnTarget(self):
        """
        For multiaxis stages, query with None reports for all of them
        """
        return self._all_on_target
    
    def close(self):
        self.loop_active = False
        with self._lock:
            self.pi.CloseConnection()
    
    def _start_loop(self):
        self.loop_active = True
        self.tloop = threading.Thread(target=self._Loop)
        self.tloop.daemon=True
        self.tloop.start()

    def enable_updating_ontarget(self):
        # first reset the relevant variables
        self.target_positions = np.copy(self.positions)
        self._last_target_positions = np.copy(self.positions)
        self._all_on_target = True
        self._on_target = np.asarray([True for axis in self.axes])
        # now also switch the flag
        self._updating_ontarget = True

    def disable_updating_ontarget(self): # does this need a lock, any race conditions?
        self._updating_ontarget = False

    def _Loop(self):
        while self.loop_active:
            time.sleep(self._update_rate) # we should sleep while NOT holding the lock so that others can grab it if needed
            with self._lock:
                try:
                    # check position
                    for ind, axis in enumerate(self.axes):
                        # does this need a lock?
                        self.positions[ind] = self.pi.qPOS([axis])[axis]
                    
                    if not self._updating_ontarget:
                        continue # skip below if not currently in update target mode (e.g. when joystick active)
                    
                    # update ontarget
                    old_on_target = np.copy(self._on_target)
                    on_targets = pitools.ontarget(self.pi, None)
                    self._on_target = np.asarray([on_targets[axis] for axis in self.axes])
                    if not np.all(self._on_target == old_on_target):
                        # FIXME - something to log which axis would be cool?
                        logEvent('PiezoOnTarget', '%s' % self.positions, time.time())
                        self._all_on_target = np.all(self._on_target)
                    targets_matched = np.isclose(self.target_positions, self._last_target_positions)
                    if all(targets_matched):
                        self._all_on_target = True
                    else:
                        self._all_on_target = False
                        for ind, matched in enumerate(targets_matched):
                            if not matched:
                                new_pos = np.clip(self.target_positions[ind], self._min[ind], self._max[ind])
                                self.pi.MOV(self.axes[ind], new_pos)
                                self._on_target[ind] = False
                        
                        self._last_target_positions = np.copy(self.target_positions)
                
                except Exception as e:
                    logger.error(str(e))
                
        logger.debug('exiting')
   

# a piezo_pipython_gcs compatible joystick class
# - provides Enable() and IsEnabled() methods for use by PYME position ui and scope object
# - an init() method to set the joystick parameters via GCS commands and register the gcspiezo "parent" instance
# - an enablecommands() method of GCS commands that will be used by the gcspiezo "parent" object to enable the joystick
# - a disablecommands() method of GCS commands that will be used by the gcspiezo "parent" object to disable the joystick

# example usage:
#    from PYME.Acquire.Hardware.Piezos.piezo_pipython_gcs import GCSPiezoThreaded
#    from PYMEcs.Acquire.Hardware.Piezos.joystick_c867_digital import DigitalJoystick
#    scope.stage = GCSPiezoThreaded('PI C-867 Piezomotor Controller SN 0122013807', axes=['1', '2'],
#                                   refmodes='FRF',joystick=DigitalJoystick())

class JoystickBase(object):
    def __init__(self):
        self.gcspiezo = None
        self._initialised = False

    # this method needs to be tweaked to the specific controller and joystick model in derived class
    def init(self,gcspiezo):
        # register the piezo device
        self.gcspiezo = gcspiezo # needs to be retained in derived class implementation
        
        # further initialisation code should go below (e.g. GCS commands)
        # i.e., this method needs to be implemented in a derived class
        
        # when completed needs to set the initialised flag: self._initialised = True
        
        raise NotImplementedError('Needs to be implemented in derived class.')

    def Enable(self, enabled = True):
        self.check_initialised()
        if not self.IsEnabled() == enabled:
            if enabled:
                self.gcspiezo.enable_joystick()
            else:
                self.gcspiezo.disable_joystick()

    def IsEnabled(self):
        self.check_initialised()
        return self.gcspiezo._joystick_enabled

    def check_initialised(self):
        if not self._initialised:
            raise RuntimeError("joystick must be initialised before using these methods")

    # GCS commands to enable the joystick
    def enablecommands(self): # this method is supposed to be used from the parent gcspiezo object
        #    e.g. self.gcspiezo.pi.HIN(self.gcspiezo.axes,[True for axis in self.gcspiezo.axes])
        raise NotImplementedError('Needs to be implemented in derived class.')

    # GCS commands to enable the joystick
    def disablecommands(self): # this method is supposed to be used from the parent gcspiezo object
        #    e.g. self.gcspiezo.pi.HIN(self.gcspiezo.axes,[False for axis in self.gcspiezo.axes])
        raise NotImplementedError('Needs to be implemented in derived class.')
