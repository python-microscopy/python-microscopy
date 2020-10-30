

from PYME.Acquire.Hardware.Piezos.base_piezo import PiezoBase
import threading
import ctypes
import numpy as np
import logging
logger = logging.getLogger(__name__)

# Download the Tango DLLs from https://www.marzhauser.com/nc/en/service/downloads.html?tx_abdownloads_pi1%5Baction%5D=getviewclickeddownload&tx_abdownloads_pi1%5Buid%5D=594
# Place the appropriate version of Tango_DLL.dll for your operating system in C:\Windows\System32.
mazlib = ctypes.WinDLL('Tango_DLL.dll')
# mazlib = ctypes.WinDLL('MwPCIeUi_x64.dll')


# --- Initialization

CreateLSID = mazlib.LSX_CreateLSID
# int LSX_CreateLSID(int *plLSID);
CreateLSID.argtypes = [ctypes.POINTER(ctypes.c_int)]

ConnectSimple = mazlib.LSX_ConnectSimple
# int LSX_ConnectSimple(int lLSID, int lAnInterfaceType, char *pcAComName, int lABaudRate, BOOL bAShowProt);
ConnectSimple.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char),
                          ctypes.c_int, ctypes.c_bool]

ClearPos = mazlib.LSX_ClearPos
# int LSX_ClearPos (int lLSID, int lFlags);
ClearPos.argtypes = [ctypes.c_int, ctypes.c_int]
ClearPos.__doc__ = """ Sets current position and internal position counter to 0. This function is needed for endless 
axes, as controller can only process =/-1,000 motor revolutions within its parameters. This instruction will be ignored 
for axes with encoders. lFlags argument determines axis, Bit 0=X, Bit 1=Y, Bit 2=Z, Bit 3=A
"""

# --- Status
GetError = mazlib.LSX_GetError
# int LSX_GetError (int lLSID, int *plErrorCode);
GetError.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
GetError.__doc__ = """Provides current error number. Example: GetError(1, &ErrorCode);"""

GetPos = mazlib.LSX_GetPos
# int LSX_GetPos (int lLSID, double *pdX, double *pdY, double *pdZ, double *pdA);
GetPos.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
GetPos.__doc__ = """Retrieves current position of all axes.
Parameters
X, Y, Z, A: Positions
Example
pTango->GetPos(1, &X, &Y, &Z, &A);"""

GetEncoder = mazlib.LSX_GetEncoder 
GetEncoder.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), 
                       ctypes.POINTER(ctypes.c_double), 
                       ctypes.POINTER(ctypes.c_double), 
                       ctypes.POINTER(ctypes.c_double)]
GetEncoder.__doc__ = """ Retrieves all encoder positions
Parameters 
XP, YP, ZP, AP: Counter values, 4x interpolated
Example 
pTango->GetEncoder(1, &XP, &YP, &ZP, &AP);"""

# --- Units
GetDimensions = mazlib.LSX_GetDimensions
# int LSX_GetDimensions (int lLSID, int *plXD, int *plYD, int *plZD, int *plAD);
GetDimensions.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
GetDimensions.__doc__ = """
Provides the applied measuring units of axes

Parameters XD, YD, ZD, AD: Dimension units
0  Microsteps
1  µm
2  mm (Pre-set)
3  Degree
4  Revolutions
5  cm
6  m
7  Inch
8  mil (1/1000 Inch)
Example pTango->GetDimensions(1, &XD, &YD,&ZD,&AD);
"""

SetDimensions = mazlib.LSX_SetDimensions
#int LSX_SetDimensions (int lLSID, int lXD, int lYD, int lZD, int lAD);
SetDimensions.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
SetDimensions.__doc__ = """Set measuring units of axes

Parameters XD, YD, ZD, AD: Dimension units
0  Microsteps
1  µm
2  mm (Pre-set)
3  Degree
4  Revolutions
5  cm
6  m
7  Inch
8  mil (1/1000 Inch)

Example pTango->SetDimensions(1, 3, 2, 2, 1);
// X-Axis in degree, Y- and Z-Axis in mm and A-Axis in µm
"""

# --- Movement
MoveRel = mazlib.LSX_MoveRel
# int LSX_MoveRel (int lLSID, double dX, double dY, double dZ, double dA, BOOL bWait);
MoveRel.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_bool]
MoveRel.__doc__ = """Move relative position.
Axes X, Y, Z and A are moved by the transmitted distances. All axes reach their destinations simultaneously.

Parameters
X, Y, Z, A: +/- Travel range, command depends on measuring unit (dimension)
Wait: TRUE = function waits until position is reached FALSE = function does not wait
Example
pTango->MoveRel(1, 10.0, 10.0, -10.0, 10.0, TRUE);"""

MoveRelSingleAxis = mazlib.LSX_MoveRelSingleAxis
MoveRelSingleAxis.argtypes = [ctypes.c_int, ctypes.c_int32, ctypes.c_double, ctypes.c_bool]
# int LSX_MoveRelSingleAxis (int lLSID, int lAxis, double dValue, BOOL bWait);
MoveRelSingleAxis.__doc__ = """Description Move single axis relative.
Parameters Axis: X, Y, Z and A numbered from 1 to 4
Value: Distance, command depends on set measuring unit
Example pTango->MoveRelSingleAxis(1, 3, 5,0);
// Z-Axis is moved by 5mm in positive direction"""

MoveRelShort = mazlib.LSX_MoveRelShort
# int LSX_MoveRelShort (int lLSID);
MoveRelShort.argtypes = [ctypes.c_int]
MoveRelShort.__doc__ = """Relative positioning (short command).
This command may be used to execute several fast equal distance relative moves.
Distances have to be pre-set once with LSX_SetDistance.
Parameters -
Example pTango->SetDistance(1, 1.0, 1.0, 0, 0);
for (i = 0; i < 10; i++) pTango->MoveRelShort(1); // position X- and Y-Axis 10 times relatively by 1mm"""

MoveAbs = mazlib.LSX_MoveAbs
# int LSX_MoveAbs (int lLSID, double dX, double dY, double dZ, double dA, BOOL bWait);
MoveAbs.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_bool]
MoveAbs.__doc__ = """
All axes are moved absolute positions.
Axes X, Y, Z and A are positioned at transferred position values.
Parameters
X, Y, Z, A: +/- Travel range, command depends on measuring unit
Wait: Determines, whether function shall return after reaching position (= TRUE) or directly after sending the command (= FALSE)
Example
pTango->MoveAbs(1, 10.0, 10.0, -10.0, 10.0, TRUE);"""

MoveAbsSingleAxis = mazlib.LSX_MoveAbsSingleAxis
# int LSX_MoveAbsSingleAxis (int lLSID, int lAxis, double dValue, BOOL bWait);
MoveAbsSingleAxis.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_bool]
MoveAbsSingleAxis.__doc__ = """
Positions a single axis at the transferred position.
Parameters
Axis: X, Y, Z and A, numbered from 1 to 4
Value: Position, command depends on measuring unit (dimension)
Example
pTango->MoveAbsSingleAxis(1, 2, 10.0);
// position Y-Axis absolutely at 10mm (dimension=2)"""

GetDistance = mazlib.LSX_GetDistance
# int LSX_GetDistance (int lLSID, double *pdX, double *pdY, double *pdZ, double *pdA);
GetDistance.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
GetDistance.__doc__ = """Retrieve distance values last used for LSX_MoveRelShort.
Parameters X, Y, Z, A: Current distances of all axes, depending on corresponding measuring unit.
Example pTango->GetDistance(1, &X, &Y, &Z, &A);"""

SetDistance = mazlib.LSX_SetDistance
# int LSX_SetDistance (int lLSID, double dX, double dY, double dZ, double dA);
SetDistance.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
SetDistance.__doc__ = """Sets distance parameters for command LSX_MoveRelShort. This enables very fast equal distance r
elative positioning without the need of communication overhead. 
Parameters
X, Y, Z, A: Min-/max- travel range, values depend on measuring unit.
Example
SetDistance(1, 1, 2, 0, 0); // sets distances for axes X to 1mm and Y to 2mm (if dimension=2), Z and A are not moved when calling function LSX_MoveRelShort"""


# --- velocity

GetVel = mazlib.LSX_GetVel
GetVel.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), 
                   ctypes.POINTER(ctypes.c_double), 
                   ctypes.POINTER(ctypes.c_double), 
                   ctypes.POINTER(ctypes.c_double)]
GetVel.__doc__ = """Retrieves velocity of all axes
Parameters 
pdX, pdY, pdZ, pdA: Velocity values [r/sec]
Example
GetVel(1, &X, &Y, &Z, &A);"""

SetVel = mazlib.LSX_SetVel
SetVel.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, 
                   ctypes.c_double, ctypes.c_double]
SetVel.__doc__ = """ Set velocity of all axes
Parameters
X, Y, Z, A: >0 – max. speed [r/sec]
Example 
SetVel(1, 20.0, 15.0, 0.5, 10)"""

IsVel = mazlib.LSX_IsVel
IsVel.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), 
                   ctypes.POINTER(ctypes.c_double), 
                   ctypes.POINTER(ctypes.c_double), 
                   ctypes.POINTER(ctypes.c_double)]
IsVel.__doc__ = """ Read the actual velocities at which the axes are currently 
travelling. Unlike '?vel' or '?speed' this instruction returns the currently 
travelled (true) speed of the axes, even when controlled by a HDI device.
Parameters pdX, pdY, pd Z, pdA: actual axes velocities in [mm/s]
Example pTango->IsVel(1, &vx, &vy, &vz, &va);"""

GetPitch = mazlib.LSX_GetPitch
GetPitch.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), 
                   ctypes.POINTER(ctypes.c_double), 
                   ctypes.POINTER(ctypes.c_double), 
                   ctypes.POINTER(ctypes.c_double)]
GetPitch.__doc__ = """ Provides spindle pitch.
Parameters X, Y, Z, A: Spindle pitch [mm]
Example pTango->GetPitch(1, &X, &Y, &Z, &A);
NOTE - it looks like this is actually [mm/r]"""


# --- Joystick
SetJoystickOn = mazlib.LSX_SetJoystickOn
SetJoystickOff = mazlib.LSX_SetJoystickOff


# --- Shutting down
Disconnect = mazlib.LSX_Disconnect
# int LSX_Disconnect(int lLSID);
Disconnect.argtypes = [ctypes.c_int]

FreeLSID = mazlib.LSX_FreeLSID
# int LSX_FreeLSID(int lLSID);
FreeLSID.argtypes = [ctypes.c_int]

StopAxes = mazlib.LSX_StopAxes
# int LSX_StopAxes (int lLSID);
StopAxes.argtypes = [ctypes.c_int]

# --- limits

GetLimit = mazlib.LSX_GetLimit
GetLimit.argtypes = [ctypes.c_int, ctypes.c_int, 
                     ctypes.POINTER(ctypes.c_double), 
                     ctypes.POINTER(ctypes.c_double)]
GetLimit.__doc__ = """Provides soft travel range limits
Parameters 
    Axis: Axis from which travel range limits are to be retrieved
    (X, Y, Z, A numbered from 1=X to 4=A)
    MinRange: lower travel range limit, unit depends on dimension
    MaxRange: upper travel range limit, unit depends on dimension
Example
GetLimit(1, &MinRange, &MaxRange);"""

SetLimit = mazlib.LSX_SetLimit
SetLimit.argtypes = [ctypes.c_int, ctypes.c_int, 
                     ctypes.c_double, ctypes.c_double]
SetLimit.__doc__ = """ Set soft travel range limits
Parameters
Axis: Axis from which travel range limits are to be retrieved
(X, Y, Z, A numbered from 1=X to 4=A)
MinRange: lower travel range limit, unit depends on dimension
MaxRange: upper travel range limit, unit depends on dimension
Example 
SetLimit(1, 1, -10.0, 20.0);
// assign X-Axis –10 as lower and 20 as upper travel range limits"""

GetLimitControl = mazlib.LSX_GetLimitControl
GetLimitControl.argtypes = [ctypes.c_int, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_bool)]
GetLimitControl.__doc__ = """ Retrieves, whether area control (limits) is 
switched on or off.
Parameters
    Axis: X, Y, Z and A, numbered from 1=X to 4=A
    Active: TRUE = area control of corresponding axis is active
            FALSE = area control of corresponding axis is deactivated
Example 
GetLimitControl(1, 2, &Active);
"""

SetLimitControl = mazlib.LSX_GetLimitControl
SetLimitControl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
SetLimitControl.__doc__ = """ Retrieves, whether area control (limits) is 
switched on or off.
Parameters
    Axis: X, Y, Z and A, numbered from 1=X to 4=A
    Active: TRUE = activate area control of corresponding axis
            FALSE = disable area control of corresponding axis
Example 
SetLimitControl(1, 2, TRUE); // Area control of Y-Axis is active
"""


class MarzHauserJoystick(object):
    """Shim class to be used for scope.joystick"""
    def __init__(self, stepper):
        self.stepper = stepper

    def Enable(self, enabled=True):
        if not self.IsEnabled() == enabled:
            self.stepper.SetJoystick(enabled)

    def IsEnabled(self):
        return self.stepper._joystick_enabled


class MarzhauserTango(PiezoBase):
    """
    Marzhauser Tango stage. The dll API supports 4 axes (x, y, z, a), all of
    which are exposed/supported by this class. For stages only supporting (x, y)
    or (x, y, z) axes, you can directly use this class and only register the
    axes you have wth PYMEAcquire, however it is recommended to subclass and
    expose only the axes present
    """
    units_um=1  # units of the stage default to mm, but we set to um in init

    def __init__(self):
        self.lock = threading.Lock()

        # open interface
        self.lsid = ctypes.c_int(0)
        CreateLSID(ctypes.byref(self.lsid))
        # ConnectSimple(1, -1, NULL, 57600, TRUE); // Autoconnect with the first found USB or PCI TANGO in the system
        ConnectSimple(self.lsid, ctypes.c_int(-1), None, ctypes.c_int(57600), ctypes.c_bool(False))  # baud doesn't matter for dll usage

        SetDimensions(self.lsid, ctypes.c_int(1), ctypes.c_int(1), 
                      ctypes.c_int(1), ctypes.c_int(1))  # put all axes in [um]

        self._allocate_memory()
        
        self._joystick_enabled = False
        self.SetJoystick(True)
    
    def _allocate_memory(self):
        self._c_position = (ctypes.c_double(0),  # X, [um]
                                ctypes.c_double(0),  # Y, [um]
                                ctypes.c_double(0),  # Z, [um]
                                ctypes.c_double(0))  # A, [um]
        self._c_position_ref = tuple(ctypes.byref(p) for p in self._c_position)

        self._c_target_rps = (ctypes.c_double(0),  # X, [r/sec]
                                        ctypes.c_double(0),  # Y, [r/sec]
                                        ctypes.c_double(0),  # Z, [r/sec]
                                        ctypes.c_double(0))  # A, [r/sec]
        self._c_target_rps_ref = tuple(ctypes.byref(p) for p in self._c_target_rps)
        
        self._c_encoder_positions = (ctypes.c_double(0),  # X, [um]
                                ctypes.c_double(0),  # Y, [um]
                                ctypes.c_double(0),  # Z, [um]
                                ctypes.c_double(0))  # A, [um]
        self._c_encoder_positions_ref = tuple(ctypes.byref(p) for p in self._c_encoder_positions)

        # note pitch is always measured in mm, doesn't change with SetDimensions
        self._c_pitch = (ctypes.c_double(0),  # X, [mm / r]
                                ctypes.c_double(0),  # Y, [mm / r]
                                ctypes.c_double(0),  # Z, [mm / r]
                                ctypes.c_double(0))  # A, [mm / r]
        self._c_pitch_ref = tuple(ctypes.byref(p) for p in self._c_pitch)
        
        self.c_limits_active = ctypes.c_bool(False)

        # use dict-lookup for limits as PYME axes are 0-indexed and tango's are
        # 1-indexed
        self._c_limits = {
            'axis': {
                0: ctypes.c_int(1),  # X
                1: ctypes.c_int(2),  # Y
                2: ctypes.c_int(3),  # Z
                3: ctypes.c_int(4),  # A
            },
            'active': {
                0: ctypes.c_bool(0),  # X
                1: ctypes.c_bool(0),  # Y
                2: ctypes.c_bool(0),  # Z
                3: ctypes.c_bool(0)  # A
            },
            'active_ref': {
                0: '',
                1: '',
                2: '',
                3: ''
            },
            0: (ctypes.c_double(0),  ctypes.c_double(0)),  # X min-max [um]
            1: (ctypes.c_double(0),  ctypes.c_double(0)),  # Y min-max [um]
            2: (ctypes.c_double(0),  ctypes.c_double(0)),  # Z min-max [um]
            3: (ctypes.c_double(0),  ctypes.c_double(0)),  # A min-max [um]
        }
        for ind in range(4):
            self._c_limits['ref'] = {
                ind: (ctypes.byref(self._c_limits[ind][0]),
                      ctypes.byref(self._c_limits[ind][1]))
            }
            self._c_limits['active_ref'][ind] = ctypes.byref(self._c_limits['active'][0])

    def __del__(self):
        self.close()

    def close(self):
        # stop everything
        StopAxes(self.lsid)
        # close interface
        Disconnect(self.lsid)
        FreeLSID(self.lsid)

    def ReInit(self):
        raise NotImplementedError

    def MoveTo(self, iChannel, fPos, bTimeOut=True, wait=True):
        with self.lock:
            pos = [ctypes.c_double() for i in range(4)]
            GetPos(self.lsid, ctypes.byref(pos[0]), ctypes.byref(pos[1]), ctypes.byref(pos[2]), ctypes.byref(pos[3]))

            pos[iChannel] = ctypes.c_double(fPos)
            MoveAbs(self.lsid, pos[0], pos[1], pos[2], pos[3], wait)
            # MoveAbs (int lLSID, double dX, double dY, double dZ, double dA, BOOL bWait);

    def MoveRel(self, iChannel, incr, bTimeOut=True, wait_to_finish=True):
        pos = 4 * [ctypes.c_double()]
        pos[iChannel] = incr
        with self.lock:
            MoveRel(self.lsid, pos[0], pos[1], pos[2], pos[3], wait_to_finish)

    def MoveRelOneAxis(self, increment):

        raise NotImplementedError

    def GetPos(self, iChannel=0):
        with self.lock:
            GetPos(self.lsid, *self._c_position_ref)
        return self._c_position[iChannel].value

    def _get_dimensions(self):
        """ Returns measurement unit specified for each axis:
        0  Microsteps
        1  µm
        2  mm (Pre-set)
        3  Degree
        4  Revolutions
        5  cm
        6  m
        7  Inch
        8  mil (1/1000 Inch)
        """
        dim = [ctypes.c_int() for i in range(4)]
        with self.lock:
            GetDimensions(self.lsid, ctypes.byref(dim[0]), ctypes.byref(dim[1]), ctypes.byref(dim[2]), ctypes.byref(dim[3]))
        return dim

    def SetJoystick(self, enabled=True):
        if enabled:
            SetJoystickOn(self.lsid, ctypes.c_bool(True), ctypes.c_bool(True))
            self._joystick_enabled = True
        else:
            SetJoystickOff(self.lsid)
            self._joystick_enabled = False

    def GetMin(self,iChan=0):
        return -25
    def GetMax(self, iChan=0):
        return 25

    def clear_position(self):
        ClearPos(self.lsid, 7)
    
    @property
    def _target_rps(self):
        """
        Returns
        -------
        target_rps : list
            (x, y, z, a) velocities in [revolutions / s]
        """
        GetVel(self.lsid, *self._c_target_rps_ref)
        return [v.value for v in self._c_target_rps]
    
    @_target_rps.setter
    def _target_rps(self, rps):
        try:
            for ind, value in enumerate(rps):
                self._c_target_rps[ind].value = value
            SetVel(self.lsid, *self._c_target_rps)
        except:
            GetVel(self.lsid, *self._c_target_rps_ref)  # try and fix state
            raise
    
    @property
    def _pitch(self):
        "thread pitch along each axis, in [mm] (equivalently, mm / revolution)"
        GetPitch(self.lsid, *self._c_pitch_ref)
        return [v.value for v in self._c_pitch]
    
    @property
    def _velocity(self):
        vel_mm = np.asarray(self._pitch) * np.asarray(self._target_rps)
        return vel_mm * 1e3
    
    @_velocity.setter
    def _velocity(self, velocities):
        """
        Changing pitch seems a little weird. Might be possible via gear changes,
        but for now we let the board/stage handle that on its own, so to change
        the velocity we'll just change the target revolutions per second
        """
        self._target_rps = (np.asarray(velocities) / 1e3) / self._pitch
    
    def get_encoder_positions(self):
        GetEncoder(self.lsid, *self._c_encoder_positions_ref)
        return [p.value for p in self._c_encoder_positions]
    
    def get_axis_limits(self, axis):
        """get min/max position limits for a given axes (software limits)

        Parameters
        ----------
        axis : int
            0 - x, 1 - y, 2 - z, 3 - a

        Returns
        -------
        range : list
            min [0] and max [1] limits for `axis`
        """
        GetLimit(self.lsid, self._c_limits['axis'][axis],
                 *self._c_limits['ref'][axis])
        return [p.value for p in self._c_limits['ref'][axis]]
    
    def set_axis_limits(self, axis, min_value, max_value):
        """set min/max position limits for a given axis (software limits)

        Parameters
        ----------
        axis : int
            0 - x, 1 - y, 2 - z, 3 - a
        min_value : float
            lower limit for `axis`
        max_value : float
            upper limit for `axis`
        """
        self._c_limits['ref'][axis][0].value = min_value
        self._c_limits['ref'][axis][1].value = max_value
        SetLimit(self.lsid, self._c_limits['axis'][axis],
                 *self._c_limits['ref'][axis])
    
    def get_software_limit_state(self, axis):
        """check if software limits are enabled or disabled for a given axis

        Parameters
        ----------
        axis : int
            0 - x, 1 - y, 2 - z, 3 - a
        
        Returns
        -------
        bool : enabled (True), or disabled (False)
        """
        GetLimitControl(self.lsid, self._c_limits['axis'][axis],
                        self._c_limits['active_ref'][axis])
        return self._c_limits['active_ref'][axis].value
    
    def set_software_limit_state(self, axis, state):
        """activate/inactivate software limits for a given axis

        Parameters
        ----------
        axis : int
            0 - x, 1 - y, 2 - z, 3 - a
        state : bool
            enable (True), disable (False)
        """
        self._c_limits['active_ref'][axis].value = state
        SetLimitControl(self.lsid, self._c_limits['axis'][axis],
                        self._c_limits['active'][axis])


class MarzhauserTangoXY(MarzhauserTango):
    """
    Marzhauser tango stage set up only for lateral (x, y) positioning
    """
    pass


if __name__=='__main__':
    stage = MarzhauserTangoXY()
    stage.close()