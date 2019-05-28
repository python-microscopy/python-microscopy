

from PYME.Acquire.Hardware.Piezos.base_piezo import PiezoBase
import threading
import ctypes

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



class MarzHauserJoystick(object):
    """Shim class to be used for scope.joystick"""
    def __init__(self, stepper):
        self.stepper = stepper

    def Enable(self, enabled=True):
        if not self.IsEnabled() == enabled:
            self.stepper.SetJoystick(enabled)

    def IsEnabled(self):
        return self.stepper._joystick_enabled


class MarzhauserTangoXY(PiezoBase):
    """
    Marzhauser stage set-up only for lateral positioning
    """
    units_um=1000  # units of the stage default to mm, so 1000 um per stage unit

    def __init__(self):
        self.lock = threading.Lock()

        # open interface
        self.lsid = ctypes.c_int(0)
        CreateLSID(ctypes.byref(self.lsid))
        # ConnectSimple(1, -1, NULL, 57600, TRUE); // Autoconnect with the first found USB or PCI TANGO in the system
        ConnectSimple(self.lsid, ctypes.c_int(-1), None, ctypes.c_int(57600), ctypes.c_bool(False))  # baud doesn't matter for dll usage

        # clear the current position
        ClearPos(self.lsid, 7)

        self.last_x, self.last_y, self.last_z = ctypes.c_double, ctypes.c_double, ctypes.c_double
        self.last_a = ctypes.c_double

        self._move_short_increment = dict(x=ctypes.c_double(0), y=ctypes.c_double(0))
        z, a = ctypes.c_double(0), ctypes.c_double(0)
        GetDistance(self.lsid, self._move_short_increment['x'], self._move_short_increment['y'], z, a)

        self._joystick_enabled = False
        self.SetJoystick(True)

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
        pos = (4 * ctypes.c_double)()
        pos[iChannel] = incr
        with self.lock:
            MoveRel(self.lsid, pos[0], pos[1], pos[2], pos[3], wait_to_finish)

    def MoveRelOneAxis(self, increment):

        raise NotImplementedError

    def GetPos(self, iChannel=0):
        pos = [ctypes.c_double() for i in range(4)]
        with self.lock:
            GetPos(self.lsid, ctypes.byref(pos[0]), ctypes.byref(pos[1]), ctypes.byref(pos[2]),ctypes.byref(pos[3]))
            self.last_x, self.last_y, self.last_z, self.last_a = [p.value for p in pos]

        return pos[iChannel].value

    def _get_dimensions(self):
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


import numpy as np
class MarzhauserTangoXYThreaded(PiezoBase):
    units_um = 1000
    gui_description = 'Stage %s'

    def __init__(self, portname='', maxtravel=50.00, hasTrigger=False, reference=False, maxvelocity=200.,
                 validRegion=[[-50, 50], [-50, 50]]):
        raise NotImplementedError
        self.max_travel = maxtravel
        self.maxvelocity = maxvelocity

        self.units = 'mm'

        self.validRegion = validRegion
        self.onTarget = False
        self.ptol = 5e-4

        self.onTarget = False
        self.onTargetLast = False

        self.errCode = 0

        # open interface
        self.lsid = ctypes.c_int(0)
        CreateLSID(ctypes.byref(self.lsid))
        # ConnectSimple(1, -1, NULL, 57600, TRUE); // Autoconnect with the first found USB or PCI TANGO in the system
        ConnectSimple(self.lsid, ctypes.c_int(-1), None, ctypes.c_int(57600),
                      ctypes.c_bool(False))  # baud doesn't matter for dll usage

        # clear the current position
        ClearPos(self.lsid, 7)

        if reference:
            # find reference switch (should be in centre of range)
            self.ser_port.write('FRF\n')

            time.sleep(.5)
            # self.lastPos = self.GetPos()
        # self.lastPos = [self.GetPos(1), self.GetPos(2)]

        # self.driftCompensation = False
        self.hasTrigger = hasTrigger
        self.loopActive = True
        self.stopMove = False
        self.position = np.array([12.5, 12.5])
        self.velocity = np.array([self.maxvelocity, self.maxvelocity])

        self.targetPosition = np.array([12.5, 12.5])
        self.targetVelocity = self.velocity.copy()

        self.lastTargetPosition = self.position.copy()

        self.lock = threading.Lock()
        self.tloop = threading.Thread(target=self._Loop)
        self.tloop.start()

    def _Loop(self):
        while self.loopActive:
            self.lock.acquire()
            try:
                self.ser_port.flushInput()
                self.ser_port.flushOutput()

                # check position
                self.ser_port.write('POS? 1 2\n')
                self.ser_port.flushOutput()
                # time.sleep(0.005)
                res1 = self.ser_port.readline()
                res2 = self.ser_port.readline()
                # print res1, res2
                self.position[0] = float(res1.split('=')[1])
                self.position[1] = float(res2.split('=')[1])

                self.ser_port.write('ERR?\n')
                self.ser_port.flushOutput()
                self.errCode = int(self.ser_port.readline())

                if not self.errCode == 0:
                    # print(('Stage Error: %d' %self.errCode))
                    logger.error('Stage Error: %d' % self.errCode)

                # print self.targetPosition, self.stopMove

                if self.stopMove:
                    self.ser_port.write('HLT\n')
                    time.sleep(.1)
                    self.ser_port.write('POS? 1 2\n')
                    self.ser_port.flushOutput()
                    # time.sleep(0.005)
                    res1 = self.ser_port.readline()
                    res2 = self.ser_port.readline()
                    # print res1, res2
                    self.position[0] = float(res1.split('=')[1])
                    self.position[1] = float(res2.split('=')[1])
                    self.targetPosition[:] = self.position[:]
                    self.stopMove = False

                if self.servo:
                    if not np.all(self.velocity == self.targetVelocity):
                        for i, vel in enumerate(self.targetVelocity):
                            self.ser_port.write('VEL %d %3.9f\n' % (i + 1, vel))
                        self.velocity = self.targetVelocity.copy()
                        # print('v')
                        logger.debug('Setting stage target vel: %s' % self.targetVelocity)

                    # if not np.all(self.targetPosition == self.lastTargetPosition):
                    if not np.allclose(self.position, self.targetPosition, atol=self.ptol):
                        # update our target position
                        pos = np.clip(self.targetPosition, 0, self.max_travel)

                        self.ser_port.write('MOV 1 %3.9f 2 %3.9f\n' % (pos[0], pos[1]))
                        self.lastTargetPosition = pos.copy()
                        # print('p')
                        logger.debug('Setting stage target pos: %s' % pos)
                        time.sleep(.01)

                # check to see if we're on target
                self.ser_port.write('ONT?\n')
                self.ser_port.flushOutput()
                time.sleep(0.005)
                res1 = self.ser_port.readline()
                ont1 = int(res1.split('=')[1]) == 1
                res1 = self.ser_port.readline()
                ont2 = int(res1.split('=')[1]) == 1

                onT = (ont1 and ont2) or (self.servo == False)
                self.onTarget = onT and self.onTargetLast
                self.onTargetLast = onT
                self.onTarget = np.allclose(self.position, self.targetPosition, atol=self.ptol)

                # time.sleep(.1)

            except serial.SerialTimeoutException:
                # print('Serial Timeout')
                logger.debug('Serial Timeout')
                pass
            finally:
                self.stopMove = False
                self.lock.release()

        # close port on loop exit
        self.ser_port.close()
        logger.info("Stage serial port closed")

    def close(self):
        logger.info("Shutting down XY Stage")
        with self.lock:
            self.loopActive = False
            # time.sleep(.01)
            # self.ser_port.close()

    def SetServo(self, state=1):
        self.lock.acquire()
        try:
            self.ser_port.write('SVO 1 %d\n' % state)
            self.ser_port.write('SVO 2 %d\n' % state)
            self.servo = state == 1
        finally:
            self.lock.release()

    #    def SetParameter(self, paramID, state):
    #        self.lock.acquire()
    #        try:
    #            self.ser_port.write('SVO 1 %d\n' % state)
    #            self.ser_port.write('SVO 2 %d\n' % state)
    #            self.servo = state == 1
    #        finally:
    #            self.lock.release()

    def ReInit(self, reference=True):
        # self.ser_port.write('WTO A0\n')
        self.lock.acquire()
        try:
            self.ser_port.write('RBT\n')
            time.sleep(1)
            self.ser_port.write('SVO 1 1\n')
            self.ser_port.write('SVO 2 1\n')
            self.servo = True

            if reference:
                # find reference switch (should be in centre of range)
                self.ser_port.write('FRF\n')

            time.sleep(1)
            self.stopMove = True
        finally:
            self.lock.release()

        # self.lastPos = [self.GetPos(1), self.GetPos(2)]

    def SetVelocity(self, chan, vel):
        # self.ser_port.write('VEL %d %3.4f\n' % (chan, vel))
        # self.ser_port.write('VEL 2 %3.4f\n' % vel)
        self.targetVelocity[chan - 1] = vel

    def GetVelocity(self, chan):
        return self.velocity[chan - 1]

    def MoveTo(self, iChannel, fPos, bTimeOut=True, vel=None):
        chan = iChannel - 1
        if vel is None:
            vel = self.maxvelocity
        self.targetVelocity[chan] = vel
        self.targetPosition[chan] = min(max(fPos, self.validRegion[chan][0]), self.validRegion[chan][1])
        self.onTarget = False

    # def MoveRel(self, iChannel, incr, bTimeOut=True):
    #        self.ser_port.write('MVR %d %3.6f\n' % (iChannel, incr))

    def MoveToXY(self, xPos, yPos, bTimeOut=True, vel=None):
        if vel is None:
            vel = self.maxvelocity
        self.targetPosition[0] = min(max(xPos, self.validRegion[0][0]), self.validRegion[0][1])
        self.targetPosition[1] = min(max(yPos, self.validRegion[1][0]), self.validRegion[1][1])
        self.targetVelocity[:] = vel
        self.onTarget = False

    def GetPos(self, iChannel=0):
        # self.ser_port.flush()
        # time.sleep(0.005)
        # self.ser_port.write('POS? %d\n' % iChannel)
        # self.ser_port.flushOutput()
        # time.sleep(0.005)
        # res = self.ser_port.readline()
        pos = self.GetPosXY()

        return pos[iChannel - 1]

    def GetPosXY(self):
        return self.position

    def MoveInDir(self, dx, dy, th=.0000):
        self.targetVelocity[0] = abs(dx) * self.maxvelocity
        self.targetVelocity[1] = abs(dy) * self.maxvelocity

        logger.debug('md %f,%f' % (dx, dy))

        if dx > th:
            self.targetPosition[0] = min(max(np.round(self.position[0] + 1), self.validRegion[0][0]),
                                         self.validRegion[0][1])
        elif dx < -th:
            self.targetPosition[0] = min(max(np.round(self.position[0] - 1), self.validRegion[0][0]),
                                         self.validRegion[0][1])
        else:
            self.targetPosition[0] = min(max(self.position[0], self.validRegion[0][0]), self.validRegion[0][1])

        if dy > th:
            self.targetPosition[1] = min(max(np.round(self.position[1] + 1), self.validRegion[1][0]),
                                         self.validRegion[1][1])
        elif dy < -th:
            self.targetPosition[1] = min(max(np.round(self.position[1] - 1), self.validRegion[1][0]),
                                         self.validRegion[1][1])
        else:
            self.targetPosition[1] = min(max(self.position[1], self.validRegion[1][0]), self.validRegion[1][1])

        self.onTarget = False

    def StopMove(self):
        self.stopMove = True

    def GetControlReady(self):
        return True

    def GetChannelObject(self):
        return 1

    def GetChannelPhase(self):
        return 2

    def GetMin(self, iChan=1):
        return 0

    def GetMax(self, iChan=1):
        return self.max_travel

    def OnTarget(self):
        return self.onTarget


if __name__=='__main__':
    stage = MarzhauserTangoXY()
    stage.close()