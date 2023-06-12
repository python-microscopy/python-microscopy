
# Defines a GCSPiezo class which uses the pipython package distributed 
# by Physik Instrumente to initialize and run their stages.
# Tested with a E-727 controller, on pipython 2.9.0.4 (pypi)
# see https://github.com/PI-PhysikInstrumente/PIPython


from PYME.Acquire.Hardware.Piezos.base_piezo import PiezoBase
from pipython import GCSDevice
from pipython import PILogger, WARNING
import numpy as np
import logging
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
            for a 3-axis stage. Note some PI firmwares assume ['a', 'b', 'c']

        """
        PiezoBase.__init__(self)
        self.pi = GCSDevice()
        self.pi.ConnectUSB(description)
        assert self.pi.IsConnected()

        if axes is None:
            self.axes = [1]  # default to the first axis
        else:
            self.axes = axes
        
        units = self.pi.qPUN(self.axes)
        logger.debug('stage units: %s' % [units[axis] for axis in self.axes])
        # PI appears to use unicoded um to set units to um, not funcitonal at the moment, but
        # ideally we would remove the unit check/log above and just force to um here.
        # self.pi.PUN(axes, values)

    def SetServo(self, val=1):
        # due to form of SetServo in the baseclass, just set all axes for now
        self.pi.SVO(self.axes, [val for axis in self.axes])
    
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        self.pi.MOV(iChannel, fPos)
    
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        """ relative to current target position
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        self.pi.MVR(iChannel, incr)
    
    def GetPos(self, iChannel=1):
        try:
            assert np.isscalar(iChannel)
        except:
            raise AssertionError('GetPos only supports single-axis query')
        return self.pi.qPOS([iChannel])[iChannel]
    
    def GetTargetPos(self, iChannel=1):
        # assume that target pos = current pos. Over-ride in derived class if possible
        return self.GetPos(iChannel)
    
    def GetMin(self, iChan=1):
        """
        Get lower limits ("soft limits")
        """
        try:
            assert np.isscalar(iChan)
        except:
            raise AssertionError('GetMin only supports single-axis query')
        return self.pi.qTMN(iChan)[iChan]
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
        return self.pi.qTMX(iChan)[iChan]
        # return self.pi.qPLM([iChan])[iChan]
        # qCMX max commandable closed-loop target
    
    def GetFirmwareVersion(self):
        raise NotImplementedError
    
    def OnTarget(self, axes=None):
        """
        For multiaxis stages, query with None reports for all of them
        """
        return all(self.pi.qONT(axes))
    
    def close(self):
        self.pi.CloseConnection()
