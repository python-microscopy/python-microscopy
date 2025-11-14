

# Windows only Thorlabs Elliptec stage control via Thorlabs provided dll
# requires https://pythonnet.github.io/


dll = r"C:\Program Files\Thorlabs\Elliptec\Thorlabs.Elliptec.ELLO_DLL.dll"
import clr
clr.AddReference(dll)
from Thorlabs.Elliptec.ELLO_DLL import ELLDevices, ELLDevicePort
from System import Decimal
import logging
from PYME.Acquire.Hardware.Piezos.base_piezo import PiezoBase
from PYME.Acquire.Hardware.FilterWheel import FilterWheelBase


logger = logging.getLogger(__name__)

class ElliptecBase(object):
    """
    base class for Thorlabs elliptec devices. 
    See below for stage vs multiposition slider classes
    """
    def __init__(self, com_port='COM4', device_number=0, home_on_init=False):
        """
        Parameters
        ----------
        com_port : str
            The COM port to which the Elliptec controller is connected
        device_number : int
            index of the device to control (if daisy chaining)
        home_on_init : bool
            whether to home the device during initialization. Some devices
            require homing before use, while others may not have home functionality.
        """
        ELLDevicePort.Connect(com_port)
        elldevices = ELLDevices()
        logger.info("Scanning for devices...")
        devices = elldevices.ScanAddresses('0', 'F')  # returns list of strings
        device = devices[device_number]  # something like 0IN0E1140132620251801016800023000
        logger.info(f"configuring: {device}")
        elldevices.Configure(device)
        self.device = elldevices.AddressedDevice(device[0])  # index [0], only wants channel as char
        if home_on_init:
            logger.info("Homing...")
            self.device.Home()
        
        self.dinfo = self.device.DeviceInfo.Description()
        for line in self.dinfo:
            logger.info(line)


class ElliptecStage(ElliptecBase, PiezoBase):
    """
    Controls Thorlabs Elliptec single axes stages, or rotation stages.
    Tested on ELL14 rotation stage.
    """
    def __init__(self, com_port='COM4', device_number=0, home_on_init=False):
        """
        Parameters
        ----------
        com_port : str
            The COM port to which the Elliptec controller is connected
        device_number : int
            index of the device to control (if daisy chaining)
        home_on_init : bool
            whether to home the device during initialization. Some devices
            require homing before use, while others may not have home functionality.
        """
        ElliptecBase.__init__(self, com_port, device_number, home_on_init)

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        self.device.MoveAbsolute(Decimal(fPos))
    
    def MoveRel(self, iChannel, incr, bTimeOut=True):
        self.device.MoveRelative(Decimal(incr))
    
    def GetPos(self, iChannel=0):
        self.device.GetPosition()
        pos = Decimal.ToDouble(self.device.Position)
        return pos
    
    def GetMin(self, iChan=1):
        return 0.0
    
    def GetMax(self, iChan=1):
        return Decimal.ToDouble(self.device.DeviceInfo.Travel)
    
    def GetFirmwareVersion(self):
        for line in self.dinfo:
            if 'firmware' in line.lower():
                return line.split(': ')[-1]
    
    @property
    def units_um(self):
        if self.device.DeviceInfo.Units == 'deg':
            # rotation stage. Pretend degrees are microns
            return 1
        elif self.device.DeviceInfo.Units == 'mm':
            return 1000
        elif self.device.DeviceInfo.Units == 'inches':
            return 25400


class ElliptecMultiPositionSlider(ElliptecBase, FilterWheelBase):
    def __init__(self, com_port='COM4', device_number=0):
        super().__init__(com_port, device_number, home_on_init=False)
        raise NotImplementedError("ElliptecMultiPositionSlider not yet implemented")

