# thorlabs_elliptec_serial.py
# Cross-platform Thorlabs Elliptec stage control via serial communication.
# Requires the elliptec package: https://github.com/roesel/elliptec
# To instead use the Thorlabs-provided DLL, see thorlabs_elliptec_dll.py

import logging
from elliptec import Controller, Rotator, Linear, Slider
from elliptec.cmd import get_
from PYME.Acquire.Hardware.Piezos.base_piezo import PiezoBase
from PYME.Acquire.Hardware.FilterWheel import FilterWheelBase

logger = logging.getLogger(__name__)

ROTATION_TYPES = frozenset({14, 18})  # ELL14, ELL18
LINEAR_TYPES = frozenset({20})        # ELL20
SLIDER_TYPES = frozenset({6, 9})      # ELL6, ELL9


def _probe_motor_type(controller, address):
    """Query device info and return the integer motor type (e.g. 14 for ELL14)."""
    status = controller.send_instruction(get_['info'], address=address)
    if isinstance(status, dict):
        return int(status['Motor Type'])
    raise RuntimeError(f"Could not get Elliptec device info at address {address!r}")


def _create_motor(controller, address):
    """Create and return the appropriate Motor subclass for the device at address."""
    motor_type = _probe_motor_type(controller, address)
    if motor_type in ROTATION_TYPES:
        return Rotator(controller, address=address)
    elif motor_type in LINEAR_TYPES:
        return Linear(controller, address=address)
    elif motor_type in SLIDER_TYPES:
        return Slider(controller, address=address)
    else:
        raise ValueError(f"Unsupported Elliptec device type: ELL{motor_type}")


class ElliptecBase(object):
    """Base class for Thorlabs Elliptec devices via serial."""

    def __init__(self, com_port, address='0', home_on_init=False):
        """
        Parameters
        ----------
        com_port : str
            Serial port.
        address : str
            Hex address of the device (0-F). When daisy-chaining use '0', '1', etc.
        home_on_init : bool
            Whether to home the device during initialization.
        """
        self.controller = Controller(port=com_port)
        self.motor = _create_motor(self.controller, address)
        logger.info("Elliptec device connected: ELL%s (S/N: %s)",
                    self.motor.motor_type, self.motor.serial_no)
        logger.info(str(self.motor))
        if home_on_init:
            logger.info("Homing...")
            self.motor.home()

    def close(self):
        self.controller.close_connection()


class ElliptecStage(ElliptecBase, PiezoBase):
    """Controls Thorlabs Elliptec linear or rotation stages via serial.
    Units follow device type: degrees for rotation stages (ELL14/ELL18),
    mm for linear stages (ELL20).
    """

    def __init__(self, com_port, address='0', home_on_init=False, soft_limits=None):
        """
        Parameters
        ----------
        soft_limits : tuple or None
            Optional (min, max) movement limits in device units (degrees or mm).
            Overrides hardware travel limits. E.g. (15.0, 45.0) for a linear stage.
        """
        ElliptecBase.__init__(self, com_port, address, home_on_init)
        if not isinstance(self.motor, (Rotator, Linear)):
            raise ValueError(
                f"ElliptecStage requires a rotation or linear device; "
                f"got ELL{self.motor.motor_type}"
            )
        self._soft_min = None
        self._soft_max = None
        if soft_limits is not None:
            self.SetSoftLimits(0, soft_limits)

    def SetSoftLimits(self, iChannel, lims):
        """Set software movement limits.

        Parameters
        ----------
        iChannel : int
            Channel index (unused, single-axis stage).
        lims : tuple
            (min, max) in device units (degrees for rotation, mm for linear).
        """
        hw_min = 0.0
        hw_max = float(self.motor.range)
        self._soft_min = max(float(lims[0]), hw_min)
        self._soft_max = min(float(lims[1]), hw_max)
        logger.info("Soft limits set to [%g, %g]", self._soft_min, self._soft_max)

    def _clamp(self, pos):
        lo = self._soft_min if self._soft_min is not None else 0.0
        hi = self._soft_max if self._soft_max is not None else float(self.motor.range)
        return max(lo, min(hi, pos))

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        self.motor._set_unit(self._clamp(fPos))

    def MoveRel(self, iChannel, incr, bTimeOut=True):
        current = self.motor._get_unit() or 0.0
        self.motor._set_unit(self._clamp(current + incr))

    def GetPos(self, iChannel=0):
        return self.motor._get_unit()

    def GetMin(self, iChan=1):
        return self._soft_min if self._soft_min is not None else 0.0

    def GetMax(self, iChan=1):
        return self._soft_max if self._soft_max is not None else float(self.motor.range)

    def GetFirmwareVersion(self):
        return self.motor.info.get('Firmware', None)

    @property
    def units_um(self):
        if self.motor.motor_type in ROTATION_TYPES:
            # Rotation stage: treat degrees as microns (matches DLL convention)
            return 1
        elif self.motor.motor_type in LINEAR_TYPES:
            # Linear stage: mm -> um
            return 1000
        return 1


class ElliptecMultiPositionSlider(ElliptecBase, FilterWheelBase):
    """Controls Thorlabs Elliptec multi-position sliders (ELL6, ELL9) via serial.

    installedFilters should be a list of WFilter objects whose .pos values are
    1-indexed slot numbers matching the physical slider positions.
    """

    def __init__(self, com_port, address='0', installedFilters=None):
        ElliptecBase.__init__(self, com_port, address, home_on_init=False)
        if not isinstance(self.motor, Slider):
            raise ValueError(
                f"ElliptecMultiPositionSlider requires a slider device (ELL6/ELL9); "
                f"got ELL{self.motor.motor_type}"
            )
        FilterWheelBase.__init__(self, installedFilters or [])

    def _set_physical_position(self, pos):
        """Move to slot. pos is a 1-indexed slot number matching WFilter.pos values."""
        self.motor.set_slot(pos)

    def _get_physical_position(self):
        """Returns the current 1-indexed slot number."""
        return self.motor.get_slot()

