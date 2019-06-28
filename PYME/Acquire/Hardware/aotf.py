
import numpy as np
import logging
logger = logging.getLogger(__name__)
from PYME.Acquire.Hardware.lasers import Laser
from scipy.interpolate import interp1d
import warnings

class AOTFControlledLaser(Laser):
    """
    Shim to make an AOTF and laser function like a "Laser" object
    """
    def __init__(self, laser, aotf, aotf_channel, initial_laser_power=0, chained_devices=None):
        """

        Parameters
        ----------
        laser: PYME.Hardware.lasers.Laser
            An initialized laser
        aotf: PYME.Acquire.Hardware.aotf.AOTF
            An initialized AOTF
        aotf_channel: int
            Channel on the AOTF corresponding to the laser
        """
        self.laser = laser
        self.name = laser.name
        self.laser_power = self.laser.GetPower()
        self.laser_on = self.laser.IsOn()
        self.aotf = aotf
        self.aotf_channel = aotf_channel

        self.chained_devices = chained_devices if chained_devices is not None else []

        if initial_laser_power > 0:
            self.laser.TurnOn()
            self.laser.SetPower(initial_laser_power)
            self.laser_power = initial_laser_power

        self.power_output = 0
        self.MAX_AOTF_FRACTIONAL_OUTPUT = self.aotf.info[self.aotf_channel]['max_fractional_output']
        self.MAX_POWER = self.laser.MAX_POWER * self.MAX_AOTF_FRACTIONAL_OUTPUT


    def IsOn(self):
        return self.laser.IsOn() and self.aotf.IsOn(self.aotf_channel)

    def TurnOn(self):
        if not self.laser_on:
            self.laser.TurnOn()
        if not self.aotf.IsOn():
            self.aotf.TurnOn()
        [device.Notify(True) for device in self.chained_devices]
        self.aotf.Enable(self.aotf_channel)

    def TurnOff(self):
        [device.Notify(False) for device in self.chained_devices]
        self.aotf.Disable(self.aotf_channel)

    def update_power_output(self):
        self.power_output = self.GetLaserPower() * self.aotf.GetFractionalOutput(self.aotf_channel)

    def GetLaserPower(self):
        return self.laser.GetPower()

    def SetLaserPower(self, power):
        self.laser.SetPower(power)
        self.laser_power = power
        self.update_power_output()

    def GetAOTFPower(self):
        self.aotf.GetPower(self.aotf_channel)

    def SetAOTFPower(self, power):
        self.aotf.SetPower(power, self.aotf_channel)
        self.update_power_output()

    def SetPower(self, power):
        # check if this is feasible
        if power > self.MAX_POWER:
            logger.error('Laser %s maximum power is %f, cannot reach requested %f' % (self.name, self.MAX_POWER, power))
            return
        # change laser power if needed
        if power > self.MAX_AOTF_FRACTIONAL_OUTPUT * self.laser_power:
            # laser power needs to be increased
            self.SetLaserPower(max(self.laser_power * 1.5, self.laser.MAX_POWER))
        # TODO - do we ever want to decrease laser power again?

        fractional_output = power / self.laser_power
        self.aotf.SetFractionalOutput(self.aotf_channel, fractional_output)
        self.update_power_output()

    def GetPower(self):
        return self.power_output

    def Close(self):
        try:
            self.TurnOff()
        except:
            pass
        self.laser.Close()

    def IsPowerControllable(self):
        return True  # assume the AOTF can be used to controll the power

    def IsPowerControlable(self):
        warnings.warn('Use IsPowerControllable', DeprecationWarning)
        return self.IsPowerControllable()

    def __del__(self):
        self.Close()

    def registerStateHandlers(self, scopeState):
        scopeState.registerHandler('Lasers.%s.On' % self.name, self.IsOn, lambda v: self.TurnOn() if v else self.TurnOff())
        # if self.IsPowerControlable():
        scopeState.registerHandler('Lasers.%s.Power' % self.name, self.GetPower, self.SetPower)
        scopeState.registerHandler('Lasers.%s.AOTFSetting' % self.name, self.GetAOTFPower, self.SetAOTFPower)
        scopeState.registerHandler('Lasers.%s.LaserPower' % self.name, self.GetLaserPower, self.SetLaserPower)
        scopeState.registerHandler('Lasers.%s.MaxPower' % self.name, lambda : self.MAX_POWER)



class AOTF(object):

    def __init__(self, name, calibrations, n_chans):
        """
        Parameters
        ----------
        name: str
            Name of the AOTF
        calibrations: dict
            Key specifies channel (int) and item should be a dictionary with the following keys
                aotf_setting: list
                    Calibration values (dB or Volt, etc.) as the abscissas in an output vs aotf setting plot
                output: list
                    Calibration values encoding the laser power passed through the AOTF at various aotf settings.
                laser_setting: float
                    Power setting of the laser during calibration. Should have same units as "output"

        """
        self.name = name
        self.calibrations = calibrations
        self.n_chans = n_chans

        self.freq = [0] * n_chans
        self.power = [0] * n_chans
        self.is_on = False
        self.channel_enabled = [False] * n_chans

        self.info = {}
        for chan, calib in self.calibrations.items():
            self.info[chan] = {}
            max_ind = np.argmax(calib['output'])
            peak_output = calib['output'][max_ind]
            normalized_output = np.asarray(calib['output']) / peak_output
            self.info[chan]['fractional_output_at_setting'] = interp1d(calib['aotf_setting'], normalized_output,
                                                                       fill_value=(0, 0))
            self.info[chan]['setting_for_fractional_output'] = interp1d(normalized_output, calib['aotf_setting'],
                                                                        fill_value=(0, 0))
            self.info[chan]['peak_output_setting'] = calib['aotf_setting'][max_ind]
            self.info[chan]['max_fractional_output'] = peak_output / calib['laser_setting']


    def IsOn(self, channel=None):
        if channel is None:
            return self.is_on
        else:
            return self.channel_enabled[channel] and self.is_on

    def TurnOn(self):
        raise NotImplementedError

    def TurnOff(self):
        raise NotImplementedError

    def Enable(self, channel):
        raise NotImplementedError

    def Disable(self, channel):
        raise NotImplementedError

    def SetPower(self, power, channel):
        raise NotImplementedError

    def GetPower(self, channel):
        raise NotImplementedError

    def GetFractionalOutput(self, channel):
        return self.info[channel]['fractional_output_at_setting'](self.GetPower(channel))

    def SetFractionalOutput(self, channel, fractional_output):
        setting = self.info[channel]['setting_for_fractional_output'](fractional_output)
        self.SetPower(setting, channel)

    def SetFreq(self, frequency, channel):
        raise NotImplementedError

    def GetFreq(self, channel):
        raise NotImplementedError

    def Close(self):
        raise NotImplementedError

    def __del__(self):
        self.Close()

