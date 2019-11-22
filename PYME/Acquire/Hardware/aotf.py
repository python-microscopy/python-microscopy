
import numpy as np
import logging
logger = logging.getLogger(__name__)
from PYME.Acquire.Hardware.lasers import Laser
from scipy.interpolate import interp1d

class AOTFControlledLaser(Laser):
    """
    Shim to make an AOTF and laser function like a "Laser" object
    """
    power_controllable = True  # assume the AOTF can be used to control the power
    powerControlable = power_controllable  # fixme
    def __init__(self, laser, aotf, aotf_channel, chained_devices=None):
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

        self.power_output = self.GetLaserPower() * self.aotf._get_fractional_output(self.aotf_channel)
        self.MAX_AOTF_FRACTIONAL_OUTPUT = self.aotf.info[self.aotf_channel]['max_fractional_output']
        self.MIN_AOTF_FRACTIONAL_OUTPUT = self.aotf.info[self.aotf_channel]['min_fractional_output']
        self.MAX_POWER = self.laser.MAX_POWER * self.MAX_AOTF_FRACTIONAL_OUTPUT
        self.MIN_POWER = self.laser.MIN_POWER * self.MIN_AOTF_FRACTIONAL_OUTPUT


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
        self.power_output = self.GetLaserPower() * self.aotf._get_fractional_output(self.aotf_channel)

    def GetLaserPower(self):
        return self.laser.GetPower()

    def SetLaserPower(self, power):
        """

        Parameters
        ----------
        power: float
            Laser power in units dictated by the laser implementation in PYME, typically [mW], sometimes [W]. Note that
            the AOTFControlledLaser will not function correctly if the calibration units are different that the units
            used in self.laser.SerPower()

        Returns
        -------

        """
        self.laser.SetPower(power)
        self.laser_power = power
        self.update_power_output()

    def SetPowerToMin(self):
        self.SetPower(self.MIN_POWER)

    def GetAOTFPower(self):
        return self.aotf.GetPower(self.aotf_channel)

    def SetAOTFPower(self, power):
        """

        Parameters
        ----------
        power: float
            Power setting for the AOTF, typically [dBm], sometimes [V]

        Returns
        -------
        None

        """
        self.aotf.SetPower(power, self.aotf_channel)
        self.update_power_output()

    def SetPower(self, power):
        """

        Parameters
        ----------
        power: float
            Desired power output at the sample, in the same units as the calibration (typically mW)

        Returns
        -------
        None

        """
        # check if this is feasible
        if power > self.MAX_POWER:
            logger.error('%s maximum power output: %.1f, %.1f requested. Setting to maximum' % (self.name,
                                                                                                self.MAX_POWER,
                                                                                                power))
            power = self.MAX_POWER
        elif power < self.MIN_POWER:
            logger.error('%s minimum power output: %.1f, %.1f requested. Setting to minimum' % (self.name,
                                                                                                self.MIN_POWER,
                                                                                                power))
            power = self.MIN_POWER
        # change laser power if needed
        if power > self.MAX_AOTF_FRACTIONAL_OUTPUT * self.laser_power:
            # increase laser power
            self.SetLaserPower(power / self.MAX_AOTF_FRACTIONAL_OUTPUT)
        elif power < self.MIN_AOTF_FRACTIONAL_OUTPUT * self.laser_power:
            # decrease laser power
            self.SetLaserPower(power / self.MIN_AOTF_FRACTIONAL_OUTPUT)

        # calculate and set AOTF settings
        fractional_output = power / self.laser_power
        self.aotf._set_fractional_output(self.aotf_channel, fractional_output)
        self.update_power_output()

    def GetPower(self):
        return self.power_output

    def Close(self):
        try:
            self.TurnOff()
        except:
            pass
        self.laser.Close()

    def __del__(self):
        self.Close()

    def registerStateHandlers(self, scopeState):
        Laser.registerStateHandlers(self, scopeState)
        scopeState.registerHandler('Lasers.%s.AOTFSetting' % self.name, self.GetAOTFPower, self.SetAOTFPower)
        scopeState.registerHandler('Lasers.%s.LaserPower' % self.name, self.GetLaserPower, self.SetLaserPower)



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
            chan = int(chan)  # json format specs keys as str
            self.info[chan] = {}
            max_ind = np.argmax(calib['output'])
            peak_output = calib['output'][max_ind]
            normalized_output = np.asarray(calib['output']) / calib['laser_setting']
            self.info[chan]['fractional_output_at_setting'] = interp1d(calib['aotf_setting'], normalized_output)
            self.info[chan]['setting_for_fractional_output'] = interp1d(normalized_output, calib['aotf_setting'])
            self.info[chan]['peak_output_setting'] = calib['aotf_setting'][max_ind]
            self.info[chan]['max_fractional_output'] = peak_output / calib['laser_setting']
            self.info[chan]['min_fractional_output'] = np.min(calib['output']) / calib['laser_setting']


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

    def _get_fractional_output(self, channel):
        """

        Parameters
        ----------
        channel: int
            channel of the AOTF to query

        Returns
        -------
        fractional_output: float
            fractional output of the selected AOTF channel at the current setting
        """
        return self.info[channel]['fractional_output_at_setting']([self.GetPower(channel)])[0]

    def _set_fractional_output(self, channel, fractional_output):
        """
        Set the relative output of the AOTF using the stored calibration to translate from desired fractional output and
        the AOTF settings for that channel.

        Parameters
        ----------
        channel: int
            channel of the AOTF to query
        fractional_output: float
            Desired fractional output of the selected AOTF channel

        Returns
        -------
        None

        """
        setting = self.info[channel]['setting_for_fractional_output']([fractional_output])[0]
        self.SetPower(setting, channel)

    def SetFreq(self, frequency, channel):
        raise NotImplementedError

    def GetFreq(self, channel):
        raise NotImplementedError

    def Close(self):
        raise NotImplementedError

    def __del__(self):
        self.Close()

