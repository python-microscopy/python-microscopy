
from PYME.Acquire.Hardware.lasers import Laser
from PYME.Acquire.Hardware.picoquant import pdlm

class Taiko(Laser):
    powerControlable = True
    # MAX_POWER = 0.055
    # MIN_POWER = 0
    units='W'

    def __init__(self, device_id, name, turnOn=False, scopeState=None):
        self.id = device_id
        pdlm.open_device(self.id)
        self._min_power_cw, self._max_power_cw = pdlm.get_cw_power_limits(self.id)
        self._min_power_pulsed, self._max_power_pulsed = pdlm.get_pulsed_power_limits(self.id)
        self.mode = self.GetEmissionMode()
        Laser.__init__(self, name, turnOn, scopeState)
    
    @property
    def MAX_POWER(self):
        if self.mode == 'cw':
            return self._max_power_cw
        elif self.mode == 'pulsed':
            return self._max_power_pulsed
    
    @property
    def MIN_POWER(self):
        if self.mode == 'cw':
            return self._min_power_cw
        elif self.mode == 'pulsed':
            return self._min_power_pulsed

    def IsOn(self):
        return not pdlm.get_lock_status(self.id)

    def TurnOn(self):
        pdlm.set_softlock(self.id, False)

    def TurnOff(self):
        pdlm.set_softlock(self.id, True)

    def SetPower(self, power):
        if self.mode == 'cw':
            return pdlm.set_cw_power(self.id, power)
        else:
            return pdlm.set_pulsed_power(self.id, power)

    def GetPower(self):
        if self.mode == 'cw':
            return pdlm.get_cw_power(self.id)
        else:
            return pdlm.get_pulsed_power(self.id)

    def SetEmissionMode(self, mode='cw'):
        if mode == 'cw':
            pdlm.set_emission_mode(self.id, pdlm.PDLM_LASER_MODE_CW)
            self.mode = 'cw'
        elif mode == 'pulsed':
            pdlm.set_emission_mode(self.id, pdlm.PDLM_LASER_MODE_PULSE)
            self.mode = 'pulsed'
        else:
            raise RuntimeError('Supported modes: cw, pulsed. Burst mode not implemented')
    
    def GetEmissionMode(self):
        return pdlm.get_emission_mode(self.id)

    def register(self, scope):
        scope.lasers.append(self)
        self.registerStateHandlers(scope.state)
    
    def GetPulseStatus(self):
        if self.mode == 'pulsed':
            return self.GetPulseShape(self.id)
        elif self.mode == 'cw':
            return 'CW'
    
    def GetTriggerMode(self):
        if self.mode == 'cw':
            return 'CW'
        else:
            return pdlm.get_trigger_mode(self.id)
    
    def GetPulseFreq(self):
        if self.mode == 'cw':
            return 0
        mode = self.GetTriggerMode()
        if 'INTERNAL' == pdlm.TRIGGER_MODE[mode]:
            return pdlm.get_frequency(self.id)
        else:
            # external trigger
            return pdlm.get_ext_trigger_freq(self.id)
    
    def SetFastGate(self, enable, high_impedance=True):
        pdlm.set_fast_gate(self.id, enable, high_impedance)
        
    def registerStateHandlers(self, scopeState):
        Laser.registerStateHandlers(self, scopeState)
        scopeState.registerHandler('Lasers.%s.EmissionMode', self.name, lambda : self.GetEmissionMode)
        scopeState.registerHandler('Lasers.%s.PulseStatus', self.name, lambda : self.GetPulseStatus)
        scopeState.registerHandler('Lasers.%s.TriggerMode', self.name, lambda : self.GetTriggerMode)
        scopeState.registerHandler('Lasers.%s.PulseFreqHz', self.name, lambda: self.GetPulseFreq)
    
    def GetPulseShape(self):
        return pdlm.get_pulse_shape(self.id)
    
    def close(self):
        self.TurnOff()
        pdlm.close_device(self.id)
        