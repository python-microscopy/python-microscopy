import visa

class PowerMeter:
    def __init__(self, ID='USB0::0x1313::0x8072::P2000343', dispScale=14):
        self.instr = visa.instrument(ID)
        self.dispScale=dispScale

    def GetPower(self):
        return float(self.instr.ask('MEAS:POW?'))

    def GetWavelength(self):
        return float(self.instr.ask('SENS:CORR:WAV?'))

    def SetWavelength(self, wavelength):
        self.instr.write('SENS:CORR:WAV %d' % wavelength)

    def GetStatusText(self):
        return 'Laser power: %3.3f mW' % (self.GetPower()*1e3*self.dispScale)


