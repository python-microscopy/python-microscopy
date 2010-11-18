import serial

class Cornerstone7400:
    def __init__(self, comPort = 0):
        self.com = serial.Serial(comPort, timeout=.1)

    def _command(self, command):
        self.com.write('%s\n' % command)
        self.com.flush()

        #throw away the echo
        self.com.readline()

    def _query(self, command):
        self.com.write('%s?\n' % command)
        self.com.flush()

        #throw away the echo
        self.com.readline()

        #grab the next line
        return self.com.readline()

    def setShutter(self, open=True):
        if open:
            self._command('SHUTTER O')
        else:
            self._command('SHUTTER C')

    def setWavelength(self, wavelength):
        self._command('GOWAVE %3.3f' % wavelength)

    def getWavelength(self):
        return float(self._query('WAVE'))