import serial
import time
import lasers

class Cornerstone7400(lasers.Laser):
    def __init__(self, comPort = 0, minWave = 300, maxWave=1100, name='Monocromator'):
        self.com = serial.Serial(comPort, timeout=.1)

        lasers.Laser.__init__(self, name)

        #self.lastPos = 0
        #self.lastPos = self.GetPos() #for piezo compat
        self.minW = minWave
        self.maxW = maxWave

    def _command(self, command):
        self.com.flush()
        time.sleep(0.05)
        self.com.write('%s\n' % command)
        self.com.flushOutput()
        time.sleep(0.05)

        #throw away the echo
        #print self.com.readline()

        #return the status byte
        #return int(self.com.readline())

    def _query(self, command):
        self.com.flush()
        time.sleep(0.05)
        cmd = '%s?\n' % command
        self.com.write(cmd)
        self.com.flushOutput()
        time.sleep(0.05)

        #throw away the echo
        #self.com.readline()

        line = cmd

        while line == cmd:
            line = self.com.readline()

        #grab the next line
        return line

    def setShutter(self, open=True):
        if open:
            self._command('SHUTTER O')
        else:
            self._command('SHUTTER C')

    def getShutter(self):
        return self._query('SHUTTER').strip() == 'O'

    def setWavelength(self, wavelength):
        self._command('GOWAVE %3.3f' % wavelength)

    def getWavelength(self):
        try:
            return float(self._query('WAVE'))
        finally:
            if 'lastPos' in dir(self):
                return self.lastPos
            else:
                return 0

    #################################
    #'Piezo' compatibility functions
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        self.lastPos = fPos
        self.setWavelength(fPos)

    def GetPos(self, iChannel=0):
        return self.getWavelength()

    def GetControlReady(self):
        return True
    def GetChannelObject(self):
        return 1
    def GetChannelPhase(self):
        return 1
    def GetMin(self,iChan=1):
        return self.minW
    def GetMax(self, iChan=1):
        return self.maxW

    ################################
    #'Laser' compatibility functions (to enable shutter)

    def IsOn(self):
        return self.getShutter()

    def TurnOn(self):
        self.setShutter(True)

    def TurnOff(self):
        self.setShutter(False)