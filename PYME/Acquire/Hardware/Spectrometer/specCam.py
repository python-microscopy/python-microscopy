#!/usr/bin/python

##################
# specCam.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################


import numpy as np

import time

from PYME.IO import MetaDataHandler
from PYME.Acquire import eventLog

from .spectrometerClient import SpecClient

import threading


#class CDataStack(example.CDataStack):
#    def getCurrentChannelSlice(self, curMemChn):
#        return example.CDataStack_AsArray(self, curMemChn)[:,:,self.getZPos()]


#calculate image in a separate thread to maintain GUI reponsiveness
class aqThread(threading.Thread):
    def __init__(self,spectrometer, bufferlength = 20):
        threading.Thread.__init__(self)
        self.spectrometer = spectrometer

        self.specLength = len(spectrometer.getSpectrum())

        self.bufferlength = bufferlength
        self.buffer = np.zeros((self.specLength, bufferlength), 'uint16')
        self.bufferWritePos = 0
        self.bufferReadPos = 0
        self.numBufferedImages = 0

        self.contMode=True

        self.kill = False
        self.aqRunning = False
        self.stopAq = False
        self.startAq = False#self.frameLock = threading.Lock()
        #self.frameLock.acquire()


    def run(self):
        while not self.kill:
            #self.frameLock.acquire()
            while ((not self.aqRunning) or (self.numBufferedImages > self.bufferlength/2.)) and (not self.kill) :
                time.sleep(.01)
            
            self.buffer[:,self.bufferWritePos] = self.spectrometer.getSpectrum().clip(0, 2**15).astype('uint16')
            self.bufferWritePos +=1
            if self.bufferWritePos >= self.bufferlength: #wrap around
                self.bufferWritePos = 0

            self.numBufferedImages = min(self.numBufferedImages +1, self.bufferlength)


            if not self.contMode:
                self.aqRunning = False

            if self.stopAq:
                self.aqRunning = False
                self.bufferWritePos = 0
                self.bufferReadPos = 0
                self.numBufferedImages = 0
                self.stopAq = False

            if self.startAq:
                self.aqRunning = True
                self.startAq = False


            #self.frameLock.release()

    def numFramesBuffered(self):
        return self.numBufferedImages

    def StartExp(self):
        self.bufferWritePos = 0
        self.bufferReadPos = 0
        self.numBufferedImages = 0
        self.aqRunning = True
        self.startAq = True
        #self.frameLock.release()

    def getIm(self):
        im = self.buffer[:,self.bufferReadPos]
        self.numBufferedImages -= 1
        self.bufferReadPos +=1
        if self.bufferReadPos >= self.bufferlength: #wrap around
            self.bufferReadPos = 0

        return im

    def StopAq(self):
        self.stopAq = True
#        self.aqRunning = False
#        self.bufferWritePos = 0
#        self.bufferReadPos = 0
#        self.numBufferedImages = 0

        




        
class SpecCamera:
    numpy_frames=1
    MODE_CONTINUOUS=True
    MODE_SINGLE_SHOT=False
    def __init__(self):
        spectrometer = SpecClient()
        self.XVals = spectrometer.getWavelengths()

        self.ROIx = (0,len(self.XVals))
        self.ROIy = (0,1)

        self.SaturationThreshold = (2**16) - 1

        self.intTime = 100

        self.compT = aqThread(spectrometer)
        self.compT.start()

        self.contMode = True
        #self.shutterOpen = True

#        #let us work with andor dialog
#        self.HorizShiftSpeeds = [[[10]]]
#        self.vertShiftSpeeds = [1]
#        self.fastestRecVSInd = 0
#        self.frameTransferMode = False
#        self.HSSpeed = 0
#        self.VSSpeed = 0

        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)    

    def GetCamType(*args): 
        raise Exception('Not implemented yet!!')
    def GetSerialNumber(self):
        return 0
    def GetDataType(*args): 
        raise Exception('Not implemented yet!!')
    def GetADBits(*args): 
        raise Exception('Not implemented yet!!')
    def GetMaxDigit(*args): 
        raise Exception('Not implemented yet!!')
    def GetNumberCh(*args): 
        raise Exception('Not implemented yet!!')
    def GetBytesPerPoint(*args):
        raise Exception('Not implemented yet!!')
    def GetCCDType(*args): 
        raise Exception('Not implemented yet!!')
    def GetCamID(*args): 
        raise Exception('Not implemented yet!!')
    def GetCamVer(*args): 
        raise Exception('Not implemented yet!!')
    def SetTrigMode(*args): 
        raise Exception('Not implemented yet!!')
    def GetTrigMode(*args): 
        raise Exception('Not implemented yet!!')
    def SetDelayTime(*args): 
        raise Exception('Not implemented yet!!')
    def GetDelayTime(*args): 
        raise Exception('Not implemented yet!!')
    
    def SetIntegTime(self, iTime): 
        self.intTime=iTime
        self.compT.spectrometer.setIntegrationTime(iTime)

    def SetAveraging(self, nAvg):
        self.compT.spectrometer.setScansToAverage(nAvg)

    def GetAveraging(self):
        return self.compT.spectrometer.getScansToAverage()

    def GetIntegTime(self): 
        return self.intTime
    
    def SetROIMode(*args): 
        raise Exception('Not implemented yet!!')
    def GetROIMode(*args): 
        raise Exception('Not implemented yet!!')
    def SetCamMode(*args): 
        raise Exception('Not implemented yet!!')
    def GetCamMode(*args): 
        raise Exception('Not implemented yet!!')
    def SetBoardNum(*args): 
        raise Exception('Not implemented yet!!')
    def GetBoardNum(*args): 
        raise Exception('Not implemented yet!!')
    
    def GetCCDWidth(self): 
        return len(self.XVals)
    def GetCCDHeight(self): 
        return 1
    
    def SetHorizBin(*args): 
        raise Exception('Not implemented yet!!')
    def GetHorizBin(*args):
        return 0
        #raise Exception, 'Not implemented yet!!'
    def GetHorzBinValue(*args): 
        raise Exception('Not implemented yet!!')
    def SetVertBin(*args): 
        raise Exception('Not implemented yet!!')
    def GetVertBin(*args):
        return 0
        #raise Exception, 'Not implemented yet!!'
    def GetNumberChannels(*args): 
        raise Exception('Not implemented yet!!')
    
    def GetElectrTemp(*args): 
        return 25
    def GetCCDTemp(self):
        return 25
    
    def GetPicWidth(self): 
        return self.ROIx[1] - self.ROIx[0]
    def GetPicHeight(self): 
        return self.ROIy[1] - self.ROIy[0]

    def SetROI(self, x1, y1, x2, y2):
        raise Exception('Not implemented yet!!')
    
    def GetROIX1(self):
        return self.ROIx[0]
        #raise Exception, 'Not implemented yet!!'
    def GetROIX2(self):
        return self.ROIx[1]
        #raise Exception, 'Not implemented yet!!'
    def GetROIY1(self):
        return self.ROIy[0]
        #raise Exception, 'Not implemented yet!!'
    def GetROIY2(self):
        return self.ROIy[1]
        #raise Exception, 'Not implemented yet!!'
    
    
    def DisplayError(*args): 
        pass

    def Init(*args): 
        pass

    def Shutdown(self):
        self.compT.kill = True
        #pass

    def GetStatus(*args): 
        pass
    
    def SetCOC(*args): 
        pass

    def StartAq(self):
        self.compT.StartExp()
        #pass

    def StopAq(self):
        self.compT.StopAq()
        #pass

    def StartExposure(self):
        eventLog.logEvent('StartAq', '')
        self.compT.StartExp()
        #self.compTOld = self.compTCur
        #self.compTCur = compThread(self.XVals, self.YVals, (self.zPiezo.GetPos() - self.zOffset)*1e3,self.fluors, self.noiseMaker, laserPowers=self.laserPowers, intTime=self.intTime*1e-3)
        #self.compTCur.start()
        return 0

    def StartLifePreview(*args): 
        raise Exception('Not implemented yet!!')
    def StopLifePreview(*args): 
        raise Exception('Not implemented yet!!')

    def ExpReady(self):
        #return not self.compTCur.isAlive() #thread has finished -> a picture is available
        return self.compT.numFramesBuffered() > 0
        #return True
        #raise Exception, 'Not implemented yet!!'

    def GetBWPicture(*args): 
        raise Exception('Not implemented yet!!')
    
    def ExtractColor(self, chSlice, mode): 
        #im = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, self.zPiezo.GetPos() - self.zOffset,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime*1e-3))[:,:].astype('uint16')

        #chSlice[:,:] = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, (self.zPiezo.GetPos() - self.zOffset)*1e3,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime*1e-3))[:,:].astype('uint16')
        try:
            chSlice[:,:] = self.compT.getIm()[:,None] #grab image from completed computation thread
            #self.compTOld = None #set computation thread to None such that we get an error if we try and obtain the same result twice
        except AttributeError:  # triggered if called with None
            print("Grabbing problem: probably called with 'None' thread")
        #pylab.figure(2)
        #pylab.hist([f.state for f in self.fluors], [0, 1, 2, 3], hold=False)
        #pylab.gca().set_xticks([0.5,1.5,2.5,3.5])
        #pylab.gca().set_xticklabels(['Caged', 'On', 'Blinked', 'Bleached'])
        #pylab.show()
        
    def CheckCoordinates(*args): 
        raise Exception('Not implemented yet!!')

    #new fcns for Andor compatibility
    def GetNumImsBuffered(self):
        return self.compT.numFramesBuffered()
    
    def GetBufferSize(self):
        return self.compT.bufferlength

    def GenStartMetadata(self, mdh):
        self.GetStatus()

        mdh.setEntry('Camera.Name', 'USB2000+ Spectrometer')

        mdh.setEntry('Camera.IntegrationTime', self.GetIntegTime())
        mdh.setEntry('Camera.CycleTime', self.GetIntegTime())
        mdh.setEntry('Camera.EMGain', self.GetEMGain())

        mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
        mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())
        mdh.setEntry('Camera.ROIWidth', self.GetROIX2() - self.GetROIX1())
        mdh.setEntry('Camera.ROIHeight',  self.GetROIY2() - self.GetROIY1())
        #mdh.setEntry('Camera.StartCCDTemp',  self.GetCCDTemp())

        mdh.setEntry('Camera.ReadNoise', 1)
        mdh.setEntry('Camera.NoiseFactor', 1.41)
        mdh.setEntry('Camera.ElectronsPerCount', 1)
        mdh.setEntry('Camera.ADOffset', 0)

        mdh.setEntry('Camera.Averaging', self.GetAveraging())
        mdh.setEntry('Camera.ElectricDarkCorrect', self.compT.spectrometer.getCorrectForElectricalDark())
        mdh.setEntry('Camera.NonlinearityCorrect', self.compT.spectrometer.getCorrectForDetectorNonlinearity())

        mdh.setEntry('Spectrum.Wavelengths', self.XVals)

        #mdh.setEntry('Simulation.Fluorophores', self.fluors.fl)
        #mdh.setEntry('Simulation.LaserPowers', self.laserPowers)

        
        mdh.setEntry('Camera.TrueEMGain', 1)

    #functions to make us look more like andor camera
    def GetEMGain(self):
        return 1

    def GetCCDTempSetPoint(self):
        return self.GetCCDTemp()

    def SetCCDTemp(self, temp):
        #self.noiseMaker.temperature = temp
        pass

    def SetEMGain(self, gain):
        #self.noiseMaker.EMGain = gain
        pass

    def SetAquisitionMode(self, mode):
        self.contMode = mode
        self.compT.contMode = mode

    def SetShutter(self, mode):
        #self.shutterOpen = mode
        #self.noiseMaker.shutterOpen = mode
        pass

    def SetBaselineClamp(self, mode):
        pass


    #def __getattr__(self, name):
    #    if name in dir(self.noiseMaker):
    #        return self.noiseMaker.__dict__[name]
    #    else:  raise AttributeError, name  # <<< DON'T FORGET THIS LINE !!
        
    def __del__(self):
        self.Shutdown()
        #self.compT.kill = True
