import rend_im
import PYME.cSMI as example
import scipy

from PYME.Acquire import MetaDataHandler
from PYME.Acquire import eventLog

import pylab

import threading
#import processing
import time

from PYME.Acquire.Hardware import EMCCDTheory
from PYME.Acquire.Hardware import ccdCalibrator

class CDataStack(example.CDataStack):
    def getCurrentChannelSlice(self, curMemChn):
        return example.CDataStack_AsArray(self, curMemChn)[:,:,self.getZPos()]

class NoiseMaker:
    def __init__(self, QE=.8, ADGain=27.32, readoutNoise=109.8, EMGain=0, background=10., floor=967, shutterOpen = True, numGainElements=536, vbreakdown=6.6, temperature = -70.):
        self.QE = QE
        self.ElectronsPerCount = ADGain
        self.ReadoutNoise=readoutNoise
        self.EMGain=EMGain
        self.background = background
        self.ADOffset = floor
        self.NGainElements = numGainElements
        self.vbreakdown = vbreakdown
        self.temperature = temperature
        self.shutterOpen = shutterOpen

    def noisify(self, im):
        M = EMCCDTheory.M((80. + self.EMGain)/(255 + 80.), self.vbreakdown, self.temperature, self.NGainElements, 2.2)
        F2 = EMCCDTheory.FSquared(M, self.NGainElements)
        return self.ADOffset + M*scipy.random.poisson(int(self.shutterOpen)*(im + self.background)*self.QE*F2)/(self.ElectronsPerCount*scipy.sqrt(F2)) + self.ReadoutNoise*scipy.random.standard_normal(im.shape)/self.ElectronsPerCount





#calculate image in a separate thread to maintain GUI reponsiveness
class compThread(threading.Thread):
#class compThread(processing.Process):
    def __init__(self,XVals, YVals,zPiezo, zOffset, fluors, noisemaker, laserPowers, intTime, contMode = True, bufferlength=20, biplane = False, biplane_z = 500):
        threading.Thread.__init__(self)
        self.XVals = XVals
        self.YVals = YVals
        self.fluors = fluors
        #self.zPos = zPos
        self.laserPowers = laserPowers
        self.intTime = intTime
        self.noiseMaker = noisemaker
        self.contMode = contMode
        self.bufferlength = bufferlength
        self.buffer = pylab.zeros((len(XVals), len(YVals), bufferlength), 'uint16')
        self.bufferWritePos = 0
        self.bufferReadPos = 0
        self.numBufferedImages = 0

        self.biplane = biplane
        self.deltaZ = biplane_z

        self.zPiezo = zPiezo
        self.zOffset = zOffset

        self.kill = False
        self.aqRunning = False
        self.stopAq = False
        self.startAq = False

        print laserPowers
        print intTime

        #self.frameLock = threading.Lock()
        #self.frameLock.acquire()


    def run(self):
        #self.im = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, self.zPos,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime))[:,:].astype('uint16')

        while not self.kill:
            #self.frameLock.acquire()
            while ((not self.aqRunning) or (self.numBufferedImages > self.bufferlength/2.)) and (not self.kill) :
                time.sleep(.01)

            zPos = (self.zPiezo.GetPos() - self.zOffset)*1e3
                
            if not self.fluors == None and not 'spec' in self.fluors.fl.dtype.fields.keys():
                if self.biplane:
                    self.im = self.noiseMaker.noisify(rend_im.simPalmImFBP(self.XVals, self.YVals, zPos,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime, deltaZ = self.deltaZ))[:,:].astype('uint16')
                else:
                    self.im = self.noiseMaker.noisify(rend_im.simPalmImFI(self.XVals, self.YVals, zPos,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime))[:,:].astype('uint16')
            else:
                self.im = self.noiseMaker.noisify(rend_im.simPalmImFSpec(self.XVals, self.YVals, zPos,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime))[:,:].astype('uint16')

            self.buffer[:,:,self.bufferWritePos] = self.im
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
        im = self.buffer[:,:,self.bufferReadPos]
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

        




        
class FakeCamera:
    numpy_frames=1
    MODE_CONTINUOUS=True
    MODE_SINGLE_SHOT=False
    def __init__(self, XVals, YVals, noiseMaker, zPiezo, zOffset=50.0, fluors=None, laserPowers=[0,50]):
        self.XVals = XVals
        self.YVals = YVals

        self.ROIx = (0,len(XVals))
        self.ROIy = (0,len(YVals))

        self.zPiezo=zPiezo
        self.fluors=fluors
        self.noiseMaker=noiseMaker

        self.SaturationThreshold = (2**16) - 1

        self.laserPowers=laserPowers

        self.intTime=0.1
        self.zOffset = zOffset
        self.compT = None #thread which is currently being computed
        #self.compT = None #finished thread holding image (c.f. camera buffer)

        self.compT = compThread(self.XVals[self.ROIx[0]:self.ROIx[1]], self.YVals[self.ROIy[0]:self.ROIy[1]], self.zPiezo, self.zOffset,self.fluors, self.noiseMaker, laserPowers=self.laserPowers, intTime=self.intTime)
        self.compT.start()

        self.contMode = True
        self.shutterOpen = True

        #let us work with andor dialog
        self.HorizShiftSpeeds = [[[10]]]
        self.vertShiftSpeeds = [1]
        self.fastestRecVSInd = 0
        self.frameTransferMode = False
        self.HSSpeed = 0
        self.VSSpeed = 0

        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)

    def setFluors(self, fluors):
        self.fluors = fluors

        running = self.compT.aqRunning

        self.compT.kill = True

        self.compT = compThread(self.XVals[self.ROIx[0]:self.ROIx[1]], self.YVals[self.ROIy[0]:self.ROIy[1]], self.zPiezo, self.zOffset,self.fluors, self.noiseMaker, laserPowers=self.compT.laserPowers, intTime=self.intTime)
        self.compT.start()

        self.compT.aqRunning = running
        

    def GetCamType(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetDataType(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetADBits(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetMaxDigit(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetNumberCh(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetBytesPerPoint(*args):
        raise Exception, 'Not implemented yet!!'
    def GetCCDType(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetCamID(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetCamVer(*args): 
        raise Exception, 'Not implemented yet!!'
    def SetTrigMode(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetTrigMode(*args): 
        raise Exception, 'Not implemented yet!!'
    def SetDelayTime(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetDelayTime(*args): 
        raise Exception, 'Not implemented yet!!'
    
    def SetIntegTime(self, iTime): 
        self.intTime=iTime*1e-3
        self.compT.intTime = iTime*1e-3
    def GetIntegTime(self): 
        return self.intTime
    
    def SetROIMode(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetROIMode(*args): 
        raise Exception, 'Not implemented yet!!'
    def SetCamMode(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetCamMode(*args): 
        raise Exception, 'Not implemented yet!!'
    def SetBoardNum(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetBoardNum(*args): 
        raise Exception, 'Not implemented yet!!'
    
    def GetCCDWidth(self): 
        return len(self.XVals)
    def GetCCDHeight(self): 
        return len(self.YVals)
    
    def SetHorizBin(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetHorizBin(*args):
        return 0
        #raise Exception, 'Not implemented yet!!'
    def GetHorzBinValue(*args): 
        raise Exception, 'Not implemented yet!!'
    def SetVertBin(*args): 
        raise Exception, 'Not implemented yet!!'
    def GetVertBin(*args):
        return 0
        #raise Exception, 'Not implemented yet!!'
    def GetNumberChannels(*args): 
        raise Exception, 'Not implemented yet!!'
    
    def GetElectrTemp(*args): 
        return 25
    def GetCCDTemp(self):
        return self.noiseMaker.temperature
    
    def CamReady(*args): 
        return True
    
    def GetPicWidth(self): 
        return self.ROIx[1] - self.ROIx[0]
    def GetPicHeight(self): 
        return self.ROIy[1] - self.ROIy[0]

    def SetROI(self, x1, y1, x2, y2):
        self.ROIx = (x1, x2)
        self.ROIy = (y1, y2)

        running = self.compT.aqRunning

        self.compT.kill = True
        while self.compT.isAlive():
            time.sleep(0.01)

        print (self.fluors.fl['state'] == 2).sum()
        print running
        print self.compT.laserPowers

        self.compT = compThread(self.XVals[self.ROIx[0]:self.ROIx[1]], self.YVals[self.ROIy[0]:self.ROIy[1]], self.zPiezo, self.zOffset, self.fluors, self.noiseMaker, laserPowers=self.compT.laserPowers, intTime=self.intTime*1e-3)
        self.compT.start()

        print (self.fluors.fl['state'] == 2).sum()

        self.compT.aqRunning = running
    
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
        raise Exception, 'Not implemented yet!!'
    def StopLifePreview(*args): 
        raise Exception, 'Not implemented yet!!'

    def ExpReady(self):
        #return not self.compTCur.isAlive() #thread has finished -> a picture is available
        return self.compT.numFramesBuffered() > 0
        #return True
        #raise Exception, 'Not implemented yet!!'

    def GetBWPicture(*args): 
        raise Exception, 'Not implemented yet!!'
    
    def ExtractColor(self, chSlice, mode): 
        #im = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, self.zPiezo.GetPos() - self.zOffset,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime*1e-3))[:,:].astype('uint16')
        
        #chSlice[:,:] = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, (self.zPiezo.GetPos() - self.zOffset)*1e3,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime*1e-3))[:,:].astype('uint16')
	try:
	    chSlice[:,:] = self.compT.getIm() #grab image from completed computation thread
	    #self.compTOld = None #set computation thread to None such that we get an error if we try and obtain the same result twice
	except AttributeError:  # triggered if called with None
	    print "Grabbing problem: probably called with 'None' thread"
        #pylab.figure(2)
        #pylab.hist([f.state for f in self.fluors], [0, 1, 2, 3], hold=False)
        #pylab.gca().set_xticks([0.5,1.5,2.5,3.5])
        #pylab.gca().set_xticklabels(['Caged', 'On', 'Blinked', 'Bleached'])
        #pylab.show()
        
    def CheckCoordinates(*args): 
        raise Exception, 'Not implemented yet!!'

    #new fcns for Andor compatibility
    def GetNumImsBuffered(self):
        return self.compT.numFramesBuffered()
    
    def GetBufferSize(self):
        return self.compT.bufferlength

    def GenStartMetadata(self, mdh):
        self.GetStatus()

        mdh.setEntry('Camera.Name', 'Simulated EM CCD Camera')

        mdh.setEntry('Camera.IntegrationTime', self.GetIntegTime())
        mdh.setEntry('Camera.CycleTime', self.GetIntegTime())
        mdh.setEntry('Camera.EMGain', self.GetEMGain())

        mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
        mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())
        mdh.setEntry('Camera.ROIWidth', self.GetROIX2() - self.GetROIX1())
        mdh.setEntry('Camera.ROIHeight',  self.GetROIY2() - self.GetROIY1())
        #mdh.setEntry('Camera.StartCCDTemp',  self.GetCCDTemp())

        mdh.setEntry('Camera.ReadNoise', self.noiseMaker.ReadoutNoise)
        mdh.setEntry('Camera.NoiseFactor', 1.41)
        mdh.setEntry('Camera.ElectronsPerCount', self.noiseMaker.ElectronsPerCount)
        mdh.setEntry('Camera.ADOffset', self.noiseMaker.ADOffset)

        #mdh.setEntry('Simulation.Fluorophores', self.fluors.fl)
        #mdh.setEntry('Simulation.LaserPowers', self.laserPowers)

        realEMGain = ccdCalibrator.getCalibratedCCDGain(self.GetEMGain(), self.GetCCDTempSetPoint())
        if not realEMGain == None:
            mdh.setEntry('Camera.TrueEMGain', realEMGain)

    #functions to make us look more like andor camera
    def GetEMGain(self):
        return self.noiseMaker.EMGain

    def GetCCDTempSetPoint(self):
        return self.GetCCDTemp()

    def SetCCDTemp(self, temp):
        self.noiseMaker.temperature = temp
        #pass

    def SetEMGain(self, gain):
        self.noiseMaker.EMGain = gain
        #pass

    def SetAquisitionMode(self, mode):
        self.contMode = mode
        self.compT.contMode = mode

    def SetShutter(self, mode):
        self.shutterOpen = mode
        self.noiseMaker.shutterOpen = mode

    def SetBaselineClamp(self, mode):
        pass

    def __getattr__(self, name):
        if name in dir(self.noiseMaker):
            return self.noiseMaker.__dict__[name]
        else:  raise AttributeError, name  # <<< DON'T FORGET THIS LINE !!
        
    def __del__(self):
        self.compT.kill = True
