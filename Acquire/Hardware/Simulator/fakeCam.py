import rend_im
import PYME.cSMI as example
import scipy

from PYME.Acquire import MetaDataHandler

import pylab

import threading

class CDataStack(example.CDataStack):
    def getCurrentChannelSlice(self, curMemChn):
        return example.CDataStack_AsArray(self, curMemChn)[:,:,self.getZPos()]

class NoiseMaker:
    def __init__(self, QE=.5, ADGain=2, readoutNoise=4, EMGain=1, background=10, floor=30):
        self.QE = QE
        self.ADGain = ADGain
        self.readoutNoise=4
        self.EMGain=1
        self.background = background
        self.floor = floor

    def noisify(self, im):
        return self.floor + scipy.random.poisson((im + self.background)*self.QE*self.EMGain)/self.ADGain + self.readoutNoise*scipy.random.standard_normal(im.shape)





#calculate image in a separate thread to maintain GUI reponsiveness
class compThread(threading.Thread):
    def __init__(self,XVals, YVals,zPos, fluors, noisemaker, laserPowers, intTime):
        threading.Thread.__init__(self)
        self.XVals = XVals
        self.YVals = YVals
        self.fluors = fluors
        self.zPos = zPos
        self.laserPowers = laserPowers
        self.intTime = intTime
        self.noiseMaker = noisemaker

    def run(self):
        #self.im = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, self.zPos,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime))[:,:].astype('uint16')

        if not self.fluors == None and not 'spec' in self.fluors.fl.dtype.fields.keys():
            self.im = self.noiseMaker.noisify(rend_im.simPalmImF(self.XVals, self.YVals, self.zPos,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime))[:,:].astype('uint16')
        else:
            self.im = self.noiseMaker.noisify(rend_im.simPalmImFSpec(self.XVals, self.YVals, self.zPos,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime))[:,:].astype('uint16')

        
class FakeCamera:
    numpy_frames=1
    def __init__(self, XVals, YVals, noiseMaker, zPiezo, zOffset=50.0, fluors=None, laserPowers=[0,50]):
        self.XVals = XVals
        self.YVals = YVals

        self.ROIx = (0,len(XVals))
        self.ROIy = (0,len(YVals))

        self.zPiezo=zPiezo
        self.fluors=fluors
        self.noiseMaker=noiseMaker

        self.laserPowers=laserPowers

        self.intTime=100
        self.zOffset = zOffset
        self.compTCur = None #thread which is currently being computed
        self.compTOld = None #finished thread holding image (c.f. camera buffer)

        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)

    def setFluors(self, fluors):
        self.fluors = fluors

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
        self.intTime=iTime
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
    def GetCCDTemp(*args): 
        return -11
    
    def CamReady(*args): 
        return True
    
    def GetPicWidth(self): 
        return self.ROIx[1] - self.ROIx[0]
    def GetPicHeight(self): 
        return self.ROIy[1] - self.ROIy[0]

    def SetROI(*args): 
        raise Exception, 'Not implemented yet!!'
    
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

    def StartExposure(self):
        self.compTOld = self.compTCur
        self.compTCur = compThread(self.XVals, self.YVals, (self.zPiezo.GetPos() - self.zOffset)*1e3,self.fluors, self.noiseMaker, laserPowers=self.laserPowers, intTime=self.intTime*1e-3)
        self.compTCur.start()
        return 0

    def StartLifePreview(*args): 
        raise Exception, 'Not implemented yet!!'
    def StopLifePreview(*args): 
        raise Exception, 'Not implemented yet!!'

    def ExpReady(self):
        return not self.compTCur.isAlive() #thread has finished -> a picture is available
        #return True
        #raise Exception, 'Not implemented yet!!'

    def GetBWPicture(*args): 
        raise Exception, 'Not implemented yet!!'
    
    def ExtractColor(self, chSlice, mode): 
        #im = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, self.zPiezo.GetPos() - self.zOffset,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime*1e-3))[:,:].astype('uint16')
        
        #chSlice[:,:] = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, (self.zPiezo.GetPos() - self.zOffset)*1e3,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime*1e-3))[:,:].astype('uint16')
	try:
	    chSlice[:,:] = self.compTOld.im #grab image from completed computation thread
	    self.compTOld = None #set computation thread to None such that we get an error if we try and obtain the same result twice
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
        return 1
    
    def GetBufferSize(self):
        return 10

    def GenStartMetadata(self, mdh):
        self.GetStatus()

        mdh.setEntry('Camera.Name', 'Simulated Standard CCD Camera')

        mdh.setEntry('Camera.IntegrationTime', self.GetIntegTime())
        mdh.setEntry('Camera.CycleTime', self.GetIntegTime())
        mdh.setEntry('Camera.EMGain', self.noiseMaker.EMGain)

        mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
        mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())
        #mdh.setEntry('Camera.StartCCDTemp',  self.GetCCDTemp())

        mdh.setEntry('Camera.ReadNoise', self.noiseMaker.readoutNoise)
        mdh.setEntry('Camera.NoiseFactor', 1)
        mdh.setEntry('Camera.ElectronsPerCount', self.noiseMaker.ADGain)
