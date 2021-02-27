#!/usr/bin/python

##################
# fakeCam.py
#
# Copyright David Baddeley, 2009
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

from . import rend_im
import scipy

from PYME.IO import MetaDataHandler
from PYME.Acquire import eventLog

import numpy as np

import threading
#import processing
import time

import ctypes
import sys

if sys.platform == 'win32':
    memcpy = ctypes.cdll.msvcrt.memcpy
elif sys.platform == 'darwin':
    memcpy = ctypes.CDLL('libSystem.dylib').memcpy
else: #linux
    memcpy = ctypes.CDLL('libc.so.6').memcpy

from PYME.Acquire.Hardware import EMCCDTheory
from PYME.Acquire.Hardware import ccdCalibrator

import logging
logger = logging.getLogger(__name__)

def generate_camera_maps(size_x = 1024, size_y = 1024, seed=100, read_median=1.38, offset=100):
    """
    Generate camera maps for sCMOS simulation, using a constant random seed so that the maps are reproducible
    
    The use (and parameterization) of pareto distributions is designed to match the distribution of values observed in
    actual camera maps. Note that the pareto gives a somewhat better match than lognormal.
    
    Parameters
    ----------
    size_x
    size_y
    seed
    read_median
    offset

    Returns
    -------

    """
    
    np.random.seed(seed)
    
    #Variance
    s = 2.0
    #var = (np.random.lognormal(np.log(read_median), s, [size_x, size_y]))**2
    var = (read_median/(2**(1./s)) * (1 + np.random.pareto(s, [size_x, size_y]))) ** 2
    
    #the dark map has 3 components - a pareto distributed base distribution, a small ammount of Gaussian spread, and Gaussian distributed fixed pattern
    # line noise
    dark = offset + np.random.pareto(2.7, [size_x, size_y]) + np.random.normal(0, 1.8, [size_x, size_y]) + np.random.normal(0, 0.35, [size_x,])[:,None]
    
    flatfield = np.ones_like(dark)
    
    np.random.seed()
    return {'variance': var, 'dark':dark, 'flat' : flatfield}

class NoiseMaker:
    def __init__(self, QE=.8, electronsPerCount=27.32, readoutNoise=109.8, EMGain=0, background=0., floor=967, shutterOpen = True,
                 numGainElements=536, vbreakdown=6.6, temperature = -70., fast_read_approx=True):
        self.QE = QE
        self.ElectronsPerCount = electronsPerCount
        self.ReadoutNoise=readoutNoise
        self.EMGain=EMGain
        self.background = background
        self.ADOffset = floor
        self.NGainElements = numGainElements
        self.vbreakdown = vbreakdown
        self.temperature = temperature
        self.shutterOpen = shutterOpen
        
        self.approximate_read_noise = fast_read_approx #approximate readout noise
        
        self._ar_key = None
        self._ar_cache = None
        
    def _read_approx(self, im_shape):
        """
        Really dirty fast approximation to readout noise by indexing into a random location within a pre-calculated noise
        matrix. Note that this may result in undesired correlations in the read noise.
        
        Parameters
        ----------
        im_shape

        Returns
        -------

        """
        nEntries = int(np.prod(im_shape))
        ar_key = (nEntries, self.ADOffset, self.ReadoutNoise, self.ElectronsPerCount)
        
        if not self._ar_key == ar_key or self._ar_cache is None:
            self._ar_cache = self.ADOffset + (self.ReadoutNoise / self.ElectronsPerCount)*np.random.normal(size=2*nEntries)
            self._ar_key = ar_key
            
        offset = np.random.randint(0, nEntries)
        return self._ar_cache[offset:(offset+nEntries)].reshape(im_shape)

    def noisify(self, im):
        """Add noise to image using an EMCCD noise model
        
        Inputs
        ------
        
        im : NxM array of intensities (in photons)
        
        Outputs
        -------
        
        out: NxM array of simulated camera pixel intensities (in ADUs)
        
        """

        M = EMCCDTheory.M((80. + self.EMGain)/(255 + 80.), self.vbreakdown, self.temperature, self.NGainElements, 2.2)
        F2 = 1.0/EMCCDTheory.FSquared(M, self.NGainElements)

        if self.approximate_read_noise:
            o = self._read_approx(im.shape)
        else:
            o = self.ADOffset + (self.ReadoutNoise / self.ElectronsPerCount) * scipy.random.standard_normal(im.shape)
        
        if self.shutterOpen:
            o = o +  (M/(self.ElectronsPerCount*F2))*scipy.random.poisson((self.QE*F2)*(im + self.background))

        return o
        
    def getbg(self):
        M = EMCCDTheory.M((80. + self.EMGain)/(255 + 80.), self.vbreakdown, self.temperature, self.NGainElements, 2.2)
        F2 = 1.0/EMCCDTheory.FSquared(M, self.NGainElements)

        return self.ADOffset + M*(int(self.shutterOpen)*(0 + self.background)*self.QE*F2)/(self.ElectronsPerCount*F2) 





#calculate image in a separate thread to maintain GUI reponsiveness
class compThread(threading.Thread):
    def __init__(self,XVals, YVals,zPiezo, zOffset, fluors, noisemaker, laserPowers, intTime, contMode = True,
                 bufferlength=500, biplane = False, biplane_z = 500, xpiezo=None, ypiezo=None, illumFcn = 'ConstIllum'):
        #TODO - Do we need to change the default buffer length. This shouldn't really be an issue as we pause the simulation the buffer starts to fill up.
        
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
        self.buffer = np.zeros((bufferlength, len(XVals), len(YVals)), 'uint16')
        self.bufferWritePos = 0
        self.bufferReadPos = 0
        self.numBufferedImages = 0

        self.biplane = biplane
        self.deltaZ = biplane_z

        self.zPiezo = zPiezo
        self.zOffset = zOffset

        self.xPiezo = xpiezo
        self.yPiezo = ypiezo
        self.illumFcn = illumFcn

        self.kill = False
        self.aqRunning = False
        self.stopAq = False
        self.startAq = False

        #print(laserPowers)
        #print(intTime)

        #self.frameLock = threading.Lock()
        #self.frameLock.acquire()

    def setSplitterInfo(self, chan_z_offsets, chan_specs):
        self._chan_z_offsets = chan_z_offsets
        self._chan_specs = chan_specs

        nChans = len(chan_z_offsets)
        x_pixels = len(self.XVals)
        x_chan_pixels = x_pixels/nChans
        x_chan_size = (self.XVals[1] - self.XVals[0])*x_chan_pixels
        self._chan_x_offsets = [i*x_chan_size for i in range(nChans)]

    @property
    def ChanXOffsets(self):
        try:
            return getattr(self, '_chan_x_offsets')
        except AttributeError:
            if not self.fluors:
                return [0,]
            elif not self.biplane and not 'spec' in self.fluors.fl.dtype.fields.keys():
                return [0,]
            else:
                return [0, self.XVals[self.XVals.shape[0] / 2] - self.XVals[0]]

    @property
    def ChanZOffsets(self):
        try:
            return getattr(self, '_chan_z_offsets')
        except AttributeError:
            if not self.fluors:
                return [0, ]
            elif not self.biplane and not 'spec' in self.fluors.fl.dtype.fields.keys():
                return [0, ]
            else:
                return [0, self.deltaZ]

    @property
    def ChanSpecs(self):
        try:
            return getattr(self, '_chan_specs')
        except AttributeError:
            if not self.fluors:
                return None
            elif not 'spec' in self.fluors.fl.dtype.fields.keys():
                return None
            else:
                return [0,1]


    def run(self):
        #self.im = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, self.zPos,self.fluors,
        #                                   laserPowers=self.laserPowers, intTime=self.intTime))[:,:].astype('uint16')

        while not self.kill:
            #self.frameLock.acquire()
            while ((not self.aqRunning) or (self.numBufferedImages > self.bufferlength/2.)) and (not self.kill) :
                time.sleep(.01)

            zPos = (self.zPiezo.GetPos() - self.zOffset)*1e3

            xp = 0
            yp = 0
            if not self.xPiezo is None:
                xp = (self.xPiezo.GetPos() - self.xPiezo.max_travel/2)*1e3

            if not self.xPiezo is None:
                yp = (self.yPiezo.GetPos() - self.yPiezo.max_travel/2)*1e3

            #print self.ChanSpecs, self.ChanXOffsets
            
            r_i = rend_im.simPalmImFI(self.XVals + xp, self.YVals + yp, zPos,self.fluors,
                                                                  laserPowers=self.laserPowers, intTime=self.intTime,
                                                                  position=[xp,yp,zPos], illuminationFunction=self.illumFcn,
                                                                  ChanXOffsets=self.ChanXOffsets, ChanZOffsets=self.ChanZOffsets,
                                                                  ChanSpecs=self.ChanSpecs)
            
            # Bennet's empirical code modified this to set numSubSteps to 1. This breaks normal simulation (the current default of 10 substeps
            # is the bare minimum to get somewhat realistic behaviour, although there are still artifacts at numSubsteps=10).
            # TODO - is a numSubSteps of 1 important for the empirical simulation? I would imagine that the emprical code
            # should also use substepping (for much the same reason - to simulate sub frame on-times).
            
            r_i = r_i[:,:]
            _im = self.noiseMaker.noisify(r_i)
            self.im = _im.astype('uint16')

            self.buffer[self.bufferWritePos,:,:] = self.im
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
        im = np.copy(self.buffer[self.bufferReadPos,:,:], order='F')
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

        




from PYME.Acquire.Hardware.Camera import Camera
class FakeCamera(Camera):
    numpy_frames=1
    order= 'C'
    #MODE_CONTINUOUS=True
    #MODE_SINGLE_SHOT=False
    
    def __init__(self, XVals, YVals, noiseMaker, zPiezo, zOffset=50.0, fluors=None, laserPowers=[0,50], xpiezo=None, ypiezo=None, illumFcn = 'ConstIllum', pixel_size_nm=70.):
        if np.isscalar(XVals):
            self.SetSensorDimensions(XVals, YVals, pixel_size_nm, restart=False)
            self.pixel_size_nm = pixel_size_nm
        else:
            self.XVals = XVals
            self.YVals = YVals
    
            self.ROIx = (0,len(XVals))
            self.ROIy = (0,len(YVals))
            
            self.pixel_size_nm = XVals[1] - XVals[0]

        self.zPiezo=zPiezo
        self.xPiezo = xpiezo
        self.yPiezo = ypiezo
        self.fluors=fluors
        self.noiseMaker=noiseMaker

        self.SaturationThreshold = (2**16) - 1
        self.DefaultEMGain = 150

        self.laserPowers=laserPowers
        self.illumFcn = illumFcn

        self.intTime=0.1
        self.zOffset = zOffset
        self.compT = None #thread which is currently being computed
        #self.compT = None #finished thread holding image (c.f. camera buffer)

        #self.compT = compThread(self.XVals[self.ROIx[0]:self.ROIx[1]], self.YVals[self.ROIy[0]:self.ROIy[1]], self.zPiezo, self.zOffset,self.fluors, self.noiseMaker, laserPowers=self.laserPowers, intTime=self.intTime, xpiezo=self.xPiezo, ypiezo=self.yPiezo, illumFcn=self.illumFcn)
        #self.compT.start()
        self._restart_compT()
        #self._frameRate = 0

        self._acquisition_mode = self.MODE_CONTINUOUS
        #self.contMode = True
        self.shutterOpen = True

        #let us work with andor dialog
        self.HorizShiftSpeeds = [[[10]]]
        self.vertShiftSpeeds = [1]
        self.fastestRecVSInd = 0
        self.frameTransferMode = False
        self.HSSpeed = 0
        self.VSSpeed = 0

        self.active = True

        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)

    def setSplitterInfo(self, chan_z_offsets, chan_specs):
        self._chan_z_offsets = chan_z_offsets
        self._chan_specs = chan_specs


    def setFluors(self, fluors):
        self.fluors = fluors

        self._restart_compT()
        
    def SetSensorDimensions(self, x_size=256, y_size=256, pixel_size_nm=70., restart=True):
        self.XVals = pixel_size_nm*np.arange(0.0, float(x_size))
        self.YVals = pixel_size_nm * np.arange(0.0, float(y_size))
            
        self.ROIx = (0, len(self.XVals))
        self.ROIy = (0, len(self.YVals))
        
        if restart:
            self._restart_compT()

    def GetSerialNumber(self):
        return 0
    
    def SetIntegTime(self, iTime): 
        self.intTime=iTime#*1e-3
        self.compT.intTime = iTime#*1e-3
    def GetIntegTime(self): 
        return self.intTime
    
    def GetCCDWidth(self): 
        return len(self.XVals)
    def GetCCDHeight(self): 
        return len(self.YVals)
    
    def GetCCDTemp(self):
        return self.noiseMaker.temperature
    
    def GetPicWidth(self): 
        return self.ROIx[1] - self.ROIx[0]
    def GetPicHeight(self): 
        return self.ROIy[1] - self.ROIy[0]

    def SetROI(self, x1, y1, x2, y2):
        self.ROIx = (x1, x2)
        self.ROIy = (y1, y2)
        
        self._restart_compT()
        
    def _restart_compT(self):
        try:
            running = self.compT.aqRunning
            self.compT.kill = True
            while self.compT.isAlive():
                time.sleep(0.01)
                
        except AttributeError:
            running = False

        #print (self.fluors.fl['state'] == 2).sum()
        #print running
        #print self.compT.laserPowers

        self.compT = compThread(self.XVals[self.ROIx[0]:self.ROIx[1]], self.YVals[self.ROIy[0]:self.ROIy[1]],
                                self.zPiezo, self.zOffset, self.fluors, self.noiseMaker, laserPowers=self.laserPowers,
                                intTime=self.intTime, xpiezo=self.xPiezo, ypiezo=self.yPiezo, illumFcn=self.illumFcn)
        
        try:
            self.compT.setSplitterInfo(self._chan_z_offsets, self._chan_specs)
        except AttributeError:
            pass
        
        self.compT.start()

        #print (self.fluors.fl['state'] == 2).sum()

        self.compT.aqRunning = running
        
    def GetROI(self):
        return self.ROIx[0], self.ROIy[0], self.ROIx[1], self.ROIy[1]


    def Shutdown(self):
        self.compT.kill = True
        #pass

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


    def ExpReady(self):
        #return not self.compTCur.isAlive() #thread has finished -> a picture is available
        return self.compT.numFramesBuffered() > 0
 
    def ExtractColor(self, chSlice, mode): 
        #im = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, self.zPiezo.GetPos() - self.zOffset,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime*1e-3))[:,:].astype('uint16')

        #chSlice[:,:] = self.noiseMaker.noisify(rend_im.simPalmIm(self.XVals, self.YVals, (self.zPiezo.GetPos() - self.zOffset)*1e3,self.fluors, laserPowers=self.laserPowers, intTime=self.intTime*1e-3))[:,:].astype('uint16')
        try:
            d = self.compT.getIm()
            #print d.nbytes, chSlice.nbytes
            memcpy(chSlice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                   d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), chSlice.nbytes)
            #chSlice[:,:] = d #grab image from completed computation thread
            #self.compTOld = None #set computation thread to None such that we get an error if we try and obtain the same result twice
        except AttributeError:  # triggered if called with None
            logger.error("Grabbing problem: probably called with 'None' thread")
        #pylab.figure(2)
        #pylab.hist([f.state for f in self.fluors], [0, 1, 2, 3], hold=False)
        #pylab.gca().set_xticks([0.5,1.5,2.5,3.5])
        #pylab.gca().set_xticklabels(['Caged', 'On', 'Blinked', 'Bleached'])
        #pylab.show()
        
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

        #mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
        #mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())
        
        x1, y1, x2, y2 = self.GetROI()
        mdh.setEntry('Camera.ROIOriginX', x1)
        mdh.setEntry('Camera.ROIOriginY', y1)
        mdh.setEntry('Camera.ROIWidth', x2 - x1)
        mdh.setEntry('Camera.ROIHeight', y2 - y1)
        #mdh.setEntry('Camera.StartCCDTemp',  self.GetCCDTemp())

        mdh.setEntry('Camera.ReadNoise', self.noiseMaker.ReadoutNoise)
        mdh.setEntry('Camera.NoiseFactor', 1.41)
        mdh.setEntry('Camera.ElectronsPerCount', self.noiseMaker.ElectronsPerCount)
        mdh.setEntry('Camera.ADOffset', np.mean(self.noiseMaker.ADOffset))

        #mdh.setEntry('Simulation.Fluorophores', self.fluors.fl)
        #mdh.setEntry('Simulation.LaserPowers', self.laserPowers)

        realEMGain = ccdCalibrator.getCalibratedCCDGain(self.GetEMGain(), self.GetCCDTempSetPoint())
        if not realEMGain is None:
            mdh.setEntry('Camera.TrueEMGain', realEMGain)
            
        if self.fluors and 'spec' in self.fluors.fl.dtype.fields.keys(): #set the splitter parameters
            mdh['Splitter.Channel0ROI'] = [0,0,128, 256]
            mdh['Splitter.Channel1ROI'] = [128,0,128, 256]
            mdh['Splitter.Flip'] = False

        chan_specs = getattr(self, '_chan_specs', None)
        if not chan_specs is None:
            nChans  = len(chan_specs)
            x_pixels = len(self.XVals)
            x_chan_pixels = x_pixels / nChans
            y_pixels = len(self.YVals)
            mdh['Multiview.NumROIs'] = nChans
            mdh['Multiview.ROISize'] =  [x_chan_pixels, y_pixels]
            mdh['Multiview.ChannelColor'] =  list(chan_specs)
            for i in range(nChans):
                mdh['Multiview.ROI%dOrigin' % i] = [i*x_chan_pixels, 0]
                mdh['Splitter.Channel%dROI' % i] = [i*x_chan_pixels, 0, x_chan_pixels, y_pixels]

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

    def GetAcquisitionMode(self):
        return self._acquisition_mode
    
    def SetAcquisitionMode(self, mode):
        self._acquisition_mode = mode
        self.compT.contMode = (mode == self.MODE_CONTINUOUS)

    def SetShutter(self, mode):
        self.shutterOpen = mode
        self.noiseMaker.shutterOpen = mode

    def GetBaselineClamp(self):
        return True


    def SetIlluminationFcn(self, illumFcn):
        self.illumFcn = illumFcn
        self.compT.illumFcn = illumFcn

    def __getattr__(self, name):
        if name in dir(self.noiseMaker):
            return self.noiseMaker.__dict__[name]
        else:  raise AttributeError(name)  # <<< DON'T FORGET THIS LINE !!
        
    def __del__(self):
        self.Shutdown()
        #self.compT.kill = True
