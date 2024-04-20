#!/usr/bin/python

###############
# AndorNeo.py
#
# Copyright David Baddeley, 2012
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
################


from .SDK3Cam import *
import numpy as np
import threading
import ctypes

try:
    import Queue
except ImportError:
    import queue as Queue
    
import time
import traceback

from PYME.misc.aligned_array import create_aligned_array

from PYME.IO import MetaDataHandler
from PYME.Acquire import eventLog

class AndorBase(SDK3Camera):
    numpy_frames=1
    MODE_CONTINUOUS = 1
    MODE_SINGLE_SHOT = 0
    
    validROIS = [(2048,2048,1, 1),
                 (2040,2048,1,1),
                 (2000,2000,24,24),
                 (1780,1760,140,140),
                 (1920,1080,50,500),
                 (1400,1040,320,500),
                 (520,512,825,1033),
                 (240,280,953,1177),
                 (140,128,1017,1225)]
    
    SimpleGainModes = {
        'low noise':
            { 'name' : '11-bit (low noise)', 'PEncoding' : 'Mono12' },
        'high capacity':
            { 'name' : '11-bit (high well capacity)', 'PEncoding' : 'Mono12' },
        'high dynamic range':
            { 'name' : '16-bit (low noise & high well capacity)', 'PEncoding' : 'Mono16' }}

    NoiseProperties = {
        'low noise': {
            'ReadNoise' : 1.1,
            'ElectronsPerCount' : 0.28,
            'NGainStages' : -1,
            'ADOffset' : 100, # check mean (or median) offset
            'DefaultEMGain' : 85,
            'SaturationThreshold' : 2**11-1#(2**16 -1) # check this is really 11 bit
        },
        'high capacity': {
            'ReadNoise' : 5.96,
            'ElectronsPerCount' : 6.97,
            'NGainStages' : -1,
            'ADOffset' : 100,
            'DefaultEMGain' : 85,
            'SaturationThreshold' : 2**11-1#(2**16 -1)         
        },
        'high dynamic range': {
            'ReadNoise' : 1.33,
            'ElectronsPerCount' : 0.5,
            'NGainStages' : -1,
            'ADOffset' : 100,
            'DefaultEMGain' : 85,
            'SaturationThreshold' : (2**16 -1)
        }}
    
    def __init__(self, camNum):
        #define properties
        self.CameraAcquiring = ATBool()
        self.SensorCooling = ATBool()
        
        self.AcquisitionStart = ATCommand()
        self.AcquisitionStop = ATCommand()
        
        self.CycleMode = ATEnum()
        self.ElectronicShutteringMode = ATEnum()
        self.FanSpeed = ATEnum()
        self.PreAmpGainChannel = ATEnum()
        self.PixelEncoding = ATEnum()
        self.PixelReadoutRate = ATEnum()
        self.PreAmpGain = ATEnum()
        self.PreAmpGainSelector = ATEnum()
        self.SimplePreAmpGainControl = ATEnum()
        self.TriggerMode = ATEnum()
        
        self.AOIHeight = ATInt()
        self.AOILeft = ATInt()
        self.AOITop = ATInt()
        self.AOIWidth = ATInt()
        self.FrameCount = ATInt()
        self.ImageSizeBytes = ATInt()
        self.SensorHeight = ATInt()
        self.SensorWidth = ATInt()
        
        self.CameraModel = ATString()
        self.SerialNumber = ATString()
        
        self.ExposureTime = ATFloat()
        self.FrameRate = ATFloat()
        self.SensorTemperature = ATFloat()
        self.TargetSensorTemperature = ATFloat()
        self.FullAOIControl = ATBool()
        
        SDK3Camera.__init__(self,camNum)
        
        #end auto properties
        
        self.camLock = threading.Lock()
        
        self.buffersToQueue = Queue.Queue()        
        self.queuedBuffers = Queue.Queue()
        self.fullBuffers = Queue.Queue()
        
        self.nQueued = 0
        self.nFull = 0
        
        self.nBuffers = 100
        
       
        
        self.contMode = True
        self.burstMode = False
        
        self._temp = 0
        self._frameRate = 0
        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)
        
    def Init(self):
        SDK3Camera.Init(self)        
        
        #set some intial parameters
        self.FrameCount.setValue(1)
        self.CycleMode.setString(u'Continuous')
        try:
            self.SetEMGain(0) # note that this sets 'high dynamic range' mode
        except:
            print("error setting gain mode")
            pass
        # spurious noise filter off by default
        try:
            self.SpuriousNoiseFilter.setValue(0) # this will also fail with the SimCams
        except:
            print("error disabling spurios noise filter")
            pass

        # Static Blemish Correction off by default
        try:
            self.StaticBlemishCorrection.setValue(0) # this will also fail with the SimCams
        except:
            print("error disabling Static Blemish Correction")
            pass
        
        self.SensorCooling.setValue(True)
        # Zyla does not like a temperature to be set
        if self.TemperatureControl.isImplemented() and self.TemperatureControl.isWritable():
            self.TemperatureControl.setString('-30.00')
        #self.PixelReadoutRate.setIndex(1)
        # test if we have only fixed ROIs
        self._fixed_ROIs = not self.FullAOIControl.isImplemented() or not self.FullAOIControl.getValue()
        self.noiseProps = self.NoiseProperties[self.GetGainMode()] # gain mode has been set via EMGain above

        # this one to deal with the requirement to have width divisible by 10
        if not self._fixed_ROIs:
            self.SetROI(1,1, self.GetCCDWidth(), self.GetCCDHeight())
        #set up polling thread        
        self.doPoll = False
        self.pollLoopActive = True
        self.pollThread = threading.Thread(target = self._pollLoop)
        self.pollThread.start()
        
        
        
    #Neo buffer helper functions    
        
    def InitBuffers(self):
        self._flush()
        bufSize = self.ImageSizeBytes.getValue()
        #print bufSize
        for i in range(self.nBuffers):
            #buf = np.empty(bufSize, 'uint8')
            buf = create_aligned_array(bufSize, 'uint8')
            self._queueBuffer(buf)
            
        self.doPoll = True
            
    def _flush(self):
        self.doPoll = False
        #purge camera buffers
        SDK3.Flush(self.handle)
        
        #purge our local queues
        while not self.queuedBuffers.empty():
            self.queuedBuffers.get()
            
        while not self.buffersToQueue.empty():
            self.buffersToQueue.get()
            
        self.nQueued = 0
            
        while not self.fullBuffers.empty():
            self.fullBuffers.get()
            
        self.nFull = 0
        #purge camera buffers
        SDK3.Flush(self.handle)
            
            
    def _queueBuffer(self, buf):
        #self.queuedBuffers.put(buf)
        #print np.base_repr(buf.ctypes.data, 16)
        #SDK3.QueueBuffer(self.handle, buf.ctypes.data_as(SDK3.POINTER(SDK3.AT_U8)), buf.nbytes)
        #self.nQueued += 1
        self.buffersToQueue.put(buf)
        
    def _queueBuffers(self):
        #self.camLock.acquire()
        while not self.buffersToQueue.empty():
            buf = self.buffersToQueue.get(block=False)
            self.queuedBuffers.put(buf)
            #print np.base_repr(buf.ctypes.data, 16)
            SDK3.QueueBuffer(self.handle, buf.ctypes.data_as(SDK3.POINTER(SDK3.AT_U8)), buf.nbytes)
            #self.fLog.write('%f\tq\n' % time.time())
            self.nQueued += 1
        #self.camLock.release()
        
    def _pollBuffer(self):
        try:
            #self.fLog.write('%f\tp\n' % time.time())
            pData, lData = SDK3.WaitBuffer(self.handle, 100)
            #self.fLog.write('%f\tb\n' % time.time())
        except SDK3.TimeoutError as e:
            #Both AT_ERR_TIMEDOUT and AT_ERR_NODATA
            #get caught as TimeoutErrors
            #if e.errNo == SDK3.AT_ERR_TIMEDOUT:
            #    self.fLog.write('%f\tt\n' % time.time())
            #else:
            #    self.fLog.write('%f\tn\n' % time.time())
            return
        #except SDK3.CameraError as e:
        #    if not e.errNo == SDK3.AT_ERR_NODATA:
        #        traceback.print_exc()
        #    return
            
        #self.camLock.acquire()
        buf = self.queuedBuffers.get()
        self.nQueued -= 1
        if not buf.ctypes.data == ctypes.addressof(pData.contents):
            print((ctypes.addressof(pData.contents), buf.ctypes.data))
            #self.camLock.release()
            raise RuntimeError('Returned buffer not equal to expected buffer')
            #print 'Returned buffer not equal to expected buffer'
            
        self.fullBuffers.put(buf)
        self.nFull += 1
        #self.camLock.release()
        
    def _pollLoop(self):
        #self.fLog = open('poll.txt', 'w')
        while self.pollLoopActive:
            self._queueBuffers()
            if self.doPoll: #only poll if an acquisition is running
                self._pollBuffer()
            else:
                #print 'w',
                time.sleep(.05)
            time.sleep(.0005)
            #self.fLog.flush()
        #self.fLog.close()
        
    #PYME Camera interface functions - make this look like the other cameras
    def ExpReady(self):
        #self._pollBuffer()
        
        return not self.fullBuffers.empty()
        
    def ExtractColor(self, chSlice, mode):
        #grab our buffer from the full buffers list
        buf = self.fullBuffers.get()
        self.nFull -= 1
        
        #copy to the current 'active frame' 
        #print chSlice.shape, buf.view(chSlice.dtype).shape
        #bv = buf.view(chSlice.dtype).reshape(chSlice.shape)
        #chSlice[:] = bv
        #chSlice[:,:] = bv
        ctypes.cdll.msvcrt.memcpy(chSlice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), chSlice.nbytes)
        #print 'f'
        
        #recycle buffer
        self._queueBuffer(buf)
        
    def GetSerialNumber(self):
        return self.SerialNumber.getValue()
    
    def SetIntegTime(self, iTime): 
        self.ExposureTime.setValue(iTime*1e-3)
        self.FrameRate.setValue(self.FrameRate.max())
        
    def GetIntegTime(self): 
        return self.ExposureTime.getValue()
    
    def GetCCDWidth(self): 
        # limit width to multiple of 10 - Zyla specific!!!
        # NOTE: we are not sure why only this size works
        if True:
            return self.SensorWidth.getValue()
        else:
            return 10*int(self.SensorWidth.getValue()/10)
    def GetCCDHeight(self): 
        return self.SensorHeight.getValue()
    

    def GetNumberChannels(*args): 
        raise Exception('Not implemented yet!!')
    
    def GetElectrTemp(*args): 
        return 25
        
    def GetCCDTemp(self):
        #for some reason querying the temperature takes a lot of time - do it less often
        #return self.SensorTemperature.getValue()
        
        return self._temp
    
    
    def GetPicWidth(self): 
        return self.AOIWidth.getValue()
    def GetPicHeight(self):
        
        return self.AOIHeight.getValue()
        
    def SetROIIndex(self, index):
        width, height, top, left = self.validROIS[index]
        
        #print 'in SetROIIndex'
        self.AOIWidth.setValue(width)
        self.AOILeft.setValue(left)
        self.AOIHeight.setValue(height)
        self.AOITop.setValue(top)

    def ROIsAreFixed(self):
        return self._fixed_ROIs

    def SetROI(self, x1, y1, x2, y2):
        #shouldn't do GUI stuff here, but quick way of making it work
        if self._fixed_ROIs:
            print('Setting ROI, using fixed sizes')
            import wx
            dlg = wx.SingleChoiceDialog(None, 'Please select the ROI size', 'Camera ROI', ['%dx%d at (%d, %d)' % roi for roi in self.validROIS])
            dlg.ShowModal()
            self.SetROIIndex(dlg.GetSelection())
            dlg.Destroy()
        else: # allow to select ROI size more freely
            print("free ROI choice")
            print("requested: ", x1,y1,x2-x1+1,y2-y1+1)
            if x1 < 1:
                x1 = 1
            if y1 < 1:
                y1 = 1
            # for some weird reason the width must be divisble by 10 on the zyla
            if (x1 > x2):
                xtmp = x2
                x2 = x1
                x1 = xtmp
            if (y1 > y2):
                ytmp = y2
                y2 = y1
                y1 = ytmp
            if x1 == x2:
                raise RuntimeError('Error Setting x ROI - Zero sized ROI')
            if y1 == y2:
                raise RuntimeError('Error Setting y ROI - Zero sized ROI')
            #if (x2-x1+1) == 2048:
            #    w10 = 2048  # allow full size
            #else:
            #    w10 = int((x2-x1+9)/10.0) * 10
            w10 = int((x2-x1+9)/10.0) * 10
            if x1+w10-1 > 2048: # x1+w10-1 should be index of largest row
                w10 -= 10
            h = y2-y1+1
            # now set as specified
            print("setting: ", x1,y1,w10,h)
            self.AOIWidth.setValue(w10)
            self.AOILeft.setValue(x1)
            self.AOIHeight.setValue(h)
            self.AOITop.setValue(y1)

    def SetGainMode(self,mode):
        from warnings import warn
        if not any(mode in s for s in self.SimpleGainModes.keys()):
            warn('invalid mode "%s" requested - ignored' % mode)
            return
        self._gainmode = mode
        self.SimplePreAmpGainControl.setString(self.SimpleGainModes[mode]['name'])
        self.PixelEncoding.setString(self.SimpleGainModes[mode]['PEncoding'])
        self.noiseProps = self.NoiseProperties[self._gainmode] # update noise properties for new mode
        
    def GetGainMode(self):
        return self._gainmode

    def GetROIX1(self):
        return self.AOILeft.getValue()
        
    def GetROIX2(self):
        return self.AOILeft.getValue() + self.AOIWidth.getValue()
        
    def GetROIY1(self):
        return self.AOITop.getValue()
        
    def GetROIY2(self):
        return self.AOITop.getValue() + self.AOIHeight.getValue()
    
    def DisplayError(*args): 
        pass

    #def Init(*args): 
    #    pass

    def Shutdown(self):
        self.pollLoopActive = False
        self.shutdown()
        #pass

    def GetStatus(*args): 
        pass
    
    def SetCOC(*args): 
        pass

    def StartExposure(self):
        #make sure no acquisition is running
        self.StopAq()
        self._temp = self.SensorTemperature.getValue()
        self._frameRate = self.FrameRate.getValue()
        self.tKin = 1.0 / self._frameRate
        
        eventLog.logEvent('StartAq', '')
        self._flush()
        self.InitBuffers()
        self.AcquisitionStart()

        return 0
        
    def StopAq(self):
        if self.CameraAcquiring.getValue():
            self.AcquisitionStop()
        

    def StartLifePreview(*args): 
        raise Exception('Not implemented yet!!')
    def StopLifePreview(*args): 
        raise Exception('Not implemented yet!!')

    def GetBWPicture(*args): 
        raise Exception('Not implemented yet!!')
    
    def CheckCoordinates(*args): 
        raise Exception('Not implemented yet!!')

    #new fcns for Andor compatibility
    def GetNumImsBuffered(self):
        return self.nFull
    
    def GetBufferSize(self):
        return self.nBuffers
        
    def SetActive(self, active=True):
        """flag the camera as active (or inactive) to dictate whether it writes it's metadata or not"""
        self.active = active

    def GenStartMetadata(self, mdh):
        if self.active:
            self.GetStatus()
    
            mdh.setEntry('Camera.Name', 'Andor sCMOS')
            mdh.setEntry('Camera.Model', self.CameraModel.getValue())
            mdh.setEntry('Camera.SerialNumber', self.GetSerialNumber())

            mdh.setEntry('Camera.IntegrationTime', self.GetIntegTime())
            mdh.setEntry('Camera.CycleTime', self.GetIntegTime())
            mdh.setEntry('Camera.EMGain', self.GetEMGain())
            mdh.setEntry('Camera.DefaultEMGain', 85) # needed for some protocols
    
            mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
            mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())
            mdh.setEntry('Camera.ROIOriginX', self.GetROIX1() - 1)
            mdh.setEntry('Camera.ROIOriginY', self.GetROIY1() - 1)
            mdh.setEntry('Camera.ROIWidth', self.GetROIX2() - self.GetROIX1())
            mdh.setEntry('Camera.ROIHeight',  self.GetROIY2() - self.GetROIY1())
            #mdh.setEntry('Camera.StartCCDTemp',  self.GetCCDTemp())

            # values for readnoise and EpC from Neo sheet for Gain 4
            # should be selected with 11 bit low noise setting
            np = self.NoiseProperties[self.GetGainMode()]
            mdh.setEntry('Camera.ReadNoise', np['ReadNoise'])
            mdh.setEntry('Camera.NoiseFactor', 1.0)
            mdh.setEntry('Camera.ElectronsPerCount', np['ElectronsPerCount'])
            mdh.setEntry('Camera.ADOffset', np['ADOffset'])
    
            #mdh.setEntry('Simulation.Fluorophores', self.fluors.fl)
            #mdh.setEntry('Simulation.LaserPowers', self.laserPowers)
    
            #realEMGain = ccdCalibrator.getCalibratedCCDGain(self.GetEMGain(), self.GetCCDTempSetPoint())
            #if not realEMGain == None:
            mdh.setEntry('Camera.TrueEMGain', 1)

            # this should not be needed anymore with the changes in NoiseProperties dict
            #update the scmos Metadata for different gain modes
            #if int(self.EMGain) == 0:
            #    mdh.setEntry('Camera.ElectronsPerCount', 0.5)
            #    mdh.setEntry('Camera.ReadNoise', 1.33)

            if  self.StaticBlemishCorrection.isImplemented():
                mdh.setEntry('Camera.StaticBlemishCorrection', self.StaticBlemishCorrection.getValue())
            if  self.SpuriousNoiseFilter.isImplemented():
                mdh.setEntry('Camera.SpuriousNoiseFilter', self.SpuriousNoiseFilter.getValue())


    #functions to make us look more like andor camera
    def GetEMGain(self):
        return self.EMGain

    def GetCCDTempSetPoint(self):
        return self.TargetSensorTemperature.getValue()

    def SetCCDTemp(self, temp):
        self.TargetSensorTemperature.setValue(temp)
        #pass

    def SetEMGain(self, gain):
        self.EMGain = gain
        if gain > 0:
            print('low noise mode')
            self.SetGainMode('low noise')
        else:
            print('high dynamic range mode')
            self.SetGainMode('high dynamic range')
    
    def SetAcquisitionMode(self, aqMode):
        self.CycleMode.setIndex(aqMode)
        self.contMode = aqMode == self.MODE_CONTINUOUS

    def SetBurst(self, burstSize):
        if burstSize > 1:
            self.SetAcquisitionMode(self.MODE_SINGLE_SHOT)
            self.FrameCount.setValue(burstSize)
            self.contMode = True
            self.burstMode = True
        else:
            self.FrameCount.setValue(1)
            self.SetAcquisitionMode(self.MODE_CONTINUOUS)
            self.burstMode = False

    def SetShutter(self, mode):
        pass

    def SetBaselineClamp(self, mode):
        pass
    
    def GetFPS(self):
        #return self.FrameRate.getValue()
        return self._frameRate

    def __del__(self):
        self.Shutdown()
        #self.compT.kill = True


class AndorNeo(AndorBase):              
    def __init__(self, camNum):
        #define properties
        self.Overlap = ATBool()
        self.SpuriousNoiseFilter = ATBool()
        self.StaticBlemishCorrection = ATBool()
        
        self.CameraDump = ATCommand()
        self.SoftwareTrigger = ATCommand()
        
        self.TemperatureControl = ATEnum()
        self.TemperatureStatus = ATEnum()
        self.PreAmpGainControl = ATEnum()
        self.BitDepth = ATEnum()
        
        self.ActualExposureTime = ATFloat()
        self.BurstRate = ATFloat()
        self.ReadoutTime = ATFloat()
        
        self.AccumulateCount = ATInt()
        self.BaselineLevel = ATInt()
        self.BurstCount = ATInt()
        self.LUTIndex = ATInt()
        self.LUTValue = ATInt()
        
        self.ControllerID = ATString()
        self.FirmwareVersion = ATString()
        self.active = True # we need that for metadata initialisation I believe, not sure if best to do here or elsewhere
        
        AndorBase.__init__(self,camNum)
        
class AndorSim(AndorBase):
    def __init__(self, camNum):
        #define properties
        self.SynchronousTriggering = ATBool()
        
        self.PixelCorrection = ATEnum()
        self.TriggerSelector = ATEnum()
        self.TriggerSource = ATEnum()
        
        self.PixelHeight = ATFloat()
        self.PixelWidth = ATFloat()
        
        self.AOIHBin = ATInt()
        self.AOIVbin = ATInt()
        
        AndorBase.__init__(self,camNum)

        

