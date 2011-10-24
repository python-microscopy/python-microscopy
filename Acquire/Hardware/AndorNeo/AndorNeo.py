# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:13:16 2011

@author: dbad004
"""

from SDK3Cam import *
import numpy as np
#import threading
import ctypes
import Queue

from PYME.Acquire import MetaDataHandler
from PYME.Acquire import eventLog

class AndorBase(SDK3Camera):
    numpy_frames=1
    MODE_CONTINUOUS = 1
    MODE_SINGLE_SHOT = 0
    
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
        
        SDK3Camera.__init__(self,camNum)
        
        #end auto properties
        
        self.queuedBuffers = Queue.Queue()
        self.fullBuffers = Queue.Queue()
        
        self.nQueued = 0
        self.nFull = 0
        
        self.nBuffers = 100
        
        self.doPoll = True
        
        #set some intial parameters
        self.FrameCount.setValue(1)
        self.CycleMode.setString(u'Continuous')
        
        self.contMode = True
        
        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)
        
    #Neo buffer helper functions    
        
    def InitBuffers(self):
        self._flush()
        bufSize = self.ImageSizeBytes.getValue()
        #print bufSize
        for i in range(self.nBuffers):
            buf = np.empty(bufSize, 'uint8')
            self._queueBuffer(buf)
            
    def _flush(self):
        #purge camera buffers
        SDK3.Flush(self.handle)
        
        #purge our local queues
        while not self.queuedBuffers.empty():
            self.queuedBuffers.get()
            
        self.nQueued = 0
            
        while not self.fullBuffers.empty():
            self.fullBuffers.get()
            
        self.nFull = 0
            
            
    def _queueBuffer(self, buf):
        self.queuedBuffers.put(buf)
        #print np.base_repr(buf.ctypes.data, 16)
        SDK3.QueueBuffer(self.handle, buf.ctypes.data_as(SDK3.POINTER(SDK3.AT_U8)), buf.nbytes)
        self.nQueued += 1
        
    def _pollBuffer(self):
        try:
            pData, lData = SDK3.WaitBuffer(self.handle, 10)
        except RuntimeError:
            return
            
        buf = self.queuedBuffers.get()
        self.nQueued -= 1
        if not buf.ctypes.data == ctypes.addressof(pData.contents):
            print ctypes.addressof(pData.contents), buf.ctypes.data
            raise RuntimeError('Returned buffer not equal to expected buffer')
            #print 'Returned buffer not equal to expected buffer'
            
        self.fullBuffers.put(buf)
        self.nFull += 1
        
    #PYME Camera interface functions
    def ExpReady(self):
        self._pollBuffer()
        
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
        
        #recycle buffer
        self._queueBuffer(buf)
        
    def GetSerialNumber(self):
        return self.SerialNumber.getValue()
    
    def SetIntegTime(self, iTime): 
        self.ExposureTime.setValue(iTime*1e-3)
        
    def GetIntegTime(self): 
        return self.ExposureTime.getValue()
    
    def GetCCDWidth(self): 
        return self.SensorWidth.getValue()
    def GetCCDHeight(self): 
        return self.SensorHeight.getValue()
    
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
        return self.SensorTemperature.getValue()
    
    def CamReady(*args): 
        return True
    
    def GetPicWidth(self): 
        return self.AOIWidth.getValue()
    def GetPicHeight(self):
        
        return self.AOIHeight.getValue()

    def SetROI(self, x1, y1, x2, y2):
        self.AOILeft.setValue(x1)
        self.AOITop.setValue(y1)
        self.AOIWidth.setValue(x2-x1)
        self.AOIHeight.setValue(y2 - y1)
    
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

    def Init(*args): 
        pass

    def Shutdown(self):
        self.shutdown()
        #pass

    def GetStatus(*args): 
        pass
    
    def SetCOC(*args): 
        pass

    def StartExposure(self):
        #make sure no acquisiton is running
        self.StopAq()
        
        eventLog.logEvent('StartAq', '')
        self._flush()
        self.InitBuffers()
        self.AcquisitionStart()

        return 0
        
    def StopAq(self):
        if self.CameraAcquiring.getValue():
            self.AcquisitionStop()
        

    def StartLifePreview(*args): 
        raise Exception, 'Not implemented yet!!'
    def StopLifePreview(*args): 
        raise Exception, 'Not implemented yet!!'

    def GetBWPicture(*args): 
        raise Exception, 'Not implemented yet!!'
    
    def CheckCoordinates(*args): 
        raise Exception, 'Not implemented yet!!'

    #new fcns for Andor compatibility
    def GetNumImsBuffered(self):
        return self.nFull
    
    def GetBufferSize(self):
        return self.nBuffers

    def GenStartMetadata(self, mdh):
        self.GetStatus()

        mdh.setEntry('Camera.Name', 'Andor Neo')

        mdh.setEntry('Camera.IntegrationTime', self.GetIntegTime())
        mdh.setEntry('Camera.CycleTime', self.GetIntegTime())
        mdh.setEntry('Camera.EMGain', 1)

        mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
        mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())
        mdh.setEntry('Camera.ROIWidth', self.GetROIX2() - self.GetROIX1())
        mdh.setEntry('Camera.ROIHeight',  self.GetROIY2() - self.GetROIY1())
        #mdh.setEntry('Camera.StartCCDTemp',  self.GetCCDTemp())

        mdh.setEntry('Camera.ReadNoise', 1)
        mdh.setEntry('Camera.NoiseFactor', 1)
        mdh.setEntry('Camera.ElectronsPerCount', 1)
        #mdh.setEntry('Camera.ADOffset', self.noiseMaker.ADOffset)

        #mdh.setEntry('Simulation.Fluorophores', self.fluors.fl)
        #mdh.setEntry('Simulation.LaserPowers', self.laserPowers)

        #realEMGain = ccdCalibrator.getCalibratedCCDGain(self.GetEMGain(), self.GetCCDTempSetPoint())
        #if not realEMGain == None:
        mdh.setEntry('Camera.TrueEMGain', 1)

    #functions to make us look more like andor camera
    def GetEMGain(self):
        return 1

    def GetCCDTempSetPoint(self):
        return self.TargetSensorTemperature.getValue()

    def SetCCDTemp(self, temp):
        self.TargetSensorTemperature.setValue(temp)
        #pass

    def SetEMGain(self, gain):
        pass
    
    def SetAcquisitionMode(self, aqMode):
        self.CycleMode.setIndex(aqMode)
        self.contMode = aqMode == self.MODE_CONTINUOUS

    def SetShutter(self, mode):
        pass

    def SetBaselineClamp(self, mode):
        pass
    
    def GetFPS(self):
        return self.FrameRate.getValue()
        
    def __del__(self):
        self.Shutdown()
        #self.compT.kill = True

        
        
        
        
class AndorNeo(AndorBase):
    def __init__(self, camNum):
        #define properties
        self.Overlap = ATBool()
        self.SpuriousNoiseFilter = ATBool()
        
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
        
        
        
        