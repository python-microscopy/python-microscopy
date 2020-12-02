#!/usr/bin/python

###############
# AndorZyla.py
#
# Copyright David Baddeley, CS, 2015
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
import os
import logging

try:
    import Queue
except ImportError:
    import queue as Queue
    
import time
import traceback
from PYME.misc.aligned_array import create_aligned_array
from PYME.IO import MetaDataHandler
from PYME.Acquire import eventLog
from PYME.Acquire.Hardware.Camera import CameraMapMixin, MultiviewCameraMixin

logger = logging.getLogger(__name__)


class AndorBase(SDK3Camera, CameraMapMixin):
    numpy_frames=1
    #MODE_CONTINUOUS = 1
    #MODE_SINGLE_SHOT = 0

    DefaultPixelEncodingForGain = {'12-bit (low noise)': 'Mono12',
                                   '12-bit (high well capacity)': 'Mono12',
                                   '16-bit (low noise & high well capacity)' : 'Mono16'
    }

    _noise_properties = {
        'VSC-00954': {
            '12-bit (low noise)': {
                'ReadNoise' : 1.1,
                'ElectronsPerCount' : 0.28,
                'ADOffset' : 100, # check mean (or median) offset
                'SaturationThreshold' : 2**11-1#(2**16 -1) # check this is really 11 bit
            },
            '12-bit (high well capacity)': {
                'ReadNoise' : 5.96,
                'ElectronsPerCount' : 6.97,
                'ADOffset' : 100,
                'SaturationThreshold' : 2**11-1#(2**16 -1)         
            },
            '16-bit (low noise & high well capacity)': {
                'ReadNoise' : 1.33,
                'ElectronsPerCount' : 0.5,
                'ADOffset' : 100,
                'SaturationThreshold' : (2**16 -1)
            }},
        'CSC-00425': { # this is info for a Sona
            u'12-bit (low noise)': {
                'ReadNoise' : 1.21,
                'ElectronsPerCount' : 0.45,
                'ADOffset' : 100, # check mean (or median) offset
                'SaturationThreshold' : 1776  #(2**16 -1) # check this is really 11 bit
            },
            u'16-bit (high dynamic range)': {
                'ReadNoise' : 1.84,
                'ElectronsPerCount' : 1.08,
                'ADOffset' : 100,
                'SaturationThreshold' : 44185
            }},
        'VSC-02858': {
             '12-bit (low noise)': {
                'ReadNoise' : 1.19,
                'ElectronsPerCount' : 0.3,
                'ADOffset' : 100, # check mean (or median) offset
                'SaturationThreshold' : 2**11-1#(2**16 -1) # check this is really 11 bit
            },
            '12-bit (high well capacity)': {
                'ReadNoise' : 6.18,
                'ElectronsPerCount' : 7.2,
                'ADOffset' : 100,
                'SaturationThreshold' : 2**11-1#(2**16 -1)         
            },
            '16-bit (low noise & high well capacity)': {
                'ReadNoise' : 1.42,
                'ElectronsPerCount' : 0.5,
                'ADOffset' : 100,
                'SaturationThreshold' : (2**16 -1)
            }},
        'VSC-02698': {
             '12-bit (low noise)': {
                'ReadNoise' : 1.16,
                'ElectronsPerCount' : 0.26,
                'ADOffset' : 100, # check mean (or median) offset
                'SaturationThreshold' : 2**11-1#(2**16 -1) # check this is really 11 bit
            },
            '12-bit (high well capacity)': {
                'ReadNoise' : 6.64,
                'ElectronsPerCount' : 7.38,
                'ADOffset' : 100,
                'SaturationThreshold' : 2**11-1#(2**16 -1)         
            },
            '16-bit (low noise & high well capacity)': {
                'ReadNoise' : 1.36,
                'ElectronsPerCount' : 0.49,
                'ADOffset' : 100,
                'SaturationThreshold' : (2**16 -1)
            }}}

    # this class is compatible with the ATEnum object properties that are used in ZylaControlPanel
    # we use it as a higher level alternative to setting gainmode and encoding directly
    class SimpleGainEnum(object):
        def __init__(self, cam):
            self.cam = cam
            self.gainmodes = cam.SimplePreAmpGainControl.getAvailableValues()
            self.propertyName = 'SimpleGainMode'
            
        def getAvailableValues(self):
            return self.gainmodes

        def setString(self,str):
            self.cam.SetSimpleGainMode(str)

        def getString(self):
            return self.cam.GetSimpleGainMode()


    
    @property
    def noise_properties(self):
        """return the noise properties for a the given camera

        TODO: make this look in config, rather than storing noise properties here
        """
        try:
            return self._noise_properties[self.GetSerialNumber()][self.GetSimpleGainMode()]
        except KeyError:
            logger.warn('camera specific noise props not found - using default noise props')
            return {'ReadNoise' : 1.1,
                    'ElectronsPerCount' : 0.28,
                    'ADOffset' : 100, # check mean (or median) offset
                    'SaturationThreshold' : 2**11-1#(2**16 -1) # check this is really 11 bit,
                    }


    def __init__(self, camNum):
        #define properties

        self.CameraAcquiring = ATBool()
        self.SensorCooling = ATBool()
        
        self.AcquisitionStart = ATCommand()
        self.AcquisitionStop = ATCommand()
        
        self.SoftwareTrigger = ATCommand()
        
        self.CycleMode = ATEnum()
        self.ElectronicShutteringMode = ATEnum()
        self.FanSpeed = ATEnum()
        self.PreAmpGainChannel = ATEnum()
        self.PixelEncoding = ATEnum()
        self.PixelReadoutRate = ATEnum()
        self.PreAmpGain = ATEnum()
        self.PreAmpGainSelector = ATEnum()
        self.TriggerMode = ATEnum()
        self.Overlap = ATBool()
        self.RollingShutterGlobalClear = ATBool()
        
        self.AOIHeight = ATInt()
        self.AOILeft = ATInt()
        self.AOITop = ATInt()
        self.AOIWidth = ATInt()
        self.AOIStride = ATInt()
        self.FrameCount = ATInt()
        self.ImageSizeBytes = ATInt()
        self.SensorHeight = ATInt()
        self.SensorWidth = ATInt()
        
        self.Baseline = ATInt()
        
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
        self.defBuffers = 100

        self._n_timeouts = 0
       
        
        #self.contMode = True
        self.burstMode = False
        
        self._temp = 0
        self._frameRate = 0
        self._frame_wait_time = 100

        self.hardware_overflowed=False
        
        self.active = False
        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)
        
    def Init(self):
        SDK3Camera.Init(self)        
        
        #set some intial parameters
        #self.setNoisePropertiesByCam(self.GetSerialNumber())

        # figure out preamp gain modes for this camera type
        if not self.CameraModel.getValue().startswith('SIM'):
            self.PixelEncodingForGain = {}
            for mode in self.SimplePreAmpGainControl.getAvailableValues():
                if mode.startswith('12'):
                    self.PixelEncodingForGain[mode] = 'Mono12'
                elif mode.startswith('16'):
                    self.PixelEncodingForGain[mode] = 'Mono16'
                else:
                    raise RuntimeError('PixelEncodingForGain mode "%s" unknown bit depth (neither 12 nor 16 bit)' % (mode))
        else:
            self.PixelEncodingForGain = self.DefaultPixelEncodingForGain

        # this instance is compatible with use in Zylacontrolpanel
        # note we make this only once the camera has been initialised and PixelEncodingForGain been made
        self.SimpleGainEnumInstance = self.SimpleGainEnum(self)


        self.FrameCount.setValue(1)
        self.CycleMode.setString(u'Continuous')

        #need this to get full frame rate
        try:
            self.Overlap.setValue(True)
        except:
            logger.info("error setting overlap mode")
            pass

        # we use a try block as this will allow us to use the SDK software cams for simple testing
        try:
            self.SetSimpleGainMode('12-bit (low noise)')
        except:
            logger.info("error setting gain mode")
            pass

        # spurious noise filter off by default
        try:
            self.SpuriousNoiseFilter.setValue(0) # this will also fail with the SimCams
        except:
            logger.info("error disabling spurios noise filter")
            pass

        # Static Blemish Correction off by default
        try:
            self.StaticBlemishCorrection.setValue(0) # this will also fail with the SimCams
        except:
            logger.info("error disabling Static Blemish Correction")
            pass
        
        
        self.TriggerMode.setString('Internal')
        
        try:
            self.SensorCooling.setValue(True)
        except:
            logger.info("error setting cooling mode")
            pass

        try:
            TCModes = self.TemperatureControl.getAvailableValues()
            self.TemperatureControl.setString(TCModes[0])
        except:
            pass

        #self.PixelReadoutRate.setIndex(1)
        # test if we have only fixed ROIs
        self._fixed_ROIs = not self.FullAOIControl.isImplemented() or not self.FullAOIControl.getValue()
        #self.noiseProps = self.baseNoiseProps[self.GetSimpleGainMode()]

        self.SetIntegTime(.100)
        
        if not self._fixed_ROIs:
            self.SetROI(0,0, self.GetCCDWidth(), self.GetCCDHeight())
        #set up polling thread        
        self.doPoll = False
        self.pollLoopActive = True
        self.pollThread = threading.Thread(target = self._pollLoop)
        self.pollThread.start()

        #self.active = True
        
        
    #Neo buffer helper functions    
        
    def InitBuffers(self):
        self._flush()
        bufSize = self.ImageSizeBytes.getValue()
        vRed = int(self.SensorHeight.getValue()/self.AOIHeight.getValue())
        self.nBuffers = min(vRed*self.defBuffers, 5000)
        
        if not self.contMode:
            self.nBuffers = 5
        #print bufSize
        for i in range(self.nBuffers):
            #buf = np.empty(bufSize, 'uint8')
            buf = create_aligned_array(bufSize, 'uint8')
            self._queueBuffer(buf)

        #wait for the buffers to be queued - stops us from calling startAcquisition prematurely
        while not self.buffersToQueue.empty():
            time.sleep(0.01)

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
        n_queued = 0 #number queued in this pass. Restrict this so that we don't spend all our time queuing buffers
        while (n_queued < 30) and (not self.buffersToQueue.empty()):
            try:
                buf = self.buffersToQueue.get(block=False)
                try:
                    #print np.base_repr(buf.ctypes.data, 16)
                    SDK3.QueueBuffer(self.handle, buf.ctypes.data_as(SDK3.POINTER(SDK3.AT_U8)), buf.nbytes)
                    self.queuedBuffers.put(buf)
                except SDK3.CameraError as e:
                    traceback.print_exc()
                    if not SDK3.errorCodes[e.errNo] == 'AT_ERR_INVALIDSIZE':
                        raise
                #self.fLog.write('%f\tq\n' % time.time())
                self.nQueued += 1
                n_queued += 1
            except Queue.Empty:
                logger.exception('Buffer Queue Empty')
                pass
        #self.camLock.release()
        
    def _pollBuffer(self):
        try:
            #self.fLog.write('%f\tp\n' % time.time())
            pData, lData = SDK3.WaitBuffer(self.handle, self._frame_wait_time)
            #self.fLog.write('%f\tb\n' % time.time())
        except SDK3.TimeoutError as e:
            #Both AT_ERR_TIMEDOUT and AT_ERR_NODATA
            #get caught as TimeoutErrors
            #if e.errNo == SDK3.AT_ERR_TIMEDOUT:
            #    self.fLog.write('%f\tt\n' % time.time())
            #else:
            #    self.fLog.write('%f\tn\n' % time.time())
            if e.errNo == SDK3.AT_ERR_NODATA:
                logger.debug('AT_ERR_NODATA')
                # if we had no data, wait a little before trying again
                time.sleep(0.1)
            elif e.errNo == SDK3.AT_ERR_TIMEDOUT:
                pass
                self._n_timeouts += 1
                if self._n_timeouts > 10:
                    self.hardware_overflowed = True
                    logger.debug('AT_ERR_TIMEDOUT (_n_timeouts = %d)' % self._n_timeouts)

            return
        except SDK3.CameraError as e:
            if not e.errNo == SDK3.AT_ERR_NODATA:
                traceback.print_exc()

                if SDK3.errorCodes[e.errNo] == 'AT_ERR_HARDWARE_OVERFLOW':
                    self.hardware_overflowed = True
            return
        except:
            #catch and ignore / print any other errors (e.g. WindowsError)
            traceback.print_exc()
            return
            
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
            if self.doPoll and (self.nQueued > 0): #only poll if an acquisition is running
                self._pollBuffer()
            else:
                #print 'w',
                time.sleep(.01)
            time.sleep(.0001)
            #self.fLog.flush()
        #self.fLog.close()
        
    #PYME Camera interface functions - make this look like the other cameras
    def ExpReady(self):
        #self._pollBuffer()
        
        return not self.fullBuffers.empty()
        
    def ExtractColor(self, chSlice, mode):
        #grab our buffer from the full buffers list
        buf = self.fullBuffers.get(timeout=.1)
        self.nFull -= 1
        
        #copy to the current 'active frame' 
        #print chSlice.shape, buf.view(chSlice.dtype).shape
        #bv = buf.view(chSlice.dtype).reshape(chSlice.shape)
        xs, ys = chSlice.shape[:2]
        
        a_s = self.AOIStride.getValue()
        
        #print buf.nbytes
        #bv = buf.view(chSlice.dtype).reshape([-1, ys], order='F')
        
#        bv = np.ndarray(shape=[xs,ys], dtype='uint16', strides=[2, a_s], buffer=buf)
#        chSlice[:] = bv
        
        #chSlice[:,:] = bv
        #ctypes.cdll.msvcrt.memcpy(chSlice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), chSlice.nbytes)
        #ctypes.cdll.msvcrt.memcpy(chSlice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), chSlice.nbytes)
        #print 'f'
        
        dt = self.PixelEncoding.getString()
        
        SDK3.ConvertBuffer(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), chSlice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), xs, ys, a_s, dt, 'Mono16')
        
        #recycle buffer
        if self.doPoll:
            self._queueBuffer(buf)
        
#    def SetContinuousMode(self, value=True):
#        if value:
#            self.CycleMode.setString(u'Continuous')
#            self.contMode = True
#        else:
#            self.CycleMode.setString(u'Fixed')
#            self.contMode = False
#            
#    def GetContinuousMode(self):
#        return self.contMode
        
    def SetAcquisitionMode(self, mode):
        if mode in [self.MODE_CONTINUOUS, self.MODE_SOFTWARE_TRIGGER]:
            if not self.contMode:
                self.CycleMode.setString(u'Continuous')
            
            if mode == self.MODE_SOFTWARE_TRIGGER:
                print('Setting software triggered mode')
                self.TriggerMode.setString(u'Software')
            else:
                self.TriggerMode.setString(u'Internal')
                
        elif self.contMode:
            self.CycleMode.setString(u'Fixed')
            self.FrameCount.setValue(1)
            
    def GetAcquisitionMode(self):
        if self.contMode:
            if self.TriggerMode.getString() == u'Software':
                return self.MODE_SOFTWARE_TRIGGER
            else:
                return self.MODE_CONTINUOUS
        else:
            return self.MODE_SINGLE_SHOT
    
    def FireSoftwareTrigger(self):
        self.SoftwareTrigger()
    
    @property
    def contMode(self):
        return self.CycleMode.getString() == u'Continuous'
        
    def GetSerialNumber(self):
        return self.SerialNumber.getValue()
    
    def SetIntegTime(self, iTime):
        #logger.debug('SetIntegTime')
        self.ExposureTime.setValue(max(iTime, self.ExposureTime.min()))
        self.FrameRate.setValue(self.FrameRate.max())
        #logger.debug('SetIntegTime : done')
        
    def GetIntegTime(self): 
        return self.ExposureTime.getValue()
        
    def GetCycleTime(self):
        return 1.0/self.FrameRate.getValue()
    
    def GetCCDWidth(self): 
        return self.SensorHeight.getValue()
    def GetCCDHeight(self): 
        return self.SensorWidth.getValue()
    
        
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
        
        self.AOIWidth.setValue(width)
        self.AOILeft.setValue(left)
        self.AOIHeight.setValue(height)
        self.AOITop.setValue(top)

    def ROIsAreFixed(self):
        return self._fixed_ROIs

    def SetROI(self, x1, y1, x2, y2):
        #support ROIs which have been dragged in any direction
        #TODO - this should really be in the GUI, not here
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        
        #have to set width before x, height before y
        self.AOIWidth.setValue(x2-x1)
        self.AOIHeight.setValue(y2 - y1)
        self.AOILeft.setValue(x1+1)
        self.AOITop.setValue(y1+1)

    def SetSimpleGainMode(self,mode):
        if not any(mode in s for s in self.PixelEncodingForGain.keys()):
            logger.warn('invalid mode "%s" requested - ignored' % mode)
            return

        self.SimplePreAmpGainControl.setString(mode)
        self.PixelEncoding.setString(self.PixelEncodingForGain[mode])

    def GetSimpleGainMode(self):
        return self.SimplePreAmpGainControl.getString()
    
    def GetROI(self):
        x1 = self.AOILeft.getValue() - 1
        y1 = self.AOITop.getValue() - 1
        
        x2 = x1 + self.AOIWidth.getValue()
        y2 = y1 + self.AOIHeight.getValue()
        
        return [x1, y1, x2, y2]

    # def GetROIX1(self):
    #     return self.AOILeft.getValue()
    #
    # def GetROIX2(self):
    #     return self.AOILeft.getValue() + self.AOIWidth.getValue()
    #
    # def GetROIY1(self):
    #     return self.AOITop.getValue()
    #
    # def GetROIY2(self):
    #     return self.AOITop.getValue() + self.AOIHeight.getValue()
    

    def Shutdown(self):
        logger.info('Shutting down sCMOS camera')
        self.pollLoopActive = False
        self.shutdown()
        #pass


    def StartExposure(self):
        #make sure no acquisiton is running
        self.StopAq()
        self._temp = self.SensorTemperature.getValue()
        self._frameRate = self.FrameRate.getValue()
        self.tKin = 1.0 / self._frameRate

        self._frame_wait_time = ctypes.c_uint(int(max(2*1000*self.tKin, 100)))

        self.hardware_overflowed = False
        self._n_timeouts = 0
        #logger.debug('StartAq')
        eventLog.logEvent('StartAq', '')
        self._flush()
        self.InitBuffers()
        self.AcquisitionStart()

        return 0
        
    def StopAq(self):
        #logger.debug('StopAq')
        self.doPoll = False
        if self.CameraAcquiring.getValue():
            self.AcquisitionStop()

        #self._flush() #TODO - should we be calling this?

        #flush at least the buffers to queue (so that we don't queue buffers of the wrong size)
        while not self.buffersToQueue.empty():
            self.buffersToQueue.get()

        #logger.debug('StopAq : done')
        
    def GetNumImsBuffered(self):
        return self.nFull
    
    def GetBufferSize(self):
        return self.nBuffers

    def GenStartMetadata(self, mdh):
        if self.active:
            self.GetStatus()
    
            mdh.setEntry('Camera.Name', 'Andor sCMOS')
            mdh.setEntry('Camera.Model', self.CameraModel.getValue())
            mdh.setEntry('Camera.SerialNumber', self.GetSerialNumber())

            mdh.setEntry('Camera.SensorWidth',self.GetCCDWidth())
            mdh.setEntry('Camera.SensorHeight',self.GetCCDHeight())

            mdh.setEntry('Camera.IntegrationTime', self.GetIntegTime())
            mdh.setEntry('Camera.CycleTime', self.GetCycleTime())
            mdh.setEntry('Camera.EMGain', 1)
            mdh.setEntry('Camera.DefaultEMGain', 1) # needed for some protocols
            mdh.setEntry('Camera.SimpleGainMode', self.GetSimpleGainMode())

            #mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
            #mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())

            x1, y1, x2, y2 = self.GetROI()
            mdh.setEntry('Camera.ROIOriginX', x1)
            mdh.setEntry('Camera.ROIOriginY', y1)
            mdh.setEntry('Camera.ROIWidth', x2 - x1)
            mdh.setEntry('Camera.ROIHeight', y2 - y1)

            #mdh.setEntry('Camera.StartCCDTemp',  self.GetCCDTemp())
            # these make sense with the Sona choices but should also be ok for the Zyla
            mdh.setEntry('Camera.TemperatureControl', self.TemperatureControl.getString())
            mdh.setEntry('Camera.TemperatureStatus', self.TemperatureStatus.getString())
            mdh.setEntry('Camera.SensorTemperature', self.SensorTemperature.getValue())
            
            # pick up noise settings for gain mode
            np = self.noise_properties
            mdh.setEntry('Camera.ReadNoise', np['ReadNoise'])
            mdh.setEntry('Camera.NoiseFactor', 1.0)
            mdh.setEntry('Camera.ElectronsPerCount', np['ElectronsPerCount'])

            if (self.Baseline.isImplemented()):
                mdh.setEntry('Camera.ADOffset', self.Baseline.getValue())
            else:
                mdh.setEntry('Camera.ADOffset', np['ADOffset'])


            #mdh.setEntry('Simulation.Fluorophores', self.fluors.fl)
            #mdh.setEntry('Simulation.LaserPowers', self.laserPowers)
    
            #realEMGain = ccdCalibrator.CalibratedCCDGain(self.GetEMGain(), self.GetCCDTempSetPoint())
            #if not realEMGain == None:
            mdh.setEntry('Camera.TrueEMGain', 1)
            
            self.fill_camera_map_metadata(mdh)
            

            if  self.StaticBlemishCorrection.isImplemented():
                mdh.setEntry('Camera.StaticBlemishCorrection', self.StaticBlemishCorrection.getValue())
            if  self.SpuriousNoiseFilter.isImplemented():
                mdh.setEntry('Camera.SpuriousNoiseFilter', self.SpuriousNoiseFilter.getValue())


    #functions to make us look more like EMCCD camera
    def GetEMGain(self):
        return 1

    def GetCCDTempSetPoint(self):
        return self.TargetSensorTemperature.getValue()

    def SetCCDTemp(self, temp):
        self.TargetSensorTemperature.setValue(temp)
        #pass

    def SetEMGain(self, gain):
        logger.info("EMGain ignored")


    def SetBurst(self, burstSize):
        if burstSize > 1:
            self.SetAcquisitionMode(self.MODE_SINGLE_SHOT)
            self.FrameCount.setValue(burstSize)
            #self.contMode = True
            self.burstMode = True
        else:
            self.FrameCount.setValue(1)
            self.SetAcquisitionMode(self.MODE_CONTINUOUS)
            self.burstMode = False

    
    def GetFPS(self):
        #return self.FrameRate.getValue()
        return self._frameRate

    def TemperatureStatusText(self):
        return "Zyla target T %s - %s" % (self.TemperatureControl.getString(),
                                          self.TemperatureStatus.getString())


    def __del__(self):
        self.Shutdown()
        #self.compT.kill = True

        
        
        
        
class AndorZyla(AndorBase):              
    def __init__(self, camNum):
        #define properties
        self.Overlap = ATBool()
        self.SpuriousNoiseFilter = ATBool()
        self.StaticBlemishCorrection = ATBool()
        
        self.VerticallyCentreAOI = ATBool()
        
        self.CameraDump = ATCommand()
        self.SoftwareTrigger = ATCommand()
        
        self.TemperatureControl = ATEnum()
        self.TemperatureStatus = ATEnum()
        self.SimplePreAmpGainControl = ATEnum()

        self.BitDepth = ATEnum()
        
        self.ActualExposureTime = ATFloat()
        self.BurstRate = ATFloat()
        self.ReadoutTime = ATFloat()
        
        self.TimestampClock = ATInt()
        self.TimestampClockFrequency = ATInt()
        
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


class MultiviewZyla(MultiviewCameraMixin, AndorZyla):
    def __init__(self, camNum, multiview_info):
        AndorZyla.__init__(self, camNum)
        # default to the whole chip
        default_roi = dict(xi=0, xf=2048, yi=0, yf=2048)
        MultiviewCameraMixin.__init__(self, multiview_info, default_roi, AndorZyla)
