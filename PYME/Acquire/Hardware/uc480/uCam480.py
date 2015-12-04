#!/usr/bin/python

##################
# uCam480.py
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

from . import uc480
from ctypes import *
import ctypes
import time
import sys
from PYME.Acquire import MetaDataHandler

import os
from PYME.FileUtils import nameUtils

import threading

try:
    import Queue
except ImportError:
    import queue as Queue

import numpy as np

from PYME.Acquire import eventLog

def init(cameratype='uc480'):
    uc480.init(cameratype)

def GetError(camHandle):
    err = ctypes.c_int()
    errMessage = ctypes.c_char_p()
    uc480.CALL("GetError",camHandle,ctypes.byref(err),ctypes.byref(errMessage))
    
    return err.value, errMessage.value
    
def GetNumCameras():
    numCams = ctypes.c_int()
    uc480.CALL("GetNumberOfCameras", ctypes.byref(numCams))
    
    return numCams.value
    
def GetCameraList():
    nCams = GetNumCameras()
    
    class UEYE_CAMERA_LIST(ctypes.Structure):
        _fields_ = [("dwCount", ctypes.wintypes.ULONG ),] + [("uci%d" %n, uc480.UEYE_CAMERA_INFO) for n in range(nCams)] #
    
    camlist = UEYE_CAMERA_LIST()
    camlist.dwCount = nCams
    
    uc480.CALL("GetCameraList", ctypes.byref(camlist))
    
    return camlist
    

class uc480Camera:
    numpy_frames=1
    contMode = True

    #define a couple of acquisition modes

    MODE_CONTINUOUS = 5
    MODE_SINGLE_SHOT = 1

    ROIlimitlist = {
        'UI306X' : {
            'xmin' : 96,
            'xstep' : 8,
            'ymin' : 32,
            'ystep' : 2
        },
        'UI324X' : {
            'xmin' : 16,
            'xstep' : 4,
            'ymin' : 4,
            'ystep' : 2
        }        
    }

    ROIlimitsDefault = ROIlimitlist['UI324X']

    def __init__(self, boardNum=0, nbits = 8):
        self.initialised = False
        self.active = True
        if nbits not in [8,10,12]:
            raise RuntimeError('Supporting only 8, 10 or 12 bit depth, requested %d bit' % (nbits))
        self.nbits = nbits

        self.boardHandle = wintypes.HANDLE(boardNum)

        ret = uc480.CALL('InitCamera', byref(self.boardHandle), wintypes.HWND(0))
        print(('I',ret))
        if not ret == 0:
            raise RuntimeError('Error getting camera handle: %d: %s' % GetError(self.boardHandle))
            
        self.initialised = True

        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)

        caminfo = uc480.CAMINFO()
        ret = uc480.CALL('GetCameraInfo', self.boardHandle, ctypes.byref(caminfo))
        if not ret == 0:
            raise RuntimeError('Error getting camera info: %d: %s' % GetError(self.boardHandle))
        
        self.serialNum = caminfo.SerNo

        #get the CCD size 
        sensorProps = uc480.SENSORINFO()
        
        ret = uc480.CALL('GetSensorInfo', self.boardHandle, ctypes.byref(sensorProps))
        if not ret == 0:
            raise RuntimeError('Error getting CCD size: %d: %s' % GetError(self.boardHandle))

        self.CCDSize=(sensorProps.nMaxWidth, sensorProps.nMaxHeight)
        senstype = ctypes.cast(sensorProps.strSensorName, ctypes.c_char_p)
        self.sensortype = senstype.value

        # work out the ROI limits for this sensortype
        matches = [self.ROIlimitlist[st] for st in self.ROIlimitlist.keys()
                   if self.sensortype.startswith(st)]
        if len(matches) > 0:
            self.ROIlimits = matches[0]
        else:
            self.ROIlimits = self.ROIlimitsDefault

        #-------------------
        #Do initial setup with a whole bunch of settings I've arbitrarily decided are
        #reasonable and make the camera behave more or less like the PCO cameras.
        #These settings will hopefully be able to be changed with methods/ read from
        #a file later.
        
        dEnable = c_double(0); # don't enable
        uc480.CALL('SetAutoParameter', self.boardHandle, uc480.IS_SET_ENABLE_AUTO_GAIN, byref(dEnable), 0)
        uc480.CALL('SetAutoParameter', self.boardHandle, uc480.IS_SET_ENABLE_AUTO_SHUTTER, byref(dEnable), 0)
        #uc480.CALL('SetAutoParameter', self.boardHandle, uc480.IS_SET_ENABLE_AUTO_SENOR_GAIN, byref(dEnable), 0)
        
        # may need to revisit the gain - currently using 10 which translates into different gains depending on chip (I believe)
        # had to reduce from 100 to avoid saturating camera at lowish light levels
        ret = uc480.CALL('SetGainBoost', self.boardHandle, uc480.IS_SET_GAINBOOST_OFF)
        self.errcheck(ret,'SetGainBoost',fatal=False)
        ret = uc480.CALL('SetHardwareGain', self.boardHandle, 10, uc480.IS_IGNORE_PARAMETER, uc480.IS_IGNORE_PARAMETER, uc480.IS_IGNORE_PARAMETER)
        self.errcheck(ret,'SetHardwareGain',fatal=False)

        uc480.CALL('SetImageSize', self.boardHandle, self.CCDSize[0],self.CCDSize[1] )
        
        # CS - BITS: memory depth in here - the color mode may require something in accordance with bit depth!
        # CS: 8bit  constant: IS_SET_CM_BAYER is supposed to be replaced by IS_CM_BAYER_RG8
        # CS: 8bit monochrome: IS_CM_SENSOR_RAW8 should replace IS_CM_BAYER_RG8
        # CS: 12bit constant: IS_CM_SENSOR_RAW12
        # CS: info from SetColorMode and "Color and memory formats" appendix
        # note: the monochrome constants may be preferable
        if self.nbits == 8:
            colormode = uc480.IS_CM_MONO8
        elif self.nbits == 10:
            colormode = uc480.IS_CM_MONO10
        elif self.nbits == 12:
            colormode = uc480.IS_CM_MONO12
        ret = uc480.CALL('SetColorMode', self.boardHandle, colormode)
        self.errcheck(ret,'setting ColorMode')

        uc480.CALL('SetBinning', self.boardHandle, uc480.IS_BINNING_DISABLE)
        self.binning=False #binning flag - binning is off
        self.binX=1 #1x1
        self.binY=1
        
        self.background = None
        self.flatfield = None
        self.flat = None
        
        #load flatfield (if present)
        calpath = nameUtils.getCalibrationDir(self.serialNum)
        ffname = os.path.join(calpath, 'flatfield.npy')
        if os.path.exists(ffname):
            self.flatfield = np.load(ffname).squeeze()
            self.flat = self.flatfield
        
        self.SetROI(0,0, self.CCDSize[0],self.CCDSize[1])
        #self.ROIx=(1,self.CCDSize[0])
        #self.ROIy=(1,self.CCDSize[1])
        
        self._buffers = []
        
        self.fullBuffers = Queue.Queue()
        self.freeBuffers = None
        
        self.nFull = 0
        
        self.nAccum = 1
        self.nAccumCurrent = 0
        
        self.Init()
        
    def errcheck(self,value,msg,fatal=True):
        if not value == uc480.IS_SUCCESS:
            if fatal:
                raise RuntimeError('Error %s: %d: %s' % [msg]+GetError(self.boardHandle))
            else:
                print 'Error %s: %d: %s' % [msg]+GetError(self.boardHandle)

    def Init(self):        
        #set up polling thread        
        self.doPoll = False
        self.pollLoopActive = True
        self.pollThread = threading.Thread(target = self._pollLoop)
        self.pollThread.start()
        
    def InitBuffers(self, nBuffers = 50, nAccumBuffers = 50):
        for i in range(nBuffers):
            pData = POINTER(c_char)()
            bufID = c_int(0)
            # CS - BITS: memory depth in here
            if self.nbits == 8:
                bitsperpix = 8
            else: # 10 & 12 bits
                bitsperpix = 16

            ret = uc480.CALL('AllocImageMem', self.boardHandle, self.GetPicWidth(), self.GetPicHeight(), bitsperpix, ctypes.byref(pData), ctypes.byref(bufID))
            if not ret == uc480.IS_SUCCESS:
                raise RuntimeError('Error allocating memory: %d: %s' % GetError(self.boardHandle))
            #ret = uc480.CALL('SetImageMem', self.boardHandle, pData, bufID)
            #if not ret == uc480.IS_SUCCESS:
            #    raise RuntimeError('Error setting memory: %d: %s' % GetError(self.boardHandle))
            ret = uc480.CALL('AddToSequence', self.boardHandle, pData, bufID)
            if not ret == uc480.IS_SUCCESS:
                raise RuntimeError('Error adding to sequence: %d: %s' % GetError(self.boardHandle))
            
            self._buffers.append((bufID, pData))
            
        ret = uc480.CALL('InitImageQueue', self.boardHandle, 0)
        if not ret == uc480.IS_SUCCESS:
                raise RuntimeError('error initialising queue: %d: %s' % GetError(self.boardHandle))
                
        # CS - BITS: memory depth in here
        if self.nbits == 8:
            bufferdtype = np.uint8
        else: # 10 & 12 bits
            bufferdtype = np.uint16
        self.transferBuffer = np.zeros([self.GetPicHeight(), self.GetPicWidth()], bufferdtype)
        
        self.freeBuffers = Queue.Queue()
        # CS - BITS: memory depth (potentially) in here
        # CS: we leave this as uint16 regardless of 8 or 12 bits for now as accumulation
        #     of the underlying 12 bit data should be ok (but maybe not?)
        for i in range(nAccumBuffers):
            self.freeBuffers.put(np.zeros([self.GetPicHeight(), self.GetPicWidth()], np.uint16))
        self.accumBuffer = self.freeBuffers.get()
        self.nAccumCurrent = 0
        self.doPoll = True
        
    def DestroyBuffers(self):
        self.doPoll = False
        uc480.CALL('ExitImageQueue', self.boardHandle)
        uc480.CALL('ClearSequence', self.boardHandle)
        
        while len(self._buffers) > 0:
            bID, pData = self._buffers.pop()
            uc480.CALL('FreeImageMem', self.boardHandle, pData, bID)
            
        self.freeBuffers = None
            
    
    def _pollBuffer(self):
        pData = POINTER(c_char)()
        bufID = c_int(0)
        
        ret = uc480.CALL('WaitForNextImage', self.boardHandle, 1000, byref(pData), byref(bufID))
        
        if not ret == uc480.IS_SUCCESS:
            print 'Wait for image failed with:', ret
            return
            
        ret = uc480.CALL('CopyImageMem', self.boardHandle, pData, bufID, self.transferBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        if not ret == uc480.IS_SUCCESS:
            print 'CopyImageMem failed with:', ret
            return
        
        #chSlice[:] = self.transferBuffer[:].T #.reshape(chSlice.shape)
        if self.nAccumCurrent == 0:
            self.accumBuffer[:] = self.transferBuffer
        else:
            self.accumBuffer[:] = self.accumBuffer + self.transferBuffer
        self.nAccumCurrent += 1
        
        ret = uc480.CALL('UnlockSeqBuf', self.boardHandle, uc480.IS_IGNORE_PARAMETER, pData)
        
        if self.nAccumCurrent >= self.nAccum:    
            self.fullBuffers.put(self.accumBuffer)
            self.accumBuffer = self.freeBuffers.get()
            self.nAccumCurrent = 0
            self.nFull += 1
        #self.camLock.release()
        
    def _pollLoop(self):
        #self.fLog = open('poll.txt', 'w')
        while self.pollLoopActive:
            #self._queueBuffers()
            if self.doPoll: #only poll if an acquisition is running
                self._pollBuffer()
            else:
                #print 'w',
                time.sleep(.05)
            time.sleep(.0005)

    def GetCamType(*args):
        raise Exception('Not implemented yet!!')

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
        
    def SetAcquisitionMode(self, mode):
        if mode == self.MODE_SINGLE_SHOT:
            self.contMode = False
        else:
            self.contMode = True

    def SetIntegTime(self, iTime):
        #self.__selectCamera()
        newExp = c_double(0)
        newFrameRate = c_double(0)
        ret = uc480.CALL('SetFrameRate', self.boardHandle, c_double(1.0e3/iTime), byref(newFrameRate))
        if not ret == 0:
            raise RuntimeError('Error setting exp time: %d: %s' % GetError(self.boardHandle))
        
        ret = uc480.CALL('SetExposureTime', self.boardHandle, c_double(iTime), ctypes.byref(newExp))
        if not ret == 0:
            raise RuntimeError('Error setting exp time: %d: %s' % GetError(self.boardHandle))
            
        self.expTime = newExp.value

    def GetIntegTime(self):
        return self.expTime

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
        return self.CCDSize[0]

    def GetCCDHeight(self):
        return self.CCDSize[1]

    def SetHorizBin(self, val):
        raise Exception('Not implemented yet!!')

    def GetHorizBin(self):
        return self.binning
        #raise Exception, 'Not implemented yet!!'

    def GetHorzBinValue(*args):
        #raise Exception, 'Not implemented yet!!'
        return self.binX

    def SetVertBin(self, val):
        raise Exception('Not implemented yet!!')

    def GetVertBin(*args):
        return 0
        #raise Exception, 'Not implemented yet!!'

    def GetVertBinValue(*args):
        #raise Exception, 'Not implemented yet!!'
        return self.binY

    def GetNumberChannels(*args):
        raise Exception('Not implemented yet!!')

    def GetElectrTemp(*args):
        return 25

    def GetCCDTemp(self):
        return  25 #self.CCDTemp

    def CamReady(self):
        return self.initialised

    def GetPicWidth(self):
        return (self.ROIx[1] - self.ROIx[0] + 1)/self.binX

    def GetPicHeight(self):
        return (self.ROIy[1] - self.ROIy[0] + 1)/self.binY


    def SetROI(self, x1,y1,x2,y2):
        #must be on xstep x ystep pixel grid

        xstep = self.ROIlimits['xstep']
        ystep = self.ROIlimits['ystep']
        x1 = xstep*(x1/xstep)
        x2 = xstep*(x2/xstep)
        y1 = ystep*(y1/ystep)
        y2 = ystep*(y2/ystep)
        
        
        #if coordinates are reversed, don't care
        if (x2 > x1):
            self.ROIx = (x1+1, x2)
        elif (x2 < x1):
            self.ROIx = (x2+1, x1)
        else: #x1 == x2 - BAD
            raise RuntimeError('Error Setting x ROI - Zero sized ROI')

        if (y2 > y1):
            self.ROIy = (y1+1, y2)

        elif (y2 < y1):
            self.ROIy = (y2+1, y1)

        else: #y1 == y2 - BAD
            raise RuntimeError('Error Setting y ROI - Zero sized ROI')
        
        rect = uc480.IS_RECT()
        rect.s32X =  self.ROIx[0] - 1
        rect.s32Y =  self.ROIy[0] - 1
        rect.s32Width  = max(1 + self.ROIx[1] - self.ROIx[0], self.ROIlimits['xmin']) # ensure minimim size in x
        rect.s32Height = max(1 + self.ROIy[1] - self.ROIy[0], self.ROIlimits['ymin']) # ensure minimim size in y

        print((rect.s32X, rect.s32Width))
        
        #ret = uc480.CALL('SetImageSize', self.boardHandle, rect.s32Width, rect.s32Height )
        #if not ret == 0:
        #    raise RuntimeError('Error setting image size: %d: %s' % GetError(self.boardHandle))
        
        ret = uc480.CALL('AOI', self.boardHandle, uc480.IS_AOI_IMAGE_SET_AOI, byref(rect), ctypes.sizeof(rect))
        if not ret == 0:
            raise RuntimeError('Error setting ROI: %d: %s' % GetError(self.boardHandle))
            
        if not self.flatfield == None:
            self.flat = self.flatfield[x1:x2, y1:y2]

        #raise Exception, 'Not implemented yet!!'

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

    def GetStatus(*args):
        pass

    def SetCOC(*args):
        pass

    def StartExposure(self):
        
        if self.doPoll:
            print('StartAq')
            self.StopAq()
        self.InitBuffers()
        
        

        eventLog.logEvent('StartAq', '')
        if self.contMode:
            ret = uc480.CALL('CaptureVideo', self.boardHandle, uc480.IS_DONT_WAIT)
        else:
            ret = uc480.CALL('FreezeVideo', self.boardHandle, uc480.IS_DONT_WAIT)
        if not ret == 0:
            raise RuntimeError('Error starting exposure: %d: %s' % GetError(self.boardHandle))
        return 0

    def StartLifePreview(*args):
        raise Exception('Not implemented yet!!')

    def StopLifePreview(*args):
        raise Exception('Not implemented yet!!')

    #PYME Camera interface functions - make this look like the other cameras
    def ExpReady(self):
        #self._pollBuffer()
        
        return not self.fullBuffers.empty()
        
    def ExtractColor(self, chSlice, mode):
        #grab our buffer from the full buffers list
        #pData, bufID = self.fullBuffers.get()
        buf = self.fullBuffers.get()
        #print pData, bufID
        self.nFull -= 1
        
        #ctypes.cdll.msvcrt.memcpy(chSlice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), chSlice.nbytes)
        
        #ret = uc480.CALL('CopyImageMem', self.boardHandle, pData, bufID, chSlice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        #ret = uc480.CALL('CopyImageMem', self.boardHandle, pData, bufID, self.transferBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        
        #chSlice[:] = self.transferBuffer[:].T #.reshape(chSlice.shape)
        chSlice[:] = buf.T
        
        if (not self.background == None) and self.background.shape == chSlice.shape:
            chSlice[:] = (chSlice - np.minimum(chSlice, self.background))[:]
            
        if (not self.flat == None) and self.flat.shape == chSlice.shape:
            chSlice[:] = (chSlice*self.flat).astype('uint16')[:]
        
        #ret = uc480.CALL('UnlockSeqBuf', self.boardHandle, uc480.IS_IGNORE_PARAMETER, pData)

        #recycle buffer
        self.freeBuffers.put(buf)
        
    def CheckCoordinates(*args):
        raise Exception('Not implemented yet!!')

    def StopAq(self):
        ret = uc480.CALL('StopLiveVideo', self.boardHandle, uc480.IS_WAIT)
        self.DestroyBuffers()

    def SetCCDTemp(self, temp):
        pass

    def SetEMGain(self, gain):
        raise Exception('Not implemented yet!!')

    def GetEMGain(self):
        return self.EMGain
        
    def SetGainBoost(self, on):
        if on:
            uc480.CALL('SetGainBoost', self.boardHandle, uc480.IS_SET_GAINBOOST_ON)
        else:
            uc480.CALL('SetGainBoost', self.boardHandle, uc480.IS_SET_GAINBOOST_OFF)
            
    def SetGain(self, gain=100):
        uc480.CALL('SetHardwareGain', self.boardHandle, gain, uc480.IS_IGNORE_PARAMETER, uc480.IS_IGNORE_PARAMETER, uc480.IS_IGNORE_PARAMETER)
        
    
    def SetAccumulation(self, nFrames):
        self.nAccum = nFrames
        
    def GetAccumulation(self):
        return self.nAccum

    def Shutdown(self):
        if self.doPoll: #acquisition is running
            self.StopAq()
            
        #shutdown polling thread
        self.pollLoopActive = False
        ret = uc480.CALL('ExitCamera', self.boardHandle)
        self.initialised = False
        
    def GetFPS(self):
        fps = c_double(0)        
        
        ret = uc480.CALL('GetFramesPerSecond', self.boardHandle, byref(fps))
        return fps.value
    
    def GetNumImsBuffered(self):
        return self.nFull

    def GetBufferSize(self):
        return len(self._buffers)

    def SetShutter(self, state):
        pass
        #raise Exception, 'Not implemented yet!!'
        
    def GetSerialNumber(self):
        return self.serialNum

    def GetHeadModel(self):
        return self.sensortype

    def SetActive(self, active=True):
        '''flag the camera as active (or inactive) to dictate whether it writes it's metadata or not'''
        self.active = active

    def GenStartMetadata(self, mdh):
        if self.active: #we are active -> write metadata
            self.GetStatus()

            mdh.setEntry('Camera.Name', 'UC480-UEYE')
            mdh.setEntry('Camera.Model', self.GetHeadModel())
            mdh.setEntry('Camera.SerialNumber', self.GetSerialNumber())

            mdh.setEntry('Camera.IntegrationTime', self.GetIntegTime())
            mdh.setEntry('Camera.CycleTime', 1./self.GetFPS())

            mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
            mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())
            mdh.setEntry('Camera.ROIWidth', self.GetROIX2() - self.GetROIX1())
            mdh.setEntry('Camera.ROIHeight',  self.GetROIY2() - self.GetROIY1())

    def __del__(self):
        if self.initialised:
            self.Shutdown()

