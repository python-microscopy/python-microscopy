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
from PYME.IO import MetaDataHandler
import logging
logger = logging.getLogger(__name__)

import os
from PYME.IO.FileUtils import nameUtils

import threading
import traceback

try:
    import Queue
except ImportError:
    import queue as Queue

import numpy as np

from PYME.Acquire import eventLog

def init(cameratype='uc480'):
    uc480.loadLibrary(cameratype)

def GetError(camHandle):
    err = ctypes.c_int()
    errMessage = ctypes.c_char_p()
    uc480.CALL("GetError",camHandle,ctypes.byref(err),ctypes.byref(errMessage))
    
    return err.value, errMessage.value
    
def GetNumCameras():
    numCams = ctypes.c_int()
    uc480.CALL("GetNumberOfCameras", ctypes.byref(numCams))
    
    return numCams.value
    
def as_string(cinfo,field):
    strval = getattr(cinfo,field)
    return (ctypes.cast(strval, ctypes.c_char_p)).value

def translateCaminfo(camlist):
    print(camlist, dir(camlist), camlist.dwCount)
    tlist = {}

    tlist['count'] = int(camlist.dwCount)
    tlist['_UEYE_CAMERA_LIST'] = camlist # raw ctype object for reference

    
    tlist['cameras'] = []
    for n in range(tlist['count']):
        uci = "uci%d" % n
        print(uci)
        caminfo = getattr(camlist,uci)
        camdict = {
            'serno'       : as_string(caminfo,'SerNo[16]'),
            'model'       : as_string(caminfo,'Model[16]'),
            'ID'          : int(getattr(caminfo,'dwCameraID')),
            'DeviceID'    : int(getattr(caminfo,'dwDeviceID')),
            'SensorID'    : int(getattr(caminfo,'dwSensorID')),
            'inUse'       : int(getattr(caminfo,'dwInUse'))
        }
        tlist['cameras'].append(camdict)

    return tlist

def GetCameraList():
    nCams = GetNumCameras()
    
    class UEYE_CAMERA_LIST(ctypes.Structure):
        _fields_ = [("dwCount", uc480.DWORD ),] + [("uci%d" %n, uc480.UEYE_CAMERA_INFO) for n in range(nCams)] #
        # _fields_ = [("dwCount", ctypes.wintypes.ULONG ),
        #             ("caminfo", uc480.UEYE_CAMERA_INFO * nCams)]

    camlist = UEYE_CAMERA_LIST()
    camlist.dwCount = nCams
    
    #print('dwCount:', camlist.dwCount, nCams)

    ret = uc480.CALL("GetCameraList", ctypes.byref(camlist))
    if not ret == uc480.IS_SUCCESS:
        raise RuntimeError('Error (%d) getting camera list' % ret)

    #print('dwCount:', camlist.dwCount, nCams)
    camlist.dwCount = nCams

    return translateCaminfo(camlist)


def check_mapexists(mdh, type = 'dark'):
    import os
    import PYME.Analysis.gen_sCMOS_maps as gmaps
    
    if type == 'dark':
        id = 'Camera.DarkMapID'
    elif type == 'variance':
        id = 'Camera.VarianceMapID'
    elif type == 'flatfield':
        id = 'Camera.FlatfieldMapID'
    else:
        raise RuntimeError('unknown map type %s' % type)
        
    mapPath = gmaps.mkDefaultPath(type,mdh,create=False)
    if os.path.exists(mapPath):
        mdh[id] = mapPath
        return mapPath
    else:
        return None

from PYME.Acquire.Hardware.Camera import Camera
class uc480Camera(Camera):
    numpy_frames=1

    ROIlimitlist = {
        'UI306x' : {
            'xmin' : 96,
            'xstep' : 8,
            'ymin' : 32,
            'ystep' : 2
        },
        'UI327x' : {
            'xmin' : 256,
            'xstep' : 8,
            'ymin' : 2,
            'ystep' : 2
        },
        'UI324x' : {
            'xmin' : 16,
            'xstep' : 4,
            'ymin' : 4,
            'ystep' : 2
        }
    }

    # this info is partly from the IDS datasheets that one can request for each camera model
    BaseProps = {
        'UI306x' : {
            # from Steve Hearn (IDS)
            #    The default gain of that camera is the absolute minimum gain the camera can deliver.
            #    All other gain factors are higher than that. This means, the system gain of 0.125 DN
            #    per electron, as specified in the camera test sheet, is the smallest possible value.
            'ElectronsPerCount'  : 7.97,
            'ReadNoise' : 6.0,
            'ADOffset' : 10
        },
        'default' : { # fairly arbitrary values
            'ElectronsPerCount'  : 10,
            'ReadNoise' : 20,
            'ADOffset' : 10
        }
    }


    ROIlimitsDefault = ROIlimitlist['UI324x']

    def __init__(self, boardNum=0, nbits = 8, isDeviceID = False):
        Camera.__init__(self)
        
        self._cont_mode = True
        self.initialised = False
        
        if nbits not in [8,10,12]:
            raise RuntimeError('Supporting only 8, 10 or 12 bit depth, requested %d bit' % (nbits))
        self.nbits = nbits

        self.boardHandle = uc480.HANDLE(boardNum)
        if isDeviceID:
            self.boardHandle = uc480.HANDLE(self.boardHandle.value | uc480.IS_USE_DEVICE_ID)

        ret = uc480.CALL('InitCamera', byref(self.boardHandle), uc480.HWND(0))
        print(('I',ret))
        if not ret == 0:
            raise RuntimeError('Error getting camera handle: %d: %s' % GetError(self.boardHandle))
            
        self.expTime = None
        self.initialised = True

        #register as a provider of metadata
        #MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)

        caminfo = uc480.CAMINFO()
        ret = uc480.CALL('GetCameraInfo', self.boardHandle, ctypes.byref(caminfo))
        if not ret == 0:
            raise RuntimeError('Error getting camera info: %d: %s' % GetError(self.boardHandle))
        
        self.serialNum = caminfo.SerNo

        logger.debug('caminfo: %s' %caminfo)

        #get the CCD size 
        sensorProps = uc480.SENSORINFO()
        
        ret = uc480.CALL('GetSensorInfo', self.boardHandle, ctypes.byref(sensorProps))
        if not ret == 0:
            raise RuntimeError('Error getting CCD size: %d: %s' % GetError(self.boardHandle))

        self.CCDSize=(sensorProps.nMaxWidth, sensorProps.nMaxHeight)
        senstype = ctypes.cast(sensorProps.strSensorName, ctypes.c_char_p)
        self.sensortype = senstype.value.decode()

        # work out the ROI limits for this sensortype
        matches = [self.ROIlimitlist[st] for st in self.ROIlimitlist.keys()
                   if self.sensortype.startswith(st)]
        if len(matches) > 0:
            self.ROIlimits = matches[0]
        else:
            self.ROIlimits = self.ROIlimitsDefault

        # work out the camera base parameters for this sensortype
        matches = [self.BaseProps[st] for st in self.BaseProps.keys()
                   if self.sensortype.startswith(st)]
        if len(matches) > 0:
            self.baseProps = matches[0]
        else:
            self.baseProps= self.BaseProps['default']

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
        
        # pick the desired monochrom mode
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
        self.dark = None
        
        #load flatfield (if present)
        calpath = nameUtils.getCalibrationDir(self.serialNum)
        ffname = os.path.join(calpath, 'flatfield.npy')
        if os.path.exists(ffname):
            self.flatfield = np.load(ffname).squeeze()
            self.flat = self.flatfield

        darkname = os.path.join(calpath, 'dark.npy')
        if os.path.exists(darkname):
            self.dark = np.load(darkname).squeeze()
            self.background = self.dark
        
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
        self.SetIntegTime(.1)
        
    def errcheck(self,value,msg,fatal=True):
        if not value == uc480.IS_SUCCESS:
            if fatal:
                raise RuntimeError('Error %s: %d: %s' % [msg]+GetError(self.boardHandle))
            else:
                print('Error %s: %d: %s' % [msg]+GetError(self.boardHandle))

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
        self.nFull = 0
            
    
    def _pollBuffer(self):
        pData = POINTER(c_char)()
        bufID = c_int(0)
        
        ret = uc480.CALL('WaitForNextImage', self.boardHandle, 1000, byref(pData), byref(bufID))
        
        if not ret == uc480.IS_SUCCESS:
            print('Wait for image failed with: %s' % ret)
            return
            
        ret = uc480.CALL('CopyImageMem', self.boardHandle, pData, bufID, self.transferBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        if not ret == uc480.IS_SUCCESS:
            print('CopyImageMem failed with: %' % ret)
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
                try:
                    self._pollBuffer()
                except:
                    traceback.print_exc()
            else:
                #print 'w',
                time.sleep(.05)
            time.sleep(.0005)
        
    def SetAcquisitionMode(self, mode):
        if mode == self.MODE_SINGLE_SHOT:
            self._cont_mode = False
        else:
            self._cont_mode = True
            
    def GetAcquisitionMode(self):
        if self._cont_mode:
            return self.MODE_CONTINUOUS
        else:
            return self.MODE_SINGLE_SHOT

    def SetIntegTime(self, iTime):
        #self.__selectCamera()
        newExp = c_double(0)
        newFrameRate = c_double(0)
        # call takes units of FPS
        ret = uc480.CALL('SetFrameRate', self.boardHandle, c_double(1.0/iTime), byref(newFrameRate))
        if not ret == 0:
            raise RuntimeError('Error setting exp time: %d: %s' % GetError(self.boardHandle))

        # call takes units of milliseconds, and has been depreciated since iDS version 3.9, use Exposure instead
        ret = uc480.CALL('SetExposureTime', self.boardHandle, c_double(1e3*iTime), ctypes.byref(newExp))
        if not ret == 0:
            raise RuntimeError('Error setting exp time: %d: %s' % GetError(self.boardHandle))

        #print newExp.value, newFrameRate.value
        logger.debug('exposure time: %f, new exposure time: %f' % (1e3 * iTime, newExp.value))
        logger.debug('frame rate: %f, new frame rate: %f' % (1./iTime, newFrameRate.value))
            
        self.expTime = newExp.value*1e-3

    def GetIntegTime(self):
        return self.expTime


    def GetCCDWidth(self):
        return self.CCDSize[0]

    def GetCCDHeight(self):
        return self.CCDSize[1]

    def SetHorizBin(self, val):
        """

        Parameters
        ----------
        val: int
            Binning factor. Supported values, depending on the camera, are 2, 3, 4, 5, 6, 8, or 16. Other values will be
            changed to nearest acceptable value, though there is no check for whether the camera supports it.

        Returns
        -------
        None

        Notes
        -----
        Not all ueye cameras support each binning factor. Our Thorlabs branded DCC1545M, for example, doesn't even
        support binning, only subsampling.

        """
        from PYME.Acquire.Hardware.uc480 import uc480_h
        binning = uc480.BINNING_FACTORS[np.argmin(np.abs(np.asarray(uc480.BINNING_FACTORS) - val))]
        logger.debug('Target binning: %d, Actual binning: %d' % (val, binning))
        setting = getattr(uc480_h, 'IS_BINNING_%dX_HORIZONTAL' % binning)
        uc480.CALL('SetBinning', self.boardHandle, setting)
        # calling SetFrameRate and Exposure is recommended after changing binning size
        self.SetIntegTime(self.GetIntegTime())

    def SetHorizontalBin(self, value):
        self.SetHorizBin(value)
        
    def GetHorzontalBin(self):
        return self.binX

    def SetVertBin(self, val):
        """

        Parameters
        ----------
        val: int
            Binning factor. Supported values, depending on the camera, are 2, 3, 4, 5, 6, 8, or 16. Other values will be
            changed to nearest acceptable value, though there is no check for whether the camera supports it.

        Returns
        -------
        None

        Notes
        -----
        Not all ueye cameras support each binning factor. Our Thorlabs branded DCC1545M, for example, doesn't even
        support binning, only subsampling.

        """
        from PYME.Acquire.Hardware.uc480 import uc480_h
        binning = uc480.BINNING_FACTORS[np.argmin(np.abs(np.asarray(uc480.BINNING_FACTORS) - val))]
        logger.debug('Target binning: %d, Actual binning: %d' % (val, binning))
        setting = getattr(uc480_h, 'IS_BINNING_%dX_VERTICAL' % binning)
        uc480.CALL('SetBinning', self.boardHandle, setting)
        # calling SetFrameRate and Exposure is recommended after changing binning size
        self.SetIntegTime(self.GetIntegTime())


    def SetVerticalBin(self, value):
        self.SetVertBin(value)
    
    def GetVerticalBin(self):
        return self.binY

    def GetCCDTemp(self):
        return  25 #self.CCDTemp

    def CamReady(self):
        return self.initialised

    def GetPicWidth(self):
        return int((self.ROIx[1] - self.ROIx[0])/self.binX)

    def GetPicHeight(self):
        return int((self.ROIy[1] - self.ROIy[0])/self.binY)


    def SetROI(self, x1,y1,x2,y2):
        #must be on xstep x ystep pixel grid

        xstep = self.ROIlimits['xstep']
        ystep = self.ROIlimits['ystep']
        x1 = int(xstep*np.floor(x1/xstep))
        x2 = int(xstep*np.floor(x2/xstep))
        y1 = int(ystep*np.floor(y1/ystep))
        y2 = int(ystep*np.floor(y2/ystep))
        
        
        #if coordinates are reversed, don't care
        if (x2 > x1):
            self.ROIx = [x1, x2]
        elif (x2 < x1):
            self.ROIx = [x2, x1]
        else: #x1 == x2 - BAD
            raise RuntimeError('Error Setting x ROI - Zero sized ROI')

        if (y2 > y1):
            self.ROIy = [y1, y2]

        elif (y2 < y1):
            self.ROIy = [y2, y1]

        else: #y1 == y2 - BAD
            raise RuntimeError('Error Setting y ROI - Zero sized ROI')
        
        rect = uc480.IS_RECT()
        rect.s32X =  self.ROIx[0]
        rect.s32Y =  self.ROIy[0]
        rect.s32Width  = max(self.ROIx[1] - self.ROIx[0], self.ROIlimits['xmin']) # ensure minimim size in x
        rect.s32Height = max(self.ROIy[1] - self.ROIy[0], self.ROIlimits['ymin']) # ensure minimim size in y
        # note: this can still fail on the right edge of the chip or bottom edge, since we then push beyond the
        # chip size limits FIXME!!!!!

        print("oldx: %d, newx: %d" % (self.ROIx[1],rect.s32X + rect.s32Width))
        print("oldy: %d, newy: %d" % (self.ROIy[1],rect.s32Y + rect.s32Height))

        self.ROIx[1] = rect.s32X + rect.s32Width  # make sure we feed the corrections back
        self.ROIy[1] = rect.s32Y + rect.s32Height # ditto

        print((rect.s32X, rect.s32Width))
        
        #ret = uc480.CALL('SetImageSize', self.boardHandle, rect.s32Width, rect.s32Height )
        #if not ret == 0:
        #    raise RuntimeError('Error setting image size: %d: %s' % GetError(self.boardHandle))
        
        ret = uc480.CALL('AOI', self.boardHandle, uc480.IS_AOI_IMAGE_SET_AOI, byref(rect), ctypes.sizeof(rect))
        if not ret == 0:
            raise RuntimeError('Error setting ROI: %d: %s' % GetError(self.boardHandle))
            
        if not self.flatfield is None:
            self.flat = self.flatfield[x1:x2, y1:y2]

        if not self.dark is None:
            self.background = self.dark[x1:x2, y1:y2]

        # we apparently have to set the integration time explicitly after a call to change the AOI
        # not sure if this is only for some IDS cameras or applies to all of them
        if self.GetIntegTime() is not None:
            self.SetIntegTime( self.GetIntegTime())
        #raise Exception, 'Not implemented yet!!'

    def GetROI(self):
        return self.ROIx[0], self.ROIy[0], self.ROIx[1], self.ROIy[1]
    

    def StartExposure(self):
        
        if self.doPoll:
            print('StartAq')
            self.StopAq()
        # allocate at least 2 seconds of buffers
        buffer_size = int(max(2 * self.GetFPS(), 50))
        self.InitBuffers(buffer_size, buffer_size)
        
        

        eventLog.logEvent('StartAq', '')
        if self._cont_mode:
            ret = uc480.CALL('CaptureVideo', self.boardHandle, uc480.IS_DONT_WAIT)
        else:
            ret = uc480.CALL('FreezeVideo', self.boardHandle, uc480.IS_DONT_WAIT)
        if not ret == 0:
            raise RuntimeError('Error starting exposure: %d: %s' % GetError(self.boardHandle))
        return 0


    #PYME Camera interface functions - make this look like the other cameras
    def ExpReady(self):
        #self._pollBuffer()
        
        return (not self.fullBuffers is None) and (not self.fullBuffers.empty())
        
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
        
        if (not self.background is None) and self.background.shape == chSlice.shape:
            chSlice[:] = (chSlice - np.minimum(chSlice, self.background))[:]
            
        if (not self.flat is None) and self.flat.shape == chSlice.shape:
            chSlice[:] = (chSlice*self.flat).astype('uint16')[:]
        
        #ret = uc480.CALL('UnlockSeqBuf', self.boardHandle, uc480.IS_IGNORE_PARAMETER, pData)

        if not self.freeBuffers is None:
            #recycle buffer
            self.freeBuffers.put(buf)
        

    def StopAq(self):
        ret = uc480.CALL('StopLiveVideo', self.boardHandle, uc480.IS_WAIT)
        self.DestroyBuffers()

    def SetCCDTemp(self, temp):
        pass

    def SetEMGain(self, gain):
        raise NotImplementedError()

    def GetEMGain(self):
        return self.EMGain
        
    def SetGainBoost(self, on):
        if on:
            uc480.CALL('SetGainBoost', self.boardHandle, uc480.IS_SET_GAINBOOST_ON)
        else:
            uc480.CALL('SetGainBoost', self.boardHandle, uc480.IS_SET_GAINBOOST_OFF)
            
    def SetGain(self, gain=100):
        uc480.CALL('SetHardwareGain', self.boardHandle, gain, uc480.IS_IGNORE_PARAMETER, uc480.IS_IGNORE_PARAMETER, uc480.IS_IGNORE_PARAMETER)
        
    def GetGain(self):
        ret = uc480.CALL('SetHardwareGain', self.boardHandle, uc480.IS_GET_MASTER_GAIN, uc480.IS_IGNORE_PARAMETER, uc480.IS_IGNORE_PARAMETER, uc480.IS_IGNORE_PARAMETER)
        return ret

    def GetGainFactor(self):
        gain = self.GetGain()
        ret = uc480.CALL('SetHWGainFactor', self.boardHandle, uc480.IS_INQUIRE_MASTER_GAIN_FACTOR, gain)
        return 0.01*ret

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
        if fps.value == 0:
            return 1.0/self.GetIntegTime()
        else:
            return fps.value
    
    def GetNumImsBuffered(self):
        return self.nFull

    def GetBufferSize(self):
        return len(self._buffers)
        
    def GetSerialNumber(self):
        return self.serialNum

    def GetHeadModel(self):
        return self.sensortype

    def GetElectronsPerCount(self):
        return (self.baseProps['ElectronsPerCount']/self.GetGainFactor())

    def GetReadNoise(self): # readnoise in e-
        return (self.baseProps['ReadNoise'])

    def GetADOffset(self):
        return self.baseProps['ADOffset']
    
    @property
    def noise_properties(self):
        return {'ElectronsPerCount': self.GetElectronsPerCount(),
                'ReadNoise': self.GetReadNoise(),
                'ADOffset': self.GetADOffset(),
                'SaturationThreshold': 2 ** self.nbits  - 1}

    def GenStartMetadata(self, mdh):
        if self.active: #we are active -> write metadata
            self.GetStatus()

            mdh.setEntry('Camera.Name', 'UC480-UEYE')
            mdh.setEntry('Camera.Model', self.GetHeadModel())
            mdh.setEntry('Camera.SerialNumber', self.GetSerialNumber())

            mdh.setEntry('Camera.IntegrationTime', self.GetIntegTime())
            mdh.setEntry('Camera.CycleTime', 1./self.GetFPS())

            mdh.setEntry('Camera.HardwareGain', self.GetGain())
            mdh.setEntry('Camera.HardwareGainFactor', self.GetGainFactor())
            mdh.setEntry('Camera.ElectronsPerCount', self.GetElectronsPerCount())
            mdh.setEntry('Camera.ADOffset', self.GetADOffset())
            mdh.setEntry('Camera.ReadNoise',self.GetReadNoise()) # in units of e-
            mdh.setEntry('Camera.NoiseFactor', 1.0)

            mdh.setEntry('Camera.SensorWidth',self.GetCCDWidth())
            mdh.setEntry('Camera.SensorHeight',self.GetCCDHeight())
            mdh.setEntry('Camera.TrueEMGain', 1)

            #mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
            #mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())
            
            x1, y1, x2, y2 = self.GetROI()
            mdh.setEntry('Camera.ROIOriginX', x1)
            mdh.setEntry('Camera.ROIOriginY', y1)
            mdh.setEntry('Camera.ROIWidth', x2 - x1)
            mdh.setEntry('Camera.ROIHeight', y2 - y1)
            #mdh.setEntry('Camera.StartCCDTemp',  self.GetCCDTemp())

            check_mapexists(mdh,type='dark')
            check_mapexists(mdh,type='variance')
            check_mapexists(mdh,type='flatfield')

    def __del__(self):
        if self.initialised:
            self.Shutdown()

