#!/usr/bin/python

##################
# AndorIXon.py
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

from . import AndorCam as ac
from ctypes import *
import time
import sys
from PYME.IO import MetaDataHandler
from PYME.Acquire.Hardware import ccdCalibrator
from PYME.Acquire.Hardware import camera_noise

#import example
#import scipy

#import pylab

from PYME.Acquire import eventLog

#import threading

from PYME.Acquire.Hardware.Camera import Camera
class iXonCamera(Camera):
    numpy_frames=False

    @property
    def _gain_mode(self):
        return 'Preamp Gain %d' % self.preampGain

    #define a couple of acquisition modes

    #MODE_CONTINUOUS = 5
    #MODE_SINGLE_SHOT = 1
    
    _IXON_MODE_CONTINUOUS = 5
    _IXON_MODE_SINGLE_SHOT = 1

    def __selectCamera(self):
        ret = ac.SetCurrentCamera(self.boardHandle)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting camera: %s' % ac.errorCodes[ret])


    def __init__(self, boardNum=0):
        Camera.__init__(self)
        
        self.initialised = False
        self.active = False

        self.boardHandle = ac.GetCameraHandle.argtypes[1]._type_()

        ret = ac.GetCameraHandle(boardNum, byref(self.boardHandle))
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error getting camera handle: %s' % ac.errorCodes[ret])

        self.__selectCamera()


        #register as a provider of metadata
        #MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)

        #initialise the camera - n.b. this take ~2s

        #ret = ac.Initialize('.')
        if 'linux' in sys.platform:
            ret = ac.Initialize(b"/usr/local/etc/andor")
        else:
            ret = ac.Initialize(b'.')

        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error initialising camera: %s' % ac.errorCodes[ret])
        else:
            self.initialised = True

        #self.noiseProps = noiseProperties[self.GetSerialNumber()]

        #get the CCD size
        ccdWidth = ac.GetDetector.argtypes[0]._type_()
        ccdHeight = ac.GetDetector.argtypes[1]._type_()

        ret = ac.GetDetector(byref(ccdWidth),byref(ccdHeight))
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error getting CCD size: %s' % ac.errorCodes[ret])

        self.CCDSize=(ccdWidth.value, ccdHeight.value)

        tMin = ac.GetTemperatureRange.argtypes[0]._type_()
        tMax = ac.GetTemperatureRange.argtypes[1]._type_()

        ret = ac.GetTemperatureRange(byref(tMin),byref(tMax))
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error getting Temperature range: %s' % ac.errorCodes[ret])

        self.tRange = (tMin.value,tMax.value)
        self.tempSet = -70 #default temperature setpoint

        ret = ac.SetTemperature(max(tMin.value, self.tempSet)) #fixme so that default T is read in
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting Temperature: %s' % ac.errorCodes[ret])

        ret = ac.CoolerON()
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error turnig cooler on: %s' % ac.errorCodes[ret])

        self.tempStable = False

        #-------------------
        #Do initial setup with a whole bunch of settings I've arbitrarily decided are
        #reasonable and make the camera behave more or less like the PCO cameras.
        #These settings will hopefully be able to be changed with methods/ read from
        #a file later.

        #continuous acquisition
        ret = ac.SetAcquisitionMode(self._IXON_MODE_CONTINUOUS)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting aq mode: %s' % ac.errorCodes[ret])

        self._contMode = True #we are in single shot mode

        ret = ac.SetReadMode(4) #readout as image rather than doing any fancy stuff
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting readout mode: %s' % ac.errorCodes[ret])



        #ret = ac.SetExposureTime(0.1)
        #if not ret == ac.DRV_SUCCESS:
        #    raise RuntimeError('Error setting exp time: %s' % ac.errorCodes[ret])

        ret = ac.SetTriggerMode(0) #use internal triggering
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting trigger mode: %s' % ac.errorCodes[ret])

        self._InitSpeedInfo() #see what shift speeds the device is capable of

        #use the fastest reccomended shift speed by default
        self.VSSpeed = self.fastestRecVSInd
        ret = ac.SetVSSpeed(self.VSSpeed)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting VS speed: %s' % ac.errorCodes[ret])

        self.HSSpeed = 0
        ret = ac.SetHSSpeed(0,self.HSSpeed)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting HS speed: %s' % ac.errorCodes[ret])

        #FIXME - do something about selecting A/D channel
        
        #set the preamp gain if we have data for our camera, otherwise default to highest
        # NOTE: this is important as other software may have left it in an undefined state
        # NOTE: We cheat a bit here and store this in noise_properties
        self.preampGain = camera_noise.noise_properties.get(self.GetSerialNumber(), {}).get('default_preamp_gain',2)
        ret = ac.SetPreAmpGain(self.preampGain)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting Preamp gain: %s' % ac.errorCodes[ret])


        self.binning=False #binning flag - binning is off
        self.binX=1 #1x1
        self.binY=1
        self.ROIx=(1,512)
        self.ROIy=(1,512)

        ret = ac.SetImage(self.binX,self.binY,self.ROIx[0],self.ROIx[1],self.ROIy[0],self.ROIy[1])
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting image size: %s' % ac.errorCodes[ret])

        ret = ac.SetEMGainMode(0)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting EM Gain Mode: %s' % ac.errorCodes[ret])

        self.EMGain = 0 #start with zero EM gain
        ret = ac.SetEMCCDGain(self.EMGain)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting EM Gain: %s' % ac.errorCodes[ret])

        #set an initial integration time        
        self.SetIntegTime(.100)        
        
        self.CCDTemp = -999
        self.SetFrameTransfer(True)
        #self.frameTransferMode = False

        #set the shutter to be open (for most applications we're going to be shuttering
        #the excitation anyway - no use in wearing the internal shutter out).
        self.shutterOpen = True
        ret = ac.SetShutter(1,1,0,0) #only the 2nd parameter is important as we're leaving the shutter open
        if not ret == ac.DRV_SUCCESS:
            #raise RuntimeError('Error setting shutter: %s' % ac.errorCodes[ret])
            print(('Error setting shutter: %s' % ac.errorCodes[ret]))

        


    def _InitSpeedInfo(self):
        #temporary vars for function returns
        tNum = ac.GetNumberVSSpeeds.argtypes[0]._type_()
        tmp = c_float()

        ret = ac.GetNumberVSSpeeds(byref(tNum))
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error getting num VS Speeds: %s' % ac.errorCodes[ret])

        numVSSpeeds = tNum.value

        self.vertShiftSpeeds = []
        for i in range(numVSSpeeds):
            ac.GetVSSpeed(i,byref(tmp))
            if not ret == ac.DRV_SUCCESS:
                raise RuntimeError('Error getting VS Speed %d: %s' % (i,ac.errorCodes[ret]))
            self.vertShiftSpeeds.append(tmp.value)

        ret = ac.GetNumberAmp(byref(tNum))
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error getting num Amps: %s' % ac.errorCodes[ret])

        self.numAmps = int(tNum.value)

        self.HorizShiftSpeeds = []
        for i in range(self.numAmps):
            HSSpeeds = []

            for gainType in [0,1]: #0 = EM gain, 1 = conventional
                ret = ac.GetNumberHSSpeeds(i,gainType, byref(tNum))
                if not ret == ac.DRV_SUCCESS:
                    print(('Error getting num HS Speeds (%d,%d): %s' % (i, gainType, ac.errorCodes[ret])))

                HSSpeedsG = []

                nhs = int(tNum.value)

                for j in range(nhs):
                    ac.GetHSSpeed(i,gainType, j, byref(tmp))
                    if not ret == ac.DRV_SUCCESS:
                        print(('Error getting VS Speed %d: %s' % (i,ac.errorCodes[ret])))
                    HSSpeedsG.append(float(tmp.value))

                HSSpeeds.append(HSSpeedsG)

            self.HorizShiftSpeeds.append(HSSpeeds)

        ret = ac.GetFastestRecommendedVSSpeed(byref(tNum), byref(tmp))
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error getting fastest rec. VS Speed: %s' % ac.errorCodes[ret])

        self.fastestRecVSInd = int(tNum.value)


#    def SetContinuousMode(self, value=True):
#        if value:
#            self.SetAcquisitionMode(self.MODE_CONTINUOUS)
#        else:
#            self.SetAcquisitionMode(self.MODE_SINGLE_SHOT)
#
#            
#    def GetContinuousMode(self):
#        return self.contMode

    def SetPreampGain(self, gain):
        self.__selectCamera()
        self.preampGain = gain
        ret = ac.SetPreAmpGain(self.preampGain)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting Preamp gain: %s' % ac.errorCodes[ret])


    def SetIntegTime(self, iTime):
        self.__selectCamera()
        ret = ac.SetExposureTime(iTime)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting exp time: %s' % ac.errorCodes[ret])

    def GetIntegTime(self):
        self.__selectCamera()
        exp = c_float()
        acc = c_float()
        kin = c_float()

        ret = ac.GetAcquisitionTimings(byref(exp),byref(acc),byref(kin))
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting exp time: %s' % ac.errorCodes[ret])

        return float(exp.value)

    def GetCCDWidth(self):
        return self.CCDSize[0]

    def GetCCDHeight(self):
        return self.CCDSize[1]

    def SetHorizBin(self, val):
        self.__selectCamera()
        self.binX = val
        
        ret = ac.SetImage(self.binX,self.binY,self.ROIx[0],self.ROIx[1],self.ROIy[0],self.ROIy[1])
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting image size: %s' % ac.errorCodes[ret])
        #raise Exception, 'Not implemented yet!!'
        
    def SetHorizontalBin(self, value):
        self.SetHorizBin(value)

    def GetHorizontalBin(self, *args):
        return self.binX

    def SetVertBin(self, val):
        self.__selectCamera()
        self.binY = val

        ret = ac.SetImage(self.binX,self.binY,self.ROIx[0],self.ROIx[1],self.ROIy[0],self.ROIy[1])
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting image size: %s' % ac.errorCodes[ret])

    def SetVerticalBin(self, value):
        self.SetVertBin(value)

    def GetVerticalBin(self):
        return self.binY


    def GetCCDTemp(self):
        return self.CCDTemp

    def _GetCCDTemp(self):
        self.__selectCamera()
        t = c_float()
        ret = ac.GetTemperatureF(byref(t))
        #print ret
        self.tempStable = (ret == ac.DRV_TEMP_STABILIZED)
        self.CCDTemp = int(t.value)

    def _GetAcqTimings(self):
        self.__selectCamera()
        exp = c_float()
        acc = c_float()
        kin = c_float()

        ret = ac.GetAcquisitionTimings(byref(exp), byref(acc), byref(kin))

        self.tExp = exp.value
        self.tAcc = acc.value
        self.tKin = kin.value

    def CamReady(self):
        return self.initialised

        #    return false
        #else:
        #    tmp = c_long()
        #    ret = ac.GetStatus(tmp)
        #    if not ret == ac.DRV_SUCCESS:
        #        raise RuntimeError('Error getting camera status: %s' % ac.errorCodes[ret])
        #    return tmp == ac.DRV_IDLE

    def GetPicWidth(self):
        return int((self.ROIx[1] - self.ROIx[0] + 1)/self.binX)
        #return self.CCDSize[0]

    def GetPicHeight(self):
        return int((self.ROIy[1] - self.ROIy[0] + 1)/self.binY)
        #return self.CCDSize[1]

    def SetROI(self, x1,y1,x2,y2):
        self.__selectCamera()
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

        ret = ac.SetImage(self.binX,self.binY,self.ROIx[0],self.ROIx[1],self.ROIy[0],self.ROIy[1])
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting image size: %s' % ac.errorCodes[ret])

    # def GetROIX1(self):
    #     return self.ROIx[0]
    #
    # def GetROIX2(self):
    #     return self.ROIx[1]
    #
    # def GetROIY1(self):
    #     return self.ROIy[0]
    #
    # def GetROIY2(self):
    #     return self.ROIy[1]
    
    def GetROI(self):
        return self.ROIx[0] - 1, self.ROIy[0] -1, self.ROIx[1], self.ROIy[1]

    def StartExposure(self):
        self.__selectCamera()
        self._GetCCDTemp()
        self._GetAcqTimings()
        self._GetBufferSize()

        eventLog.logEvent('StartAq', '')
        ret = ac.StartAcquisition()
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error starting acquisition: %s' % ac.errorCodes[ret])
        return 0

    def StartLifePreview(*args):
        raise Exception('Not implemented yet!!')

    def StopLifePreview(*args):
        raise Exception('Not implemented yet!!')

    def ExpReady(self):
        self.__selectCamera()
        tmp = ac.GetStatus.argtypes[0]._type_()
        ret = ac.GetStatus(byref(tmp))
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error getting camera status: %s' % ac.errorCodes[ret])

        #print ac.errorCodes[tmp.value]

        #return tmp.value == ac.DRV_IDLE
        first= ac.GetNumberNewImages.argtypes[0]._type_()
        last= ac.GetNumberNewImages.argtypes[1]._type_()
        ret = ac.GetNumberNewImages(byref(first), byref(last))
        #print ac.errorCodes[ret]
        #print first
        #print last
        return ret == ac.DRV_SUCCESS 

    def GetBWPicture(*args):
        raise Exception('Not implemented yet!!')

    def ExtractColor(self, chSlice, mode):
        self.__selectCamera()
        #print chSlice
        #pc = chSlice.split('_')[1]
        #pc = chSlice.split('_')[1]
        #ret = ac.GetAcquiredData16(cast(c_void_p(int(pc[6:8]+pc[4:6]+pc[2:4]+pc[0:2],16)), POINTER(c_ushort)), self.GetPicWidth()*self.GetPicHeight())
        #ret = ac.GetAcquiredData16(cast(c_void_p(int(chSlice)), POINTER(c_ushort)), self.GetPicWidth()*self.GetPicHeight())

        dt = ac.GetOldestImage16.argtypes[0]
        ret = ac.GetOldestImage16(cast(c_void_p(int(chSlice)), dt), self.GetPicWidth()*self.GetPicHeight())

        #print self.GetPicWidth()*self.GetPicHeight()
        if not ret == ac.DRV_SUCCESS:
            print(('Error getting image from camera: %s' % ac.errorCodes[ret]))


    def StopAq(self):
        self.__selectCamera()
        ac.AbortAcquisition()

    def SetCCDTemp(self, temp):
        self.__selectCamera()
        if (temp < self.tRange[0]) or (temp > self.tRange[1]):
            raise RuntimeError('Temperature setpoint out of range ([%d,%d])' % self.tRange)

        self.tempSet = temp #temperature setpoint

        ret = ac.SetTemperature(self.tempSet) #fixme so that default T is read in
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting Temperature: %s' % ac.errorCodes[ret])

    def SetEMGain(self, gain):
        self.__selectCamera()
        if (gain > 150): #150 is arbitrary, but seems like a reasomable enough boundary
            print(('WARNING: EM Gain of %d selected; overuse of high gains can lead to gain register aging' % gain))

        self.EMGain = gain
        ret = ac.SetEMCCDGain(self.EMGain)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting EM Gain: %s' % ac.errorCodes[ret])

    def GetEMGain(self):
        return self.EMGain

    def WaitForExp(self):
        self.__selectCamera()
        ret = ac.WaitForAcquisition() #block until aquisition is finished
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error waiting for acquisition: %s' % ac.errorCodes[ret])

    def Shutdown(self):
        print('Shutting down EMCCD')
        self.__selectCamera()
        self.SetShutter(False)
        self.SetEMGain(0)
        ac.CoolerOFF()
        t = ac.GetTemperature.argtypes[0]._type_(-100)
        ac.GetTemperature(byref(t))

        while False:#(t.value < -50): #wait fro temp to get above -50
            print(('Waiting for the camera to warm up - current temperature = %3.2f' % t.value))
            time.sleep(1)
            ac.GetTemperature(byref(t))

        ac.ShutDown()
        self.initialised = False

    def SpoolOn(self, filename):
        self.__selectCamera()
        ac.SetAcquisitionMode(self._IXON_MODE_CONTINUOUS)
        ac.SetSpool(1,2,filename,10)
        ac.StartAcquisition()

    def SpoolOff(self):
        self.__selectCamera()
        ac.AbortAcquisition()
        ac.SetSpool(0,2,r'D:\spool\spt',10)
        ac.SetAcquisitionMode(self._IXON_MODE_SINGLE_SHOT)

    def GetCCDTempSetPoint(self):
        return self.tempSet

    def SetAcquisitionMode(self, aqMode):
        self.__selectCamera()
        ac.AbortAcquisition()
        if aqMode == self.MODE_CONTINUOUS:
            ac.SetAcquisitionMode(self._IXON_MODE_CONTINUOUS)
            self._contMode = True
        elif aqMode == self.MODE_SINGLE_SHOT:
            ac.SetAcquisitionMode(self._IXON_MODE_SINGLE_SHOT)
            self._contMode = False
        else:
            raise RuntimeError('Mode %d not supported' % aqMode)
        
    def GetAcquisitionMode(self):
        if self.contMode:
            return self.MODE_CONTINUOUS
        else:
            return self.MODE_SINGLE_SHOT
        
    @property
    def contMode(self):
        return self._contMode
    
    @contMode.setter
    def contMode(self, val):
        self._contMode = val

    def SetFrameTransfer(self, ftMode):
        self.__selectCamera()
        self.frameTransferMode = ftMode

        if ftMode:
            ac.SetFrameTransferMode(1)
        else:
            ac.SetFrameTransferMode(0)
            
    def SetVerticalShiftSpeed(self, speedIndex):
        self.__selectCamera()
        self.VSSpeed = speedIndex
        ret = ac.SetVSSpeed(self.VSSpeed)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting VS speed: %s' % ac.errorCodes[ret])

        #increase clock voltage to prevent smear if using fastest v.s. speed
        #THIS MAY NEED TWEAKING
        if speedIndex == 0:
            ac.SetVSAmplitude(1)
        else:
            ac.SetVSAmplitude(0)
            
    def SetHorizShiftSpeed(self, speedIndex):
        self.__selectCamera()
        self.HSSpeed = 0
        ret = ac.SetHSSpeed(0,self.HSSpeed)
        if not ret == ac.DRV_SUCCESS:
            raise RuntimeError('Error setting HS speed: %s' % ac.errorCodes[ret])
        
    def GetFPS(self):
        return 1.0/self.tKin
    
    def GetNumImsBuffered(self):
        self.__selectCamera()
        first = ac.GetNumberNewImages.argtypes[0]._type_()
        last = ac.GetNumberNewImages.argtypes[1]._type_()

        ret = ac.GetNumberNewImages(byref(first), byref(last))
        #print ac.errorCodes[ret]
        #print first
        #print last
        return last.value - first.value
    
    def _GetBufferSize(self):
        self.__selectCamera()
        bs = ac.GetSizeOfCircularBuffer.argtypes[0]._type_()
        ac.GetSizeOfCircularBuffer(byref(bs))
        self.bs = bs.value

    def GetBufferSize(self):
        return self.bs

    def SetShutter(self, state):
        self.__selectCamera()
        if self.GetSerialNumber() == 1823:
            ac.SetShutter(int(state), 1, 0, 0)
        else:
            s2 = 2 - int(state)
            ac.SetShutterEx(int(state), s2, 0, 0,s2)
        self.shutterOpen = state
        
    def SetBaselineClamp(self, state):
        self.__selectCamera()
        ac.SetBaselineClamp(int(state))

    def GetBaselineClamp(self):
        self.__selectCamera()
        state = ac.GetBaselineClamp.argtypes[0]._type_()
        ac.GetBaselineClamp(byref(state))
        return state.value == 1

    def SetFan(self, state):
        self.__selectCamera()
        ac.SetFanMode(state)

    def GetSerialNumber(self):
        self.__selectCamera()
        sn = ac.GetCameraSerialNumber.argtypes[0]._type_()
        ac.GetCameraSerialNumber(byref(sn))
        return sn.value

    def GetHeadModel(self):
        self.__selectCamera()
        hm = create_string_buffer(255)
        ac.GetHeadModel(hm)
        return hm.value.decode()
    
    #@property
    #def noise_properties(self):
    #    return self.noiseProps

    def GenStartMetadata(self, mdh):
        if self.active: #we are active -> write metadata
            self.GetStatus()

            mdh.setEntry('Camera.Name', 'Andor IXon DV97')
            mdh.setEntry('Camera.Model', self.GetHeadModel())
            mdh.setEntry('Camera.SerialNumber', self.GetSerialNumber())

            mdh.setEntry('Camera.IntegrationTime', self.tExp)
            mdh.setEntry('Camera.CycleTime', self.tKin)
            mdh.setEntry('Camera.EMGain', self.GetEMGain())

            #mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
            #mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())

            x1, y1, x2, y2 = self.GetROI()
            mdh.setEntry('Camera.ROIOriginX', x1)
            mdh.setEntry('Camera.ROIOriginY', y1)
            mdh.setEntry('Camera.ROIWidth', x2-x1)
            mdh.setEntry('Camera.ROIHeight',  y2-y1)
            
            mdh.setEntry('Camera.StartCCDTemp',  self.GetCCDTemp())

            #these should really be read from a configuration file
            #hard code them here until I get around to it
            #current values are at 10Mhz using e.m. amplifier
            np = self.noise_properties
            mdh.setEntry('Camera.ReadNoise', np['ReadNoise'])
            mdh.setEntry('Camera.NoiseFactor', 1.41)
            mdh.setEntry('Camera.ElectronsPerCount', np['ElectronsPerCount'])

            realEMGain = ccdCalibrator.getCalibratedCCDGain(self.GetEMGain(), self.GetCCDTempSetPoint())
            if not realEMGain is None:
                mdh.setEntry('Camera.TrueEMGain', realEMGain)

    # we should not need this
    #def __getattr__(self, name):
    #    if name in list(self.noiseProps.keys()):
    #        return self.noiseProps[name]
    #    else:  raise AttributeError(name)  # <<< DON'T FORGET THIS LINE !!


    def __del__(self):
        if self.initialised:
            self.Shutdown()

