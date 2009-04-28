import AndorCam as ac
from ctypes import *
import time
from PYME.Acquire import MetaDataHandler
from PYME.Hardware import ccdCalibrator

#import example
#import scipy

#import pylab

#import threading

class iXonCamera:
    #numpy_frames=1

    #define a couple of acquisition modes

    MODE_CONTINUOUS = 5
    MODE_SINGLE_SHOT = 1

    def __init__(self):
        self.initialised = False

        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)

        #initialise the camera - n.b. this take ~2s
        ret = ac.Initialize('.')

        if not ret == ac.DRV_SUCCESS:
            raise 'Error initialising camera: %s' % ac.errorCodes[ret]
        else:
            self.initialised = True

        #get the CCD size
        ccdWidth = c_int()
        ccdHeight = c_int()

        ret = ac.GetDetector(byref(ccdWidth),byref(ccdHeight))
        if not ret == ac.DRV_SUCCESS:
            raise 'Error getting CCD size: %s' % ac.errorCodes[ret]

        self.CCDSize=(ccdWidth.value, ccdHeight.value)

        tMin = c_int()
        tMax = c_int()

        ret = ac.GetTemperatureRange(byref(tMin),byref(tMax))
        if not ret == ac.DRV_SUCCESS:
            raise 'Error getting Temperature range: %s' % ac.errorCodes[ret]

        self.tRange = (tMin.value,tMax.value)
        self.tempSet = -50 #default temperature setpoint

        ret = ac.SetTemperature(max(tMin.value, self.tempSet)) #fixme so that default T is read in
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting Temperature: %s' % ac.errorCodes[ret]

        ret = ac.CoolerON()
        if not ret == ac.DRV_SUCCESS:
            raise 'Error turnig cooler on: %s' % ac.errorCodes[ret]

        self.tempStable = False

        #-------------------
        #Do initial setup with a whole bunch of settings I've arbitrarily decided are
        #reasonable and make the camera behave more or less like the PCO cameras.
        #These settings will hopefully be able to be changed with methods/ read from
        #a file later.

        #continuous acquisition
        ret = ac.SetAcquisitionMode(self.MODE_CONTINUOUS)
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting aq mode: %s' % ac.errorCodes[ret]

        self.contMode = True #we are in single shot mode

        ret = ac.SetReadMode(4) #readout as image rather than doing any fancy stuff
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting readout mode: %s' % ac.errorCodes[ret]



        #ret = ac.SetExposureTime(0.1)
        #if not ret == ac.DRV_SUCCESS:
        #    raise 'Error setting exp time: %s' % ac.errorCodes[ret]

        ret = ac.SetTriggerMode(0) #use internal triggering
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting trigger mode: %s' % ac.errorCodes[ret]

        self._InitSpeedInfo() #see what shift speeds the device is capable of

        #use the fastest reccomended shift speed by default
        self.VSSpeed = self.fastestRecVSInd
        ret = ac.SetVSSpeed(self.VSSpeed)
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting VS speed: %s' % ac.errorCodes[ret]

        self.HSSpeed = 0
        ret = ac.SetHSSpeed(0,self.HSSpeed)
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting HS speed: %s' % ac.errorCodes[ret]

        #FIXME - do something about selecting A/D channel

        self.binning=False #binning flag - binning is off
        self.binX=1 #1x1
        self.binY=1
        self.ROIx=(1,512)
        self.ROIy=(1,512)

        ret = ac.SetImage(self.binX,self.binY,self.ROIx[0],self.ROIx[1],self.ROIy[0],self.ROIy[1])
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting image size: %s' % ac.errorCodes[ret]

        ret = ac.SetEMGainMode(0)
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting EM Gain Mode: %s' % ac.errorCodes[ret]

        self.EMGain = 0 #start with zero EM gain
        ret = ac.SetEMCCDGain(self.EMGain)
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting EM Gain: %s' % ac.errorCodes[ret]

        self.CCDTemp = -999
        self.SetFrameTransfer(True)
        #self.frameTransferMode = False

        #set the shutter to be open (for most applications we're going to be shuttering
        #the excitation anyway - no use in wearing the internal shutter out).
        ret = ac.SetShutter(1,1,0,0) #only the 2nd parameter is important as we're leaving the shutter open
        if not ret == ac.DRV_SUCCESS:
            #raise 'Error setting shutter: %s' % ac.errorCodes[ret]
            print 'Error setting shutter: %s' % ac.errorCodes[ret]

        


    def _InitSpeedInfo(self):
        #temporary vars for function returns
        tNum = c_int()
        tmp = c_float()

        ret = ac.GetNumberVSSpeeds(byref(tNum))
        if not ret == ac.DRV_SUCCESS:
            raise 'Error getting num VS Speeds: %s' % ac.errorCodes[ret]

        numVSSpeeds = tNum.value

        self.vertShiftSpeeds = []
        for i in range(numVSSpeeds):
            ac.GetVSSpeed(i,byref(tmp))
            if not ret == ac.DRV_SUCCESS:
                raise 'Error getting VS Speed %d: %s' % (i,ac.errorCodes[ret])
            self.vertShiftSpeeds.append(tmp.value)

        ret = ac.GetNumberAmp(byref(tNum))
        if not ret == ac.DRV_SUCCESS:
            raise 'Error getting num Amps: %s' % ac.errorCodes[ret]

        self.numAmps = int(tNum.value)

        self.HorizShiftSpeeds = []
        for i in range(self.numAmps):
            HSSpeeds = []

            for gainType in [0,1]: #0 = EM gain, 1 = conventional
                ret = ac.GetNumberHSSpeeds(i,gainType, byref(tNum))
                if not ret == ac.DRV_SUCCESS:
                    print 'Error getting num HS Speeds (%d,%d): %s' % (i, gainType, ac.errorCodes[ret])

                HSSpeedsG = []

                nhs = int(tNum.value)

                for j in range(nhs):
                    ac.GetHSSpeed(i,gainType, j, byref(tmp))
                    if not ret == ac.DRV_SUCCESS:
                        print 'Error getting VS Speed %d: %s' % (i,ac.errorCodes[ret])
                    HSSpeedsG.append(float(tmp.value))

                HSSpeeds.append(HSSpeedsG)

            self.HorizShiftSpeeds.append(HSSpeeds)

        ret = ac.GetFastestRecommendedVSSpeed(byref(tNum), byref(tmp))
        if not ret == ac.DRV_SUCCESS:
            raise 'Error getting fastest rec. VS Speed: %s' % ac.errorCodes[ret]

        self.fastestRecVSInd = int(tNum.value)


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
        ret = ac.SetExposureTime(iTime*1e-3)
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting exp time: %s' % ac.errorCodes[ret]

    def GetIntegTime(self):
        exp = c_float()
        acc = c_float()
        kin = c_float()

        ret = ac.GetAcquisitionTimings(byref(exp),byref(acc),byref(kin))
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting exp time: %s' % ac.errorCodes[ret]

        return float(exp.value)

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
        return self.CCDSize[0]

    def GetCCDHeight(self):
        return self.CCDSize[1]

    def SetHorizBin(*args):
        raise Exception, 'Not implemented yet!!'

    def GetHorizBin(self):
        return self.binning
        #raise Exception, 'Not implemented yet!!'

    def GetHorzBinValue(*args):
        #raise Exception, 'Not implemented yet!!'
        return self.binX

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
        return self.CCDTemp

    def _GetCCDTemp(self):
        t = c_int()
        ret = ac.GetTemperature(byref(t))
        #print ret
        self.tempStable = (ret == ac.DRV_TEMP_STABILIZED)
        self.CCDTemp = int(t.value)

    def _GetAcqTimings(self):
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
        #    tmp = c_int()
        #    ret = ac.GetStatus(tmp)
        #    if not ret == ac.DRV_SUCCESS:
        #        raise 'Error getting camera status: %s' % ac.errorCodes[ret]
        #    return tmp == ac.DRV_IDLE

    def GetPicWidth(self):
        return (self.ROIx[1] - self.ROIx[0] + 1)/self.binX
        #return self.CCDSize[0]

    def GetPicHeight(self):
        return (self.ROIy[1] - self.ROIy[0] + 1)/self.binY
        #return self.CCDSize[1]

    def SetROI(self, x1,y1,x2,y2):
        #if coordinates are reversed, don't care
        if (x2 > x1):
            self.ROIx = (x1+1, x2)
        elif (x2 < x1):
            self.ROIx = (x2+1, x1)
        else: #x1 == x2 - BAD
            raise 'Error Setting x ROI - Zero sized ROI'

        if (y2 > y1):
            self.ROIy = (y1+1, y2)

        elif (y2 < y1):
            self.ROIy = (y2+1, y1)

        else: #y1 == y2 - BAD
            raise 'Error Setting y ROI - Zero sized ROI'

        ret = ac.SetImage(self.binX,self.binY,self.ROIx[0],self.ROIx[1],self.ROIy[0],self.ROIy[1])
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting image size: %s' % ac.errorCodes[ret]

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

    def Init(*args):
        pass

    def GetStatus(*args):
        pass

    def SetCOC(*args):
        pass

    def StartExposure(self):
        self._GetCCDTemp()
        self._GetAcqTimings()
        self._GetBufferSize()
        ret = ac.StartAcquisition()
        if not ret == ac.DRV_SUCCESS:
            raise 'Error starting acquisition: %s' % ac.errorCodes[ret]
        return 0

    def StartLifePreview(*args):
        raise Exception, 'Not implemented yet!!'

    def StopLifePreview(*args):
        raise Exception, 'Not implemented yet!!'

    def ExpReady(self):
        tmp = c_int()
        ret = ac.GetStatus(byref(tmp))
        if not ret == ac.DRV_SUCCESS:
            raise 'Error getting camera status: %s' % ac.errorCodes[ret]

        #print ac.errorCodes[tmp.value]

        #return tmp.value == ac.DRV_IDLE
        first=c_int()
        last=c_int()
        ret = ac.GetNumberNewImages(byref(first), byref(last))
        #print ac.errorCodes[ret]
        #print first
        #print last
        return ret == ac.DRV_SUCCESS 

    def GetBWPicture(*args):
        raise Exception, 'Not implemented yet!!'

    def ExtractColor(self, chSlice, mode):
        #print chSlice
        #pc = chSlice.split('_')[1]
        #ret = ac.GetAcquiredData16(cast(c_void_p(int(pc[6:8]+pc[4:6]+pc[2:4]+pc[0:2],16)), POINTER(c_ushort)), self.GetPicWidth()*self.GetPicHeight())
        #ret = ac.GetAcquiredData16(cast(c_void_p(int(chSlice)), POINTER(c_ushort)), self.GetPicWidth()*self.GetPicHeight())

        ret = ac.GetOldestImage16(cast(c_void_p(int(chSlice)), POINTER(c_ushort)), self.GetPicWidth()*self.GetPicHeight())

        #print self.GetPicWidth()*self.GetPicHeight()
        if not ret == ac.DRV_SUCCESS:
            print 'Error getting image from camera: %s' % ac.errorCodes[ret]

    def CheckCoordinates(*args):
        raise Exception, 'Not implemented yet!!'

    def StopAq(self):
        ac.AbortAcquisition()

    def SetCCDTemp(self, temp):
        if (temp < self.tRange[0]) or (temp > self.tRange[1]):
            raise 'Temperature setpoint out of range ([%d,%d])' % self.tRange

        self.tempSet = temp #temperature setpoint

        ret = ac.SetTemperature(self.tempSet) #fixme so that default T is read in
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting Temperature: %s' % ac.errorCodes[ret]

    def SetEMGain(self, gain):
        if (gain > 150): #150 is arbitrary, but seems like a reasomable enough boundary
            print 'WARNING: EM Gain of %d selected; overuse of high gains can lead to gain register aging' % gain

        self.EMGain = gain
        ret = ac.SetEMCCDGain(self.EMGain)
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting EM Gain: %s' % ac.errorCodes[ret]

    def GetEMGain(self):
        return self.EMGain

    def WaitForExp(self):
        ret = ac.WaitForAcquisition() #block until aquisition is finished
        if not ret == ac.DRV_SUCCESS:
            raise 'Error waiting for acquisition: %s' % ac.errorCodes[ret]

    def Shutdown(self):
        ac.CoolerOFF()
        t = c_int(-100)
        ac.GetTemperature(byref(t))

        while (t.value < -50): #wait fro temp to get above -50
            time.sleep(1)
            ac.GetTemperature(byref(t))

        ac.ShutDown()
        self.initialised = False

    def SpoolOn(self, filename):
        ac.SetAcquisitionMode(5)
        ac.SetSpool(1,2,filename,10)
        ac.StartAcquisition()

    def SpoolOff(self):
        ac.AbortAcquisition()
        ac.SetSpool(0,2,r'D:\spool\spt',10)
        ac.SetAcquisitionMode(1)

    def GetCCDTempSetPoint(self):
        return self.tempSet

    def SetAcquisitionMode(self, aqMode):
        ac.AbortAcquisition()
        ac.SetAcquisitionMode(aqMode)
        self.contMode = not aqMode == 1

    def SetFrameTransfer(self, ftMode):
        self.frameTransferMode = ftMode

        if ftMode:
            ac.SetFrameTransferMode(1)
        else:
            ac.SetFrameTransferMode(0)
            
    def SetVerticalShiftSpeed(self, speedIndex):
        self.VSSpeed = speedIndex
        ret = ac.SetVSSpeed(self.VSSpeed)
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting VS speed: %s' % ac.errorCodes[ret]

        #increase clock voltage to prevent smear if using fastest v.s. speed
        #THIS MAY NEED TWEAKING
        if speedIndex == 0:
            ac.SetVSAmplitude(1)
        else:
            ac.SetVSAmplitude(0)
            
    def SetHorizShiftSpeed(self, speedIndex):
        self.HSSpeed = 0
        ret = ac.SetHSSpeed(0,self.HSSpeed)
        if not ret == ac.DRV_SUCCESS:
            raise 'Error setting HS speed: %s' % ac.errorCodes[ret]
        
    def GetFPS(self):
        return 1.0/self.tKin
    
    def GetNumImsBuffered(self):
        first=c_int()
        last=c_int()

        ret = ac.GetNumberNewImages(byref(first), byref(last))
        #print ac.errorCodes[ret]
        #print first
        #print last
        return last.value - first.value
    
    def _GetBufferSize(self):
        bs = c_int()
        ac.GetSizeOfCircularBuffer(byref(bs))
        self.bs = bs.value

    def GetBufferSize(self):
        return self.bs

    def SetShutter(self, state):
        ac.SetShutter(int(state), 1, 0,0)
        
    def SetBaselineClamp(self, state):
        ac.SetBaselineClamp(int(state))

    def GenStartMetadata(self, mdh):
        self.GetStatus()

        mdh.setEntry('Camera.Name', 'Andor IXon DV97')

        mdh.setEntry('Camera.IntegrationTime', self.tExp)
        mdh.setEntry('Camera.CycleTime', self.tKin)
        mdh.setEntry('Camera.EMGain', self.GetEMGain())

        mdh.setEntry('Camera.ROIPosX', self.GetROIX1())
        mdh.setEntry('Camera.ROIPosY',  self.GetROIY1())
        mdh.setEntry('Camera.ROIWidth', self.GetROIX2() - self.GetROIX1())
        mdh.setEntry('Camera.ROIHeight',  self.GetROIY2() - self.GetROIY1())
        mdh.setEntry('Camera.StartCCDTemp',  self.GetCCDTemp())

        #these should really be read from a configuration file
        #hard code them here until I get around to it
        #current values are at 10Mhz using e.m. amplifier
        mdh.setEntry('Camera.ReadNoise', 109.8)
        mdh.setEntry('Camera.NoiseFactor', 1.41)
        mdh.setEntry('Camera.ElectronsPerCount', 27.32)

        realEMGain = ccdCalibrator.getCalibratedCCDGain(self.GetEMGain(), self.GetCCDTempSetPoint())
        if not realEMGain == None:
            mdh.setEntry('Camera.TrueEMGain', realEMGain)

    def __del__(self):
        if self.initialised:
            self.Shutdown()

