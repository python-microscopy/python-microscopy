# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:13:16 2011

@author: dbad004
"""

from SDK3Cam import *

class AndorBase(SDK3Camera):
    def __init__(self, camNum):
        #define properties
        CameraAcquiring = ATBool()
        SensorCooling = ATBool()
        
        AcquisitionStart = ATCommand()
        AcquisitionStop = ATCommand()
        
        CycleMode = ATEnum()
        ElectronicShutteringMode = ATEnum()
        FanSpeed = ATEnum()
        PreAmpGainChannel = ATEnum()
        PixelEncoding = ATEnum()
        PixelReadoutRate = ATEnum()
        PreAmpGain = ATEnum()
        PreAmpGainSelector = ATEnum()
        TriggerMode = ATEnum()
        
        AOIHeight = ATInt()
        AOILeft = ATInt()
        AOITop = ATInt()
        AOIWidth = ATInt()
        FrameCount = ATInt()
        ImageSizeBytes = ATInt()
        SensorHeight = ATInt()
        SensorWidth = ATInt()
        
        CameraModel = ATString()
        SerialNumber = ATString()
        
        ExposureTime = ATFloat()
        FrameRate = ATFloat()
        SensorTemperature = ATFloat()
        TargetSensorTemperature = ATFloat()
        
        super().__init__(self, camNum)
        
class AndorNeo(AndorBase):
    def __init__(self, camNum):
        #define properties
        Overlap = ATBool()
        SpuriousNoiseFilter = ATBool()
        
        CameraDump = ATCommand()
        SoftwareTrigger = ATCommand()
        
        TemperatureControl = ATEnum()
        TemperatureStatus = ATEnum()
        PreAmpGainControl = ATEnum()
        BitDepth = ATEnum()
        
        ActualExposureTime = ATFloat()
        BurstRate = ATFloat()
        ReadoutTime = ATFloat()
        
        AccumulateCount = ATInt()
        BaselineLevel = ATInt()
        BurstCount = ATInt()
        LUTIndex = ATInt()
        LUTValue = ATInt()
        
        ControllerID = ATString()
        FirmwareVersion = ATString()
        
        super().__init__(self, camNum)
        
class AndorSim(AndorBase):
    def __init__(self, camNum):
        #define properties
        SynchronousTriggering = ATBool()
        
        PixelCorrection = ATEnum()
        TriggerSelector = ATEnum()
        TriggerSource = ATEnum()
        
        PixelHeight = ATFloat()
        PixelWidth = ATFloat()
        
        AOIHBin = ATInt()
        AOIVbin = ATInt()
        
        super().__init__(self, camNum)
        
        
        
        