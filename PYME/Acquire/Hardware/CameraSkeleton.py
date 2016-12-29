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


import numpy as np
import ctypes

from PYME.IO import MetaDataHandler
from PYME.Acquire import eventLog

class SkeletonCamera(object):
    # Frame format - PYME previously supported frames in a custom format, but numpy_frames should always be true for current code
    numpy_frames=1 #Frames are delivered as numpy arrays.

    #these two defines are used reasonably extensively in other parts of the code which call the SetAquisitionMode function
    MODE_CONTINUOUS = 1
    MODE_SINGLE_SHOT = 0
    
    def __init__(self, *args, **kwargs):
        """
        Create the camera object. This gets called from the PYMEAcquire init script, which is custom for any given microscope
        and can take whatever arguments are needed for a given camera. The one stipulation is that the camera should
        register itself as providing metadata
    
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
        self.SimplePreAmpGainControl = ATEnum()
        self.FullAOIControl = ATBool()

        """
        
        #register as a provider of metadata
        #this is important so that the camera settings get recorded
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)
        
    def Init(self):
        """
        Optional intialization function. Also called from the init script. Not really part of 'specification'

        """

    def SetAcquisitionMode(self, mode):
        """ Set the readout mode of the camera. PYME currently supports two modes, single shot, where the camera takes one
        image, and then a new exposure has to be manually triggered, or continuous / free running, where the camera runs
        as fast as it can until we tell it to stop.

        Parameters
        ==========
        mode : int
            One of self.MODE_CONTINUOUS, self.MODE_SINGLE_SHOT

        """

    @property
    def contMode(self):
        """ return whether the camera is runnint in continuous mode or not. This property (was previously a class member
        variable) is required to allow the calling code to determine whether it needs to restart exposures after processing
        the previous one."""
        return self.CycleMode.getString() == u'Continuous'
        
    #PYME Camera interface functions - make this look like the other cameras
    def ExpReady(self):
        """
        Checks whether there are any frames waiting in the camera buffers

        Returns
        -------
        exposureReady : bool
            True if there are frames waiting

        """
        
        return not self.fullBuffers.empty()
        
    def ExtractColor(self, chSlice, mode):
        """
        Pulls the oldest frame from the camera buffer and copies it into memory we provide. Note that the function
        signature and parameters are a legacy of very old code written for a colour camera with a bayer mask.
            self.SetEMGain(0) # note that this sets 'high dynamic range' mode
        except:
            print "error setting gain mode"
            pass
        # spurious noise filter off by default
        try:
            self.SpuriousNoiseFilter.setValue(0) # this will also fail with the SimCams
        except:
            print "error disabling spurios noise filter"
            pass

        # Static Blemish Correction off by default
        try:
            self.StaticBlemishCorrection.setValue(0) # this will also fail with the SimCams
        except:
            print "error disabling Static Blemish Correction"
            pass
        
        # Zyla does not like a temperature to be set
        if self.TemperatureControl.isImplemented() and self.TemperatureControl.isWritable():
        # test if we have only fixed ROIs
        self._fixed_ROIs = not self.FullAOIControl.isImplemented() or not self.FullAOIControl.getValue()
        self.noiseProps = self.NoiseProperties[self.GetGainMode()] # gain mode has been set via EMGain above

        Parameters
        ----------
        chSlice : numpy ndarray
            The array we want to put the data into
        mode : int, ignored
            Previously specified how to deal with the Bayer mask.

        Returns
        -------
        None

        """
        #grab our buffer from the full buffers list
        buf = self.fullBuffers.get()
        self.nFull -= 1

        ctypes.cdll.msvcrt.memcpy(chSlice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), chSlice.nbytes)
        
        #recycle buffer
        self._queueBuffer(buf)
        
    def GetSerialNumber(self):
        """

        Returns
        -------
        serialNum : str
            The camera serial number

        """
    
    def SetIntegTime(self, iTime):
        """
        Sets the exposure time in s. Currently assumes that we will want to go as fast as possible at this exposure time
        and also sets the frame rate to match.

        Parameters
        ----------
        iTime : float
            Exposure time in s

        Returns
        -------
        None

        """

        
    def GetIntegTime(self):
        """

        Returns
        -------

        The exposure time in s

        """

    
    def GetCCDWidth(self):
        """
        Returns
        -------
        The CCD width in pixels

        """

    def GetCCDHeight(self):
        """
        Returns
        -------
        The CCD height in pixels

        """


        # limit width to multiple of 10 - Zyla specific!!!
        # NOTE: we are not sure why only this size works
        if True:
        else:
            return 10*int(self.SensorWidth.getValue()/10)

    #Binning is not really supported in current software, making these commands mostly superfluous
    #Being able to read out the binning (GetHorizBin, GetVertBin) is however necessary
    #these should definitely be revisited
    def SetHorizBin(*args): 
        raise Exception('Not implemented yet!!')
    def GetHorizBin(*args):
        return 0

    def GetHorzBinValue(*args): 
        raise Exception('Not implemented yet!!')
    def SetVertBin(*args): 
        raise Exception('Not implemented yet!!')
    def GetVertBin(*args):
        return 0



    def GetNumberChannels(*args):
        """
        Returns the number of colour channels in the Bayer mask. Legacy, deprecated, and not used

        Returns
        -------
        the number of colour channels

        """
        raise Exception('Not implemented yet!!')


    def GetElectrTemp(*args):
        """
        Returns the temperature of the internal electronics. Legacy of PCO Sensicam support, which had separate sensors
        for CCD and electronics. Not actually used anywhere critical (might be recorded in metadata), can remove.

        """
        return 25
        
    def GetCCDTemp(self):
        """ Returns the CCD temperature"""

        #for some reason querying the temperature takes a lot of time - do it less often
        #return self.SensorTemperature.getValue()
        
        return self._temp
    

    def CamReady(*args):
        """ Returns true if the camera is ready (initialized) not really used for anything, but might still be checked"""
        return True
    
    def GetPicWidth(self):
        """ Returns the width (in pixels) of the currently selected ROI"""
        return self.AOIWidth.getValue()

    def GetPicHeight(self):
        """ Returns the height (in pixels) of the currently selected ROI"""
        return self.AOIHeight.getValue()
        
    def SetROIIndex(self, index):
        """ Legacy code for old Andor NEO cameras which only supported certain fixed ROIs. Should not be essential / can remove"""

        #print 'in SetROIIndex'
    def ROIsAreFixed(self):
        return self._fixed_ROIs

    def SetROI(self, x1, y1, x2, y2):
        """ Set the ROI.

        Parameters
        ==========

        co-ordinates of the 4 corners.

        NOTE/FIXME: this is somewhat inconsistent over cameras, with some cameras using 1-based and some cameras using
        0 based indexing. Ideally we would convert them all to using zero based indexing and be consistent with numpy.

        Most use 1 based (as it's a thin wrapper around the camera API), but we should really do something saner here
        """
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
            print "setting: ", x1,y1,w10,h
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
        """ gets the position of the leftmost pixel of the ROI"""
        return self.AOILeft.getValue()
        
    def GetROIX2(self):
        """ gets the position of the rightmost pixel of the ROI"""
        return self.AOILeft.getValue() + self.AOIWidth.getValue()
        
    def GetROIY1(self):
        """ gets the position of the top row of the ROI"""
        return self.AOITop.getValue()
        
    def GetROIY2(self):
        """ gets the position of the bottom row of the ROI"""
        return self.AOITop.getValue() + self.AOIHeight.getValue()

    ###
    # NOTE: it might make sense to replace the GetROIX1 etc ... commands with one GetROI command
    


    def DisplayError(*args):
        """Completely deprecated and never called. Artifact of very old code which had GUI mixed up with camera. Should remove"""
        pass


    def Shutdown(self):
        """Shutdown and clean up the camera"""

    def GetStatus(*args):
        """Called by the GUI, optionally returns a status string. This stub needs to be here"""
        pass
    
    def SetCOC(*args):
        """Legacy of sensicam support. Called by other parts of the program, so stub currently needed. Should remove"""
        pass

    def StartExposure(self):
        """ Starts an acquisition"""
        self.tKin = 1.0 / self._frameRate

        return 0
        
    def StopAq(self):
        """Stops acquiring"""
        

    def StartLifePreview(*args):
        """Legacy of old code. Not called anywhere, should remove"""
        raise Exception('Not implemented yet!!')
    def StopLifePreview(*args):
        """Legacy of old code. Not called anywhere, should remove"""
        raise Exception('Not implemented yet!!')

    def GetBWPicture(*args):
        """Legacy of old code. Not called anywhere, should remove"""
        raise Exception('Not implemented yet!!')
    
    def CheckCoordinates(*args):
        """Legacy of old code. Not called anywhere, should remove"""
        raise Exception('Not implemented yet!!')

    #new fcns for Andor compatibility
    def GetNumImsBuffered(self):
        """ Return the number of images in the buffer """
        return self.nFull
    
    def GetBufferSize(self):
        """ Return the total size of the buffer (in images) """
        return self.nBuffers
        
    def SetActive(self, active=True):
        """flag the camera as active (or inactive) to dictate whether it writes it's metadata or not"""
        self.active = active

    def GenStartMetadata(self, mdh):
        """ provide metadata"""
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
        """ Return the current EM Gain. Can be called on non-EMCCD cameras, so needs to be here"""

    def GetCCDTempSetPoint(self):
        """ Get the target CCD temperature. Only currently called in Ixon related code, but potentially generally useful """
        return self.TargetSensorTemperature.getValue()

    def SetCCDTemp(self, temp):
        """ set the target ccd temperature"""
        self.TargetSensorTemperature.setValue(temp)
        #pass

    def SetEMGain(self, gain):
        """ set the em gain. For emccds this is typically the uncalibrated, gain register setting. The calibrated gain
         is computed separately and saved in the metadata as RealEMGain.  Note that this can be called on non-emccd
         cameras, so stub is required"""
            self.SetGainMode('low noise')
        else:
            print 'high dynamic range mode'
            self.SetGainMode('high dynamic range')


    def SetBurst(self, burstSize):
        """ Some support for burst mode on the Neo and Zyla. Is not called from the GUI, and can be considered experimental
        and non-essential"""

    def SetShutter(self, mode):
        """ Set the camera shutter (if available)

        Parameters
        ==========
        mode : bool
            True (1) if open
        """
        pass

    def SetBaselineClamp(self, mode):
        """ Set the camera baseline clamp (EMCCD). Only called from the Ixon settings panel, so not relevant for other
        cameras."""
        pass
    
    def GetFPS(self):
        """ get the camera frame rate in frames per second (float)"""

    def __getattr__(self, name):
        if name in self.noiseProps.keys():
            return self.noiseProps[name]
        else:  raise AttributeError, name  # <<< DON'T FORGET THIS LINE !!




    def __del__(self):
        self.Shutdown()
        #self.compT.kill = True
        self.StaticBlemishCorrection = ATBool()
        self.active = True # we need that for metadata initialisation I believe, not sure if best to do here or elsewhere

