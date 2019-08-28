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

    # Acquisition modes
    MODE_SINGLE_SHOT = 0
    MODE_CONTINUOUS = 1
    MODE_SOFTWARE_TRIGGER = 2
    MODE_HARDWARE_TRIGGER = 3


    def __init__(self, *args, **kwargs):
        """
        Create the camera object. This gets called from the PYMEAcquire init script, which is custom for any given microscope
        and can take whatever arguments are needed for a given camera. The one stipulation is that the camera should
        register itself as providing metadata

        """
        
        #register as a provider of metadata
        #this is important so that the camera settings get recorded
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)
        
    def Init(self):
        """
        Optional intialization function. Also called from the init script. Not really part of 'specification'

        """
        pass

    

    

    @property
    def contMode(self):
        """ return whether the camera is runnint in continuous mode or not. This property (was previously a class member
        variable) is required to allow the calling code to determine whether it needs to restart exposures after processing
        the previous one."""
        return self.GetAcquisitionMode() != self.MODE_SINGLE_SHOT

    def ExpReady(self):
        """
        Checks whether there are any frames waiting in the camera buffers

        Returns
        -------
        exposureReady : bool
            True if there are frames waiting

        """

        raise NotImplementedError('Implemented in derived class.')

    def CamReady(self):
        """
        Returns true if the camera is ready (initialized) not really used for
        anything, but might still be checked.

        Returns
        -------
        bool
            Is the camera ready?
        """
    
        return True
    
    def ExtractColor(self, chSlice, mode):
        """
        Pulls the oldest frame from the camera buffer and copies it into memory we provide. Note that the function
        signature and parameters are a legacy of very old code written for a colour camera with a bayer mask.

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
        raise NotImplementedError('Implemented in derived class.')
        
    def GetSerialNumber(self):
        """
        Get the camera serial number

        Returns
        -------
        serialNum : str
            The camera serial number

        """
        raise NotImplementedError('Should be implemented in derived class.')
    
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

        See Also
        --------
        GetIntegTime
        """
        raise NotImplementedError('Implemented in derived class.')
    
    def GetIntegTime(self):
        """
        Get Camera object integration time.

        Returns
        -------
        float
            The exposure time in s

        See Also
        --------
        SetIntegTime
        """
        raise NotImplementedError('Implemented in derived class.')
    
    def GetCCDWidth(self):
        """
        Returns
        -------
        int
            The sensor width in pixels

        """
        raise NotImplementedError('Implemented in derived class.')

    def GetCCDHeight(self):
        """
        Returns
        -------
        int
            The sensor height in pixels

        """
        raise NotImplementedError('Implemented in derived class.')

    def GetPicWidth(self):
        """
        Returns the width (in pixels) of the currently selected ROI.

        Returns
        -------
        int
            Width of ROI (pixels)
        """
        raise NotImplementedError('Implemented in derived class.')

    def GetPicHeight(self):
        """
        Returns the height (in pixels) of the currently selected ROI
        
        Returns
        -------
        int
            Height of ROI (pixels)
        """
        raise NotImplementedError('Implemented in derived class.')

    
    
    #Binning is not really supported in current software, making these commands mostly superfluous
    #Being able to read out the binning (GetHorizBin, GetVertBin) is however necessary
    #these should definitely be revisited
    def SetHorizBin(*args):
        raise NotImplementedError("Implemented in derived class.")

    def GetHorizBin(*args):
        return 0

    def GetHorzBinValue(*args):
        raise NotImplementedError("Implemented in derived class.")

    def SetVertBin(*args):
        raise NotImplementedError("Implemented in derived class.")

    def GetVertBin(*args):
        return 0
    
    
    # ROI Functions
    def SetROIIndex(self, index):
        """
        Used for early Andor Neo cameras with fixed ROIs. Should not be needed for most cameras
        
        """
        raise NotImplementedError("Implement if needed.")

    def SetROI(self, x1, y1, x2, y2):
        """
        Set the ROI via coordinates (as opposed to via an index).

        NOTE/FIXME: this is somewhat inconsistent over cameras, with some
        cameras using 1-based and some cameras using 0-based indexing.
        Ideally we would convert them all to using zero based indexing and be
        consistent with numpy.

        Most use 1 based (as it's a thin wrapper around the camera API), but we
        should really do something saner here.

        Parameters
        ----------
        x1 : int
            Left x-coordinate
        y1 : int
            Top y-coordinate
        x2 : int
            Right x-coordinate
        y2 : int
            Bottom y-coordinate

        Returns
        -------
        None

        See Also
        --------
        SetROIIndex
        """
        raise NotImplementedError('Implemented in derived class.')

    def GetROIX1(self):
        """
        Gets the position of the leftmost pixel of the ROI.

        Returns
        -------
        int
            Left x-coordinate of ROI.
        """
        raise NotImplementedError('Implemented in derived class.')

    def GetROIX2(self):
        """
        Gets the position of the rightmost pixel of the ROI.

        Returns
        -------
        int
            Right x-coordinate of ROI.
        """
        raise NotImplementedError('Implemented in derived class.')

    def GetROIY1(self):
        """
        Gets the position of the top row of the ROI.

        Returns
        -------
        int
            Top y-coordinate of ROI.
        """
        raise NotImplementedError('Implemented in derived class.')

    def GetROIY2(self):
        """
        Gets the position of the bottom row of the ROI.

        Returns
        -------
        int
            Bottom y-coordinate of ROI.
        """
        raise NotImplementedError('Implemented in derived class.')

    def SetEMGain(self, gain):
        """
        Set the electron-multiplying gain. For EMCCDs this is typically the
        uncalibrated, gain register setting. The calibrated gain is computed
        separately and saved in the metadata as TrueEMGain.

        Parameters
        ----------
        gain : float
            EM gain of Camera object.

        Returns
        -------
        None

        See Also
        --------
        GetEMGain
        """
        pass

    def GetEMGain(self):
        """
        Return electron-multiplying gain register setting. The actual gain will likely be a non-linear function of this
        gain, so EMCCD cameras classes are encouraged to keep calibration data and to write this into the metadata as
        'TrueEMGain'
        
        For non-EMCCD cameras, this should return 1.

        Returns
        -------
        float
            Electron multiplying gain register setting

        See Also
        ----------
        GetTrueEMGain, SetEMGain
        """
        return 1

    def GetElectrTemp(*args):
        """
        Returns the temperature of the internal electronics. Legacy of PCO Sensicam support, which had separate sensors
        for CCD and electronics. Not actually used anywhere critical (might be recorded in metadata), can remove.

        """
        return 25

    def GetCCDTemp(self):
        """
        Gets the sensor temperature.

        Returns
        -------
        float
            The sensor's temperature in degrees Celsius
        """
        #FIXME - this should raise a NotImplementedError
        return self._temp

    def GetCCDTempSetPoint(self):
        """
        Get the target camera temperature. Only currently called in Ixon
        related code, but potentially generally useful.

        Returns
        -------
        float
            Target camera temperature (Celsius)
        """
        raise NotImplementedError('Implemented in derived class.')

    def SetCCDTemp(self, temp):
        """
        Set the target camera temperature.

        Parameters
        ----------
        temp : float
            The target camera temperature (Celsius)

        Returns
        -------
        None
        """
        raise NotImplementedError('Implemented in derived class.')

    def SetShutter(self, mode):
        """
        Set the camera shutter (if available).

        Parameters
        ----------
        mode : bool
            True (1) if open
        """
        pass

    def SetBaselineClamp(self, mode):
        """ Set the camera baseline clamp (EMCCD). Only called from the Ixon settings panel, so not relevant for other
        cameras."""
        pass


    def GetAcquisitionMode(self):
        """
        Get the Camera object readout mode.

        Returns
        -------
        int
            One of self.MODE_CONTINUOUS, self.MODE_SINGLE_SHOT

        See Also
        --------
        SetAcquisitionMode
        """
        raise NotImplementedError('Should be implemented in derived class.')

    def SetAcquisitionMode(self, mode):
        """
        Set the readout mode of the Camera object. PYME currently supports two
        modes: single shot, where the camera takes one image, and then a new
        exposure has to be manually triggered, or continuous / free running,
        where the camera runs as fast as it can until we tell it to stop.

        Parameters
        ----------
        mode : int
            One of self.MODE_CONTINUOUS, self.MODE_SINGLE_SHOT

        Returns
        -------
        None

        See Also
        --------
        GetAcquisitionMode
        """

        raise NotImplementedError('Should be implemented in derived class.')
    
    def SetBurst(self, burstSize):
        """
        Used with Andor Zyla/Neo for burst mode acquisition, can generally ignore for most cameras. Somewhat experimental
        and does not currently have any UI bindings.
        """
        raise NotImplementedError("Implement if needed")
    
    def SetActive(self, active=True):
        """
        Flag the Camera object as active (or inactive) to dictate whether or
        not it writes its metadata.

        Parameters
        ----------
        active : bool
        """

        self.active = bool(active)
    
    def StartExposure(self):
        """
        Starts an acquisition.

        Returns
        -------
        int
            Success (0) or failure (-1) of initialization.
        """
        raise NotImplementedError('Implemented in derived class.')

    def StopAq(self):
        """
        Stops acquiring.

        Returns
        -------
        None
        """
        raise NotImplementedError('Implemented in derived class.')


    def GetNumImsBuffered(self):
        """
        Return the number of images in the buffer.

        Returns
        -------
        int
            Number of images in buffer
        """
        raise NotImplementedError("Implemented in derived class.")

    def GetBufferSize(self):
        """
        Return the total size of the buffer (in images).

        Returns
        -------
        int
            Number of images that can be stored in the buffer.
        """
        raise NotImplementedError("Implemented in derived class.")


    def GetNoiseProperties(self):
        """

        Returns
        -------

        a dictionary with the following entries:

        'ReadNoise' : camera read noise in electrons
        'ElectronsPerCount' : AD conversion factor - how many electrons per camera count
        'NoiseFactor' : excess (multiplicative) noise factor 1.44 for EMCCD, 1 for standard CCD/sCMOS

        and optionally
        'ADOffset' : the dark level (in counts)
        'DefaultEMGain' : a sensible EM gain setting to use for localization recording
        'SaturationThreshold' : the full well capacity (in counts)

        """
        raise NotImplementedError('Implement in derived class')

    def GetStatus(*args):
        """
        Used to poll the camera for status information. Useful for some cameras where it makes sense
        to make one API call to get the temperature, frame rate, num frames buffered etc ... and cache
        the results.

        Parameters
        ----------
        args

        Returns
        -------

        """
        pass
    
    def GetFPS(self):
        """
        Get the camera frame rate in frames per second (float).

        Returns
        -------
        float
            Camera frame rate (frames per second)
        """
        
        #FIXME - this should really raise not-implemented
        return self._frameRate
    
    def GenStartMetadata(self, mdh):
        """ provide metadata"""
        if self.active:
            self.GetStatus()
        
            mdh.setEntry('Camera.Name', 'Andor Neo')
        
            mdh.setEntry('Camera.IntegrationTime', self.GetIntegTime())
            mdh.setEntry('Camera.CycleTime', self.GetIntegTime())
            mdh.setEntry('Camera.EMGain', 1)
        
            mdh.setEntry('Camera.ROIPosX', self.GetROIX1()) #old style, for compatibility
            mdh.setEntry('Camera.ROIPosY', self.GetROIY1())
            mdh.setEntry('Camera.ROIOriginX', self.GetROIX1() - 1) #new 0 based version
            mdh.setEntry('Camera.ROIOriginY', self.GetROIY1() - 1)
            mdh.setEntry('Camera.ROIWidth', self.GetROIX2() - self.GetROIX1())
            mdh.setEntry('Camera.ROIHeight', self.GetROIY2() - self.GetROIY1())
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
        
    def Shutdown(self):
        """Shutdown and clean up the camera"""
        pass
        
    def __del__(self):
        self.Shutdown()
    
        
    #legacy methods, unused
    
    def CheckCoordinates(*args):
        # this could possibly get a reprieve, maybe ofter re-naming. Purpose was to decide if a given
        # ROI was going to be valid
        raise DeprecationWarning("Deprecated.")
    
    def DisplayError(*args):
        """Completely deprecated and never called. Artifact of very old code which had GUI mixed up with camera. Should remove"""
        raise DeprecationWarning("Deprecated.")
    
    def GetBWPicture(*args):
        """Legacy of old code. Not called anywhere, should remove"""
        raise DeprecationWarning("Deprecated.")
    
    def GetNumberChannels(*args):
        """
        Returns the number of colour channels in the Bayer mask. Legacy, deprecated, and not used

        Returns
        -------
        the number of colour channels

        """
        raise DeprecationWarning("Deprecated.")
    
    def SetCOC(*args):
        """Legacy of sensicam support. Hopefully no longer called anywhere"""
        raise DeprecationWarning("Deprecated.")

    def StartLifePreview(*args):
        """Legacy of old code. Not called anywhere, should remove"""
        raise DeprecationWarning("Deprecated.")

    def StopLifePreview(*args):
        """Legacy of old code. Not called anywhere, should remove"""
        raise DeprecationWarning("Deprecated.")

