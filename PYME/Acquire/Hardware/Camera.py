#!/usr/bin/python

###############
# Camera.py
#
# Created: 12 September 2017
# Author : Z Marin
#
# Based on: AndorZyla.py, AndorIXon.py
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

from threading import Lock

from PYME.IO import MetaDataHandler
import numpy as np
import logging
logger = logging.getLogger(__name__)

def check_mapexists(mdh, type='dark', fill=True):
    import os
    from PYME.IO import clusterIO
    from PYME.IO.FileUtils import nameUtils
    from PYME.Analysis.gen_sCMOS_maps import map_filename

    if type == 'dark':
        id = 'Camera.DarkMapID'
    elif type == 'variance':
        id = 'Camera.VarianceMapID'
    elif type == 'flatfield':
        id = 'Camera.FlatfieldMapID'
    else:
        raise RuntimeError('unknown map type %s' % type)

    mapfn = map_filename(mdh, type)

    # find and record calibration paths
    local_path = os.path.join(nameUtils.getCalibrationDir(mdh['Camera.SerialNumber']), mapfn)
    cluster_path = 'CALIBRATION/%s/%s' % (mdh['Camera.SerialNumber'], mapfn)

    if clusterIO.exists(cluster_path):
        c_path = 'PYME-CLUSTER://%s/%s' % (clusterIO.local_serverfilter, cluster_path)
        if fill:
            mdh[id] = c_path
        return c_path
    elif os.path.exists(local_path):
        if fill:
            mdh[id] = local_path
        return local_path
    else:
        return None


class CameraMapMixin(object):
    def _map_cache_key(self, mdh, map_type):
        if map_type == 'flatfield':
            return (map_type, mdh['Camera.SerialNumber'])
        else:
            return (map_type, mdh['Camera.SerialNumber'], mdh['Camera.IntegrationTime'])
        
    def _fill_camera_map_id(self, mdh, mdh_key, map_type):
        #create cache if not already present
        if not hasattr(self, '_camera_map_cache'):
            self._camera_map_cache = {}
            
        cache_key = self._map_cache_key(mdh, map_type)
        try:
            map_fn = self._camera_map_cache[cache_key]
        except KeyError:
            map_fn = check_mapexists(mdh, map_type, fill=False)
            self._camera_map_cache[cache_key] = map_fn
        
        if not map_fn is None:
            mdh[mdh_key] = map_fn
    
    def fill_camera_map_metadata(self, mdh):
        self._fill_camera_map_id(mdh, 'Camera.DarkMapID', map_type='dark')
        self._fill_camera_map_id(mdh, 'Camera.VarianceMapID', map_type='variance')
        self._fill_camera_map_id(mdh, 'Camera.FlatfieldMapID', map_type='flatfield')


class Camera(object):
    # Frame format - PYME previously supported frames in a custom format, but numpy_frames should always be true for current code
    numpy_frames = 1 #Frames are delivered as numpy arrays.

    # Acquisition modes
    MODE_SINGLE_SHOT = 0
    MODE_CONTINUOUS = 1
    MODE_SOFTWARE_TRIGGER = 2
    MODE_HARDWARE_TRIGGER = 3


    def __init__(self, *args, **kwargs):
        """
        Create a camera object. This gets called from the PYMEAcquire init
        script, which is custom for any given microscope and can take
        whatever arguments are needed for a given camera.

        .. note:: The one stipulation is that the Camera should register itself
                  as providing metadata.

        Parameters
        ----------
        args :
            Optional arguments, usually instantiated in inherited camera.
        kwargs :
            Optional dictionary of arguments, usually instantiated in
            inherited camera.

        """

        self.active = True  # Should the camera write its metadata?

        # Register as a provider of metadata (record camera settings)
        # this is important so that the camera settings get recorded
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)

    def Init(self):
        """
        Optional intialization function. Also called from the init script.
        Not really part of 'specification'
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

    def GetName(self):
        """ Camera name.
        
        FIXME - Do we need this???
        
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
        chSlice : `~numpy.ndarray`
            The array we want to put the data into
        mode : int, ignored
            Previously specified how to deal with the Bayer mask.

        Returns
        -------
        None
        """
        raise NotImplementedError('Implemented in derived class.')

    def GetHeadModel(self):
        """
        Get the the camera head model name.

        Returns
        -------
        str
            camera model namet
        """
        raise NotImplementedError('Should be implemented in derived class.')

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


    def GetCycleTime(self):
        """
        Get camera cycle time (1/fps) in seconds (float)

        Returns
        -------
        float
            Camera cycle time (seconds)
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
    def SetHorizontalBin(self, value):
        raise NotImplementedError("Implemented in derived class.")

    def GetHorizontalBin(self):
        return 1

    def SetVerticalBin(self, value):
        raise NotImplementedError("Implemented in derived class.")

    def GetVerticalBin(self):
        return 1
    
    def GetSupportedBinnings(self):
        """
        returns a list of tuples [(binx, biny), ...] of supported binning configurations
        
        """
        
        return [(1,1)]
    
    
    # ROI Functions
    def SetROIIndex(self, index):
        """
        Used for early Andor Neo cameras with fixed ROIs. Should not be needed for most cameras
        
        """
        raise NotImplementedError("Implement if needed.")

    def SetROI(self, x1, y1, x2, y2):
        """
        Set the ROI via coordinates (as opposed to via an index).

        NOTE/FIXME: ROI handling is currently somewhat inconsistent between cameras, as a result of underlying differences
        in the Camera APIs (some of which use zero based pixel indices and some of which use 1-based indices).
        
        With the possible exception of the Hamamatsu Orca Flash support, the numpy convention is used, whereby:
         
        x1, y1 are the zero based indices of the top-left pixel in the ROI and x2, y2 are the coordinates of the
        bottom right corner of the ROI. It is important to note that, as with the python/numpy slice notation the row
        and column with (zero based) coordinates y2 and x2 respectively are **excluded** from the ROI. IE an ROI [x1, y1, x2, y2]
        goes from pixel index x1 up to, but not including pixel index x2 (this lets us define the ROI width as (x2-x1)).
        
        Due to the fact that they map directly to API functions related to the Andor cameras, GetROIX1 and GetROIY1,
        return indices which are 1-based and GetROIX2 and GetROIY2 match x2 and y2 as provided to SetROI. This is not
        strictly followed in all cameras with the orca flash breaking convention. Rather than maintaining an inconsistent
        indexing convention between the GetROIX...() functions and SetROI, these have been etc have been deprecated for
        a single GetROI() function which uses the same conventions as SetROI
        
        FIXME: How does the Orca handle x2, y2 ???
        
        Parameters
        ----------
        x1 : int
            Left x-coordinate, zero-indexed
        y1 : int
            Top y-coordinate, zero-indexed
        x2 : int
            Right x-coordinate, (excluded from ROI)
        y2 : int
            Bottom y-coordinate, (excluded from ROI)

        Returns
        -------
        None


        """
        raise NotImplementedError('Implemented in derived class.')
    
    def GetROI(self):
        """
        
        Returns
        -------
        
            The ROI, [x1, y1, x2, y2] in the numpy convention used by SetROI

        """
        raise NotImplementedError('Implemented in derived class.')

    def GetROIX1(self):
        """
        Gets the position of the leftmost pixel of the ROI. [Deprecated]

        Returns
        -------
        int
            Left x-coordinate of ROI. One based indexing

        Notes
        -----
        This is broken for the Orca
        """
        logger.warning("Deprecated - use GetROI() instead")
        return self.GetROI()[0] + 1
        

    def GetROIX2(self):
        """
        Gets the position of the rightmost pixel of the ROI. [Deprecated]

        Returns
        -------
        int
            Right x-coordinate of ROI. One base indexing

        
        """
        logger.warning("Deprecated - use GetROI() instead")
        return self.GetROI()[2] + 1

    def GetROIY1(self):
        """
        Gets the position of the top row of the ROI. [Deprecated]

        Returns
        -------
        int
            Top y-coordinate of ROI. One based indexing

        
        """
        logger.warning("Deprecated - use GetROI() instead")
        return self.GetROI()[1]

    def GetROIY2(self):
        """
        Gets the position of the bottom row of the ROI. [Deprecated]

        Returns
        -------
        int
            Bottom y-coordinate of ROI. One based indexing

        
        """
        logger.warning("Deprecated - use GetROI() instead")
        return self.GetROI()[3]

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

    def GetTrueEMGain(self):
        """
        # FIXME - what is this doing here???????
        
        Return true electron-multiplying gain of Camera object.

        Returns
        -------
        float
            Camera object adjusted true gain.

        See Also
        ----------
        GetEMGain
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
        raise NotImplementedError('Implement in derived class.')

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


    @property
    def noise_properties(self):
        """

                Returns
                -------

                a dictionary with the following entries:

                'ReadNoise' : camera read noise as a standard deviation in units of photoelectrons (e-)
                'ElectronsPerCount' : AD conversion factor - how many electrons per ADU
                'NoiseFactor' : excess (multiplicative) noise factor 1.44 for EMCCD, 1 for standard CCD/sCMOS. See
                    doi: 10.1109/TED.2003.813462

                and optionally
                'ADOffset' : the dark level (in ADU)
                'DefaultEMGain' : a sensible EM gain setting to use for localization recording
                'SaturationThreshold' : the full well capacity (in ADU)

                """
        
        raise AttributeError('Implement in derived class')
        

    def GetStatus(self):
        """
        Used to poll the camera for status information. Useful for some cameras where it makes sense
        to make one API call to get the temperature, frame rate, num frames buffered etc ... and cache
        the results. For most cameras can be safely ignored.

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

        raise NotImplementedError('Implement in derived class')
    
    def GenStartMetadata(self, mdh):
        """
        FIXME - should this be in the base class ???
        
        Create Camera metadata. This ensures the Camera's settings get
        recorded.

        Parameters
        ----------
        mdh : MetaDataHandler
            MetaDataHandler object for Camera.

        Returns
        -------
        None
        """

        if self.active:
            # Set Camera object metadata here with calls to mdh.setEntry
            self.GetStatus()

            # Personal identification
            mdh.setEntry('Camera.Name', self.GetName())
            mdh.setEntry('Camera.Model', self.GetHeadModel())
            mdh.setEntry('Camera.SerialNumber', self.GetSerialNumber())

            # Time
            mdh.setEntry('Camera.IntegrationTime', self.GetIntegTime())
            mdh.setEntry('Camera.CycleTime', self.GetCycleTime())

            # Gain
            mdh.setEntry('Camera.EMGain', self.GetEMGain())
            mdh.setEntry('Camera.TrueEMGain', self.GetTrueEMGain())

            # Noise
            noiseProps = self.noise_properties
            mdh.setEntry('Camera.ReadNoise', noiseProps['ReadNoise'])
            mdh.setEntry('Camera.NoiseFactor', noiseProps.get('NoiseFactor', 1))
            mdh.setEntry('Camera.ElectronsPerCount', noiseProps['ElectronsPerCount'])

            # Temp
            mdh.setEntry('Camera.StartCCDTemp', self.GetCCDTemp())

            # Chip size
            mdh.setEntry('Camera.SensorWidth', self.GetCCDWidth())
            mdh.setEntry('Camera.SensorHeight', self.GetCCDHeight())

            x1, y1, x2, y2 = self.GetROI()
            mdh.setEntry('Camera.ROIOriginX', x1)
            mdh.setEntry('Camera.ROIOriginY', y1)
            mdh.setEntry('Camera.ROIWidth', x2 - x1)
            mdh.setEntry('Camera.ROIHeight', y2 - y1)
            

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


# FIXME - move out of this file
class MultiviewCameraMixin(object):
        def __init__(self, multiview_info, default_roi, camera_class):
            """
            Used principally for cutting horizontally spaced ROIs out of a vertical band of the sCMOS chip, where there
            is dark space between the images and we want to avoid saving and transmitting this dark data.

            This class supports the standard full chip and cropped ROI modes, as well as a new multiview mode. The
            multiview mode makes the frames appear to the outside world as though they are just active multiview views
            concatenated horizontally.

            To implement a multiview camera, simply inherit both from the specific camera class you want as well as this
            class, e.g.
            class MultiviewCoolCamera(MultiviewCamera, CoolCamera)
                def __init__(self, whatever_cool_cam_needs, multiview_info)
                    CoolCamera.__init__(self, whatever_cool_cam_needs)
                    default_roi = dict(xi=0, xf=2048, yi=0, yf=2048)
                    MultiviewCamera.__init__(self, multiview_info, default_roi, CoolCamera)

            and that's it. Then in your init script, simply define your multiview_info dict and initialize your
            MultiviewCoolCamera class.

            Parameters
            ----------
            multiview_info : dict
                Information about how to crop the image. Can either be a dictionary, or something which behaves like a
                dictionary (e.g. a MetaDataHandler).
            default_roi: dict
                contains the default ROI of the camera. This should just be the full chip, and should be defined in the
            camera_class: class
                This should be the class of the camera you want to inherit from. The MultiviewCamera class needs to know
                this so we can clobber the inherited ExtractColor method and still be able to pull raw frames from the
                camera using camera_class.ExtractColor

            Notes
            -----
            For now, the 0th multiview ROI should be the upper-left most multiview ROI, in order to properly spoof the
            position to match up with the stage. See PYME.IO.MetaDataHandler.get_camera_roi_origin.
            """
            self.camera_class = camera_class
            self.multiview_info = multiview_info
            self._channel_color = multiview_info['Multiview.ChannelColor']

            self.n_views = multiview_info['Multiview.NumROIs']
            self.view_origins = [multiview_info['Multiview.ROI%dOrigin' % i] for i in range(self.n_views)]
            self.size_x, self.size_y = multiview_info['Multiview.DefaultROISize']
            self.multiview_roi_size_options =  multiview_info['Multiview.ROISizeOptions']

            self.view_centers = [(ox + int(0.5*self.size_x), oy + int(0.5*self.size_y)) for ox, oy in self.view_origins]

            self.multiview_enabled = False
            self.active_views = []

            # set default width and height to return to when multiview is disabled
            self._default_chip_width = default_roi['xf'] - default_roi['xi']
            self._default_chip_height = default_roi['yf'] - default_roi['yi']
            self.default_chip_roi = default_roi
            self._current_pic_width = self._default_chip_width
            self._current_pic_height = self._default_chip_height

        def ChangeMultiviewROISize(self, x_size, y_size):
            """
            Changes the ROI size of the views. Currently they are all the same size
            Parameters
            ----------
            x_size: int
                first dimension size
            y_size: int
                second dimension size

            Returns
            -------
            None
            """
            # shift the origins
            self.view_origins = [(xc - int(0.5*x_size), yc - int(0.5*y_size)) for xc, yc in self.view_centers]
            # store the new sizes
            self.size_x, self.size_y = x_size, y_size

            if self.multiview_enabled:
                # re-set the slices for frame cropping
                self.enable_multiview(self.active_views)


        def GetPicWidth(self):
            """
            This clobbers the inherited self.camera_class GetPicWidth method so that the outside world (FrameWrangler)
            only allocates memory/thinks our camera frames are the final concatenated multiview width.
            Returns
            -------
            pic_width: int
                width of the concatenated multiview frame
            """
            if self.multiview_enabled:
                return self._current_pic_width
            else:
                return self.camera_class.GetPicWidth(self)

        def GetPicHeight(self):
            """
            This clobbers the inherited self.camera_class GetPicHeight method so that the outside world (FrameWrangler)
            only allocates memory/thinks our camera frames are the final multiview frame height.
            Returns
            -------
            pic_height: int
                height of the multiview frame
            """
            if self.multiview_enabled:
                return self._current_pic_height
            else:
                return self.camera_class.GetPicHeight(self)
        
        def set_active_views(self, views):
            if len(views) == 0:
                self.disable_multiview()
            elif sorted(views) == self.active_views:
                pass
            else:
                self.enable_multiview(views)
            

        def enable_multiview(self, views):
            """

            Parameters
            ----------
            views: list
                views to activate. Should be integers which can be used to index self.multiview_info

            Returns
            -------

            Notes
            -----
            FrameWrangler must be stopped before this function is called, and "prepared" afterwards before being started
            again. This is not special to this function, but rather anytime SetROI gets called.

            """
            views = sorted(list(views))  # tuple(int) isn't iterable, make sure we avoid it
            # set the camera FOV to be just large enough so we do most of the cropping where it is already optimized
            self.x_origins, self.y_origins = zip(*[self.view_origins[view] for view in views])
            chip_x_min, chip_x_max = min(self.x_origins), max(self.x_origins)
            chip_y_min, chip_y_max = min(self.y_origins), max(self.y_origins)

            chip_width = chip_x_max + self.size_x - chip_x_min
            chip_height = chip_y_max + self.size_y - chip_y_min

            self.chip_roi = [chip_x_min, chip_y_min, chip_x_min + chip_width, chip_y_min + chip_height]
            logger.debug('setting chip ROI')
            self.SetROI(*self.chip_roi)
            actual = self.GetROI()
            try:
                assert actual == tuple(self.chip_roi)
            except AssertionError:
                raise(AssertionError('Error setting camera ROI. Check that ROI is feasible for camera, target: %s, actual: %s'
                             % (tuple(self.chip_roi), actual)))

            # hold an array for temporarily writing the roughly cropped chip
            self.chip_data = np.empty((chip_width, chip_height), dtype='uint16', order='F')

            # precalculate slices for each view
            self.view_slices, self.output_slices = [], []
            for x_ind, view in enumerate(views):
                ox, oy = self.view_origins[view]
                # calculate the offset from the chip origin
                oxp, oyp = ox - chip_x_min, oy - chip_y_min
                # calculate the slices to pull out of roi on chip
                self.view_slices.append(np.s_[oxp:oxp + self.size_x, oyp: oyp + self.size_y])
                # calculate slices to write into out array
                self.output_slices.append(np.s_[self.size_x * x_ind:self.size_x * (x_ind + 1), 0:self.size_y])

            # update our apparent height and widths, concatenating along 'x' or the 0th dim
            self._current_pic_width = len(views) * self.size_x
            self._current_pic_height = self.size_y
            # tell the world what we've accomplished here today
            self.multiview_enabled = True
            self.active_views = views

        def disable_multiview(self):
            """
            Disables multiview mode and returns camera to the default ROI (e.g. full chip)
            Returns
            -------

            """
            self.multiview_enabled = False
            self.active_views = []
            self.SetROI(self.default_chip_roi['xi'], self.default_chip_roi['yi'],
                        self.default_chip_roi['xf'], self.default_chip_roi['yf'],)


        def ExtractColor(self, output_frame, mode):
            """
            Override camera get-frame function, but with multiview cropping.

            Parameters
            ----------
            output_frame: np.array
                array sized for the final multiview frame
            mode: int
                camera acquisition mode.

            Returns
            -------

            """
            if self.multiview_enabled:
                # logger.debug('pulling frame')
                # pull data off the roughly cropped frame
                self.camera_class.ExtractColor(self, self.chip_data, mode)
                # extract the multiview frames from the cropped chip into our output
                for out_slice, view_slice in zip(self.output_slices, self.view_slices):
                    output_frame[out_slice] = self.chip_data[view_slice]

            else:
                # skip extra cropping, extract the full chip directly into the output frame
                self.camera_class.ExtractColor(self, output_frame, mode)

        def GenStartMetadata(self, mdh):
            """
            Light shim to record multiview metadata, when appropriate

            Parameters
            ----------
            mdh : MetaDataHandler
                MetaDataHandler object for Camera.

            Returns
            -------
            None
            """
            self.camera_class.GenStartMetadata(self, mdh)
            # add in multiview info
            if self.multiview_enabled:
                mdh.setEntry('Multiview.NumROIs', self.n_views)
                mdh.setEntry('Multiview.ROISize', [self.size_x, self.size_y])
                mdh.setEntry('Multiview.ChannelColor', self._channel_color)
                mdh.setEntry('Multiview.ActiveViews', self.active_views)
                for ind in range(self.n_views):
                    mdh.setEntry('Multiview.ROI%dOrigin' % ind, self.view_origins[ind])

        def register_state_handlers(self, state_manager):
            """ Allow key multiview settings to be updated easily through
            the microscope state handler

            Parameters
            ----------
            state_manager : PYME.Acquire.microscope.State
            """
            logger.debug('registering multiview camera state handlers')
            
            state_manager.registerHandler('Multiview.ActiveViews', 
                                          lambda : self.active_views, 
                                          self.set_active_views, True)
            state_manager.registerHandler('Multiview.ROISize', 
                                          lambda : [self.size_x, self.size_y],
                                          lambda p : self.ChangeMultiviewROISize(p[0], p[1]),
                                          True)
