import imp
from PYME.Acquire.Hardware import Camera
from PYME.Acquire.Hardware.Simulator import fakeCam
import numpy as np
import logging
logger = logging.getLogger(__name__)

class MultiviewWrapper(object):
    def __init__(self, base_camera, multiview_info, default_roi=None):
        """
        Used principally for cutting horizontally spaced ROIs out of a vertical band of the sCMOS chip, where there
        is dark space between the images and we want to avoid saving and transmitting this dark data.

        This class supports the standard full chip and cropped ROI modes, as well as a new multiview mode. The
        multiview mode makes the frames appear to the outside world as though they are just active multiview views
        concatenated horizontally.

        The class wraps around an existing camera class, redefining the methods associated
        with getting and setting ROIs and image data, other camera methods are passed through to the
        base camera class.

        Usage is as follows:

        ```
        #initialise base camera as normal
        base_camera = SomeCameraClass(...)
        # wrap it
        mvcam = MultiviewWrapper(base_camera, multiview_info, default_roi)
        ```

        Parameters
        ----------
        base_camera: Camera.Camera (or subclass) instance
            the object representing the actual physical camera. 
        multiview_info : dict
            Information about how to crop the image. Can either be a dictionary, or something which behaves like a
            dictionary (e.g. a MetaDataHandler).
        default_roi: dict, optional
            contains the default ROI of the camera. This will usually be automatically read from the base camera
        

        Notes
        -----
        For now, the 0th multiview ROI should be the upper-left most multiview ROI, in order to properly spoof the
        position to match up with the stage. See PYME.IO.MetaDataHandler.get_camera_roi_origin.
        """
        self._camera = base_camera # type: Camera.Camera

        if default_roi is None:
            
            #full field of base camera
            default_roi = {
                'xi' : 0,
                'yi' : 0,
                'xf' : self._camera.GetCCDWidth(),
                'yf' : self._camera.GetCCDHeight()
            }

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

        #hack for simulator TODO - move this somewhere more sensible
        if isinstance(self._camera, fakeCam.FakeCamera):
            vx = self._camera.XVals[1] - self._camera.XVals[0]
            self._camera._chan_x_offsets = [vx*x0 for x0, y0 in self.view_origins]

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
            return self._camera.GetPicWidth()

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
            return self._camera.GetPicHeight()
    
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
        self._camera.SetROI(*self.chip_roi)
        actual = self._camera.GetROI()
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
        self._camera.SetROI(self.default_chip_roi['xi'], self.default_chip_roi['yi'],
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
            self._camera.ExtractColor(self.chip_data, mode)
            # extract the multiview frames from the cropped chip into our output
            for out_slice, view_slice in zip(self.output_slices, self.view_slices):
                output_frame[out_slice] = self.chip_data[view_slice]

        else:
            # skip extra cropping, extract the full chip directly into the output frame
            self._camera.ExtractColor(output_frame, mode)

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
        self._camera.GenStartMetadata(mdh)
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

    def __getattr__(self, key):
        """
        Proxy any methods we don't implement here by passing through to the base camera class
        """
        return getattr(self._camera, key)
