from PYME.Acquire.Hardware.Camera import Camera
import time
import numpy as np
import threading
import queue
import logging
logger = logging.getLogger(__name__)


class BaseScanner(object):
    """The scanner class handles stage/mirror/etc scanning and signal acquisition
    In short, it mimics the hardware-side of a camera, with frame buffering, etc.

    Attributes
    ----------
    n_channels : int
        the dimension of the signal recorded at each scan position. Default is 1.
        This will set the color (C) dimension of the final saved image.
    dtype : type
        the data type of the signal. Default is int.
    
    """
    n_channels = 1
    dtype = int
    def __init__(self, scan_params=None, axes_order=('x', 'y')):
        self._axes_order = axes_order
        self._scan_params = {
            'n_x': 1,  # [px]
            'n_y': 1,  # [px]
            'voxelsize.x': 1,  # [nm]
            'voxelsize.y': 1,  # [nm]
            'voxel_integration_time': 1,  # [s]
            'voxel_dwell_time': 1,  # [s]
        }
        self.full_buffer_lock = threading.Lock()
        self.full_buffers = None
        self.free_buffers = None
        self.n_full = 0
        if scan_params is not None:
            self.set_scan_params(scan_params)

    def set_scan_params(self, scan_params):
        self._scan_params.update(scan_params)

    @property
    def axes_order(self):
        """order to scan the axes during image acquisition.
        Does not currently support making Z the fast-axis.

        Returns
        -------
        tuple
            str identifying axes in the order they should be scanned
        """
        return self._axes_order
    
    @property
    def frame_rate(self):
        """ calculates a proxy for framerate based on voxel dwell time and number of pixels. Override in subclass if needed.

        Returns
        -------
        FPS : float
            frame rate in frames per second, or [Hz]
        """
        return 1 / (self._scan_params['n_x'] * self._scan_params['n_y'] * self._scan_params['voxel_dwell_time'])
    
    @property
    def frame_cycle_time(self):
        """frame cycle time in seconds

        Returns
        -------
        cycle_time : float
            time to acquire a single frame in seconds
        """
        return 1 / self.frame_rate
    
    @property
    def frame_integ_time(self):
        return self._scan_params['n_x'] * self._scan_params['n_y'] * self._scan_params['voxel_integration_time']
    
    def set_frame_integ_time(self, iTime):
        n_pixels = self.width * self.height
        self._scan_params['voxel_integration_time'] = iTime / n_pixels
        # assume dwell time is the same as integration time, otherwise override in subclass
        self._scan_params['voxel_dwell_time'] = self._scan_params['voxel_integration_time']

    @property
    def width(self):
        return self._scan_params['n_x']
    
    @property
    def height(self):
        return self._scan_params['n_y']
    
    @property
    def voxel_dwell_time(self):
        return self._scan_params['voxel_dwell_time']
    
    def get_serial_number(self):
        raise NotImplementedError
    
    def scan(self):
        """raster-scan an image and add each channel as a separate frame to the full frame
        buffer

        """
        raise NotImplementedError
    
    def allocate_buffers(self, n_buffers):
            """queue up a number of single-frame buffers
            Note that each channel will be slotted into the queue as a separate frame (for now)
            Parameters
            ----------
            n_buffers : int
                number of single-frame (XY) buffers to allocate
            
            """
            self.free_buffers = queue.Queue()
            self.full_buffers = queue.Queue()
            self.n_full = 0
            for ind in range(n_buffers):
                self.free_buffers.put(np.zeros((self.width, self.height), 
                                            dtype=self.dtype))
        
    def destroy_buffers(self):
        with self.full_buffer_lock:
            self.n_full = 0
            while not self.full_buffers.empty():
                try:
                    self.full_buffers.get_nowait()
                except queue.Empty:
                    pass
    
    def stop(self):
        raise NotImplementedError


class PointscanCameraShim(Camera):
    """ Interfaces with a scanner object to provide a Camera-like interface for
    PYMEAcquire.
    """
    supports_software_trigger = True
    
    def __init__(self, position_scanner):
        self.initialized = False
        super().__init__()
        self.scanner = position_scanner
        self._mode = self.MODE_SINGLE_SHOT

        self._buffer_lock = threading.Lock()
        self.full_buffers = None
        self._buffer_poll_wait_time = 5  # [s]

        self.Init()
        self.initialized = True
    
    @property
    def n_channels(self):
        return self.scanner.n_channels
    
    def Init(self):        
        self._poll = False
        self.poll_loop_active = True
        self.poll_thread = threading.Thread(target=self._poll_loop)
        self.poll_thread.start()
    
    def _poll_loop(self):
        while self.poll_loop_active:
            if self._poll :
                try:
                    # with self._buffer_lock:
                    self._poll_buffer()
                except Exception as e:
                    logger.exception(str(e))
            else:
                time.sleep(0.05)
    
    def ExpReady(self):
        return (self.full_buffers is not None) and (self.n_full > 0)
    
    def ExtractColor(self, ch_slice, mode):
        # get nowait to hard-throw an Empty error if we've entered this method
        # and we shouldn't have
        buf = self.full_buffers.get_nowait()
        ch_slice[:] = buf
        if self.free_buffers is not None:
            # recycle buffer
            self.free_buffers.put(buf)
            self.n_full -= 1
    
    def _poll_buffer(self):
        try:
            # get a frame from the scanner
            d = self.scanner.wait_for_finished_buffer(self._buffer_poll_wait_time)
            # store it in a free camera buffer
            buf = self.free_buffers.get_nowait()
            buf[:] = d  # copy
            # requeue the scanner buffer
            self.scanner.free_buffers.put(d)
            self.scanner.n_full -= 1
            # put the camera buffer in the full buffer queue
            self.full_buffers.put(buf)
            self.n_full += 1
            
        except Exception as e:
            logger.error(f'Error polling buffer: {e}')
    
    def GetAcquisitionMode(self):
        return self._mode
    
    def SetAcquisitionMode(self, mode):
        """Set a flag so next StartExposure can toggle between continuous and single shot mode.

        Parameters
        ----------
        mode : int
            toggles between continuous and single shot mode
        """
        self._mode = mode

    def CamReady(self):
        """
        Returns true if the camera is ready (initialized) not really used for
        anything, but might still be checked.

        Returns
        -------
        bool
            Is the camera ready?
        """
    
        return self.initialized
    
    def frame_rate(self):
        return self.scanner.frame_rate
    
    def GetIntegTime(self):
        return self.scanner.frame_integ_time
    
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
        self.scanner.set_frame_integ_time(iTime)
    
    def FireSoftwareTrigger(self):
        self.scanner.scan()
        self._log_exposure_start()
    
    def StartExposure(self):
        logger.debug('StartAq')
        if self._poll:
            # stop, we'll allocate buffers and restart
            self.StopAq()
        # allocate at least 2 seconds of buffers
        buffer_size = int(max(2 * self.GetFPS(), 10)) * self.n_channels
        logger.info('Allocating {} buffers'.format(buffer_size))
        self.allocate_buffers(buffer_size)

        self._log_exposure_start()
        try:
            # if self._mode == self.MODE_SOFTWARE_TRIGGER:
            if self._mode == self.MODE_CONTINUOUS:
                raise NotImplementedError
            elif self._mode == self.MODE_SINGLE_SHOT:
                # self.FireSoftwareTrigger()
                pass
        except Exception as e:
            logger.error(f'Error starting acquisition: {e}')
            raise e
        self._poll = True
        return 0

    def DestroyBuffers(self):
        with self._buffer_lock:
            self.n_full = 0

            # remove scanner-side buffers
            self.scanner.destroy_buffers()
            
            # computer RAM: destroy free and full buffer queues
            while not self.full_buffers.empty():
                try:
                    self.full_buffers.get_nowait()
                except queue.Empty:
                    pass
            
            while not self.free_buffers.empty():
                try:
                    self.free_buffers.get_nowait()
                except queue.Empty:
                    pass
    
    def allocate_buffers(self, n_buffers=10):
        self.n_full = 0
        self._n_cam_buffers = n_buffers
        # scanner side
        self.scanner.allocate_buffers(n_buffers)

        # 'camera' shim side
        self.free_buffers = queue.Queue()
        self.full_buffers = queue.Queue()
        for ind in range(n_buffers):
            # could probably initialize w/ empty here if we wanted
            self.free_buffers.put(np.zeros((self.GetPicHeight(), self.GetPicWidth()), 
                                           dtype=self.scanner.dtype))
        self._poll = False
    
    def StopAq(self):
        self._poll = False
        try:
            # tell the scanner to stop
            self.scanner.stop()
        except Exception as e:
            logger.error(f'Error stopping acquisition: {e}')
            raise e
        self.DestroyBuffers()
    
    def GetNumImsBuffered(self):
        """
        Return the number of images in the buffer.

        Returns
        -------
        int
            Number of images in buffer
        """
        
        return self.n_full

    def GetBufferSize(self):
        """
        Return the total size of the buffer (in images).

        Returns
        -------
        int
            Number of images that can be stored in the buffer.
        """
        if self._poll:
            return self._n_cam_buffers
        else:
            # if we aren't polling, spoof infinitely large buffer so we don't
            # flag a buffer overflow while we e.g. rebuild the buffers. This
            # makes no functional difference for us, but avoids a spurious
            # warning from frameWrangler about the buffer overflowing.
            return np.iinfo(np.int32).max
    
    def GetFPS(self):
        """
        Get the camera frame rate in frames per second (float).

        Returns
        -------
        float
            Camera frame rate (frames per second)
        """

        return self.scanner.frame_rate
    
    def GetCycleTime(self):
        """
        Get camera cycle time (1/fps) in seconds (float)

        Returns
        -------
        float
            Camera cycle time (seconds)
        """
        return self.scanner.frame_cycle_time
    
    def GetSerialNumber(self):
        return self.scanner.get_serial_number()
    
    def GetCCDTemp(self):
        return 0
    
    def GetCCDHeight(self):
        return self.scanner.width
    
    def GetCCDWidth(self):
        return self.scanner.height
    
    def GetPicWidth(self):
        """
        Returns the width (in pixels) of the currently selected ROI.

        Returns
        -------
        int
            Width of ROI (pixels)
        """
        return self.scanner.width

    def GetPicHeight(self):
        """
        Returns the height (in pixels) of the currently selected ROI
        
        Returns
        -------
        int
            Height of ROI (pixels)
        """
        return self.scanner.height
    
    def GetROI(self):
        return (0, 0,  self.scanner.width, self.scanner.height)
    

