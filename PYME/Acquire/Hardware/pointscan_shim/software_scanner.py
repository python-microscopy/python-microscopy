
import time
import numpy as np
from PYME.Acquire.Hardware.pointscan_shim import pointscan_camera
import queue

class VoxelSignalProvider(object):
    """defines an interface for IO when using software-based scanning. 

    Attributes
    ----------
    n_channels : int
        the number of signal channels, i.e. the dimension of the signal 
        recorded at each scan position
    """
    n_channels = 1
    dtype = float
    def read(*args, **kwargs):
        raise NotImplementedError
    
    @property
    def voxel_dwell_time(self):
        raise NotImplementedError


class TestSignalProvider(VoxelSignalProvider):
    n_channels=None
    dtype=float
    def __init__(self, scanner):
        super().__init__()
        self.scanner = scanner
    
    @property
    def n_channels(self):
        return len(self.scanner._handlers)
    
    def read(self):
        return [self.scanner._handlers[ind].getValue() for ind in range(len(self.scanner._handlers))]

    @property
    def voxel_dwell_time(self):
        return 0.01


class LockInSignalProvider(VoxelSignalProvider):
    n_channels = 2
    dtype = float
    def __init__(self, lockin, n_tc_delay=3):
        super().__init__()
        self.lockin = lockin
        self.n_tc_delay = n_tc_delay
    
    def read(self):
        # ensure lock-in has settled
        time.sleep(self.n_tc_delay * self.lockin.time_constant)
        return self.lockin.snap()
    
    @property
    def voxel_dwell_time(self):
        return self.lockin.time_constant * self.n_tc_delay
    

class OffsetStageScanner(pointscan_camera.BaseScanner):
    """
    Handles stage scanning using offset piezos. This class MAY be useful if you
    want is a TileScanner that uses something other than a camera to acquire, but
    e.g. you have a camera you can use to find an ROI you would like to scan. 
    This scanner will scan about the current position of the stage.

    You would likely be better served to run the actual camera in a separate instance
    of PYMEAcquire, just control the stage from the raster camera shim instance.

    """
    def __init__(self, signal_provider, positioning, kwargs):
        """

        Parameters
        ----------
        positioning : dict
            PYME.Acquire.microscope.Microscopy.positioning dict, where each key
            is the axis name and each value is a tuple of the form (piezo, channel, ~)

            In order to use this class, each piezo registered to the micrscope
            object needs to be an OffsetPiezo, so that this class can adjust
            the offset. In the init script for the setup, for a multiaxis stage
            this looks something like the following:
            x = base_piezo.SingleAxisWrapper(scope.stage, 0)
            x = base_piezo.OffsetPiezo(x)
            y = base_piezo.SingleAxisWrapper(scope.stage, 2)
            y = base_piezo.OffsetPiezo(y)

            scope.register_piezo(x, 'x', needCamRestart=False, multiplier=-1)
            scope.register_piezo(y, 'y', needCamRestart=False, multiplier=1)
        axes_order : tuple, optional
            the scan order (fastest to slowest), by default ('x', 'y')
        """
        super().__init__(**kwargs)
        self.signal_provider = signal_provider
        self._positioning = positioning  #  (piezo, channel, 1*multiplier*units_um)
        self._handlers = self._gen_axes_handlers(self._axes_order)
        self._scan_buffer = np.zeros((self.width, self.height, self.n_channels), dtype=self.dtype)
    
    def refresh_scan_buffer(self):
        if self._scan_buffer.dtype == self.dtype and self._scan_buffer.shape != (self.width, self.height, self.n_channels):
            return
        self._scan_buffer = np.zeros((self.width, self.height, self.n_channels), dtype=self.dtype)
    
    def _gen_axes_handlers(self, axes_order):
        from PYME.Acquire.microscope import StateHandler #, StateManager
        _handlers = list()
        
        for ind in range(len(axes_order)):
            piezo, channel, multiplier = self._positioning[axes_order[ind]]
            # could be a good place to check type on piezo, ensuring it is offsetpiezo subclass
            _handlers.append(StateHandler(axes_order[ind], 
                                          piezo.GetOffset,  # GetTargetPos?? 
                                          piezo.SetOffset))
        return _handlers
    
    @pointscan_camera.BaseScanner.axes_order.setter
    def axes_order(self, axes_order):
        
        _handlers = self._gen_axes_handlers(axes_order)
        
        self._axes_order = axes_order
        self._handlers = _handlers
    
    @property
    def n_channels(self):
        self.signal_provider.n_channels
    
    @property
    def dtype(self):
        self.signal_provider.dtype
    
    def allocate_buffers(self, n_buffers):
        """queue up a number of single-frame buffers
        Note that each channel will be slotted into the queue as a separate frame (for now)
        Parameters
        ----------
        n_buffers : int
            number of single-frame (XY) buffers to allocate
        
        """
        self.full_frames = queue.Queue()
        self.refresh_scan_buffer()
        self.prepare()
        self.n_full = 0
    
    def update_position(self, pos):
        """Update the position of the stage. This function knows nothing about
        the offset, assumes it is moving the stage

        Parameters
        ----------
        pos : dict

        """
        for ind, axis in enumerate(self.axes_order):
            self._handlers[ind].setValue(pos[axis])
    
    def prepare(self):
        """re-zero the offsets so we can continue to scan at the same position
        """
        self.refresh_scan_buffer()
        self._axis_positions = {}
        self.n_scan_positions = 1
        for axis in self.axes_order:
            n_pixels_key = f'n_{axis}'
            self.n_scan_positions *= self._scan_params[n_pixels_key]
            if self._scan_params[n_pixels_key] == 1:
                self._axis_positions[axis] = np.array([0])
            else:
                assert (self._scan_params[n_pixels_key] % 2 == 1)
                half_roi = self._scan_params[n_pixels_key] // 2
                pixel_size = self._scan_params[f'voxelsize.{axis}']
                self._axis_positions[axis] = np.linspace(
                    -half_roi * pixel_size, half_roi * pixel_size,
                     self._scan_params[n_pixels_key], endpoint=True)
        
        # note, dicts remember insertion order for Python>=3.6
        meshgrid_scan_positions = np.meshgrid(*self._axis_positions.values())
        meshgrid_axis_indices = np.meshgrid(*[np.arange(len(self._axis_positions[axis])) for axis in self.axes_order])
        
        # FIXME - need to get the scan order right so we actually go slowest to fastest
        self._scan_positions = np.array(meshgrid_scan_positions).T.reshape(-1, len(self.axes_order))
        self._scan_axes_indices = np.array(meshgrid_axis_indices).T.reshape(-1, len(self.axes_order)).astype(int)
        
        # self.n_scan_positions = len(self._scan_positions)
        self.current_scan_position = int(-1) # start at -1 so that the first call to next() returns the first position

        # rezero the offsets so we can scan from the global microscope position
        for ind in range(len(self.axes_order)):
            self._handlers[ind].setValue(0)
    
    def next(self):
        """Return the next position to scan to

        Returns
        -------
        dict
            the next position to scan to
        """
        self.current_scan_position += 1
        pos = self._scan_positions[self.current_scan_position, :]
        # pos = {axis: self._scan_positions[self.current_scan_position, ind] for ind, axis in enumerate(self.axes_order)}
        self.update_position({axis: pos[ind] for ind, axis in enumerate(self.axes_order)})
        return self._scan_axes_indices[self.current_scan_position, :]
    
    def scan(self):
        """Scan the stage offsets and store output into image buffer.
        """
        try:
            for ind in range(self.n_scan_positions):
                indices = self.next()  # gets the scan axes indices
                signal = self.signal_provider.read()
                # tuple to avoid "Advanced indexing", which we don't want here
                self._scan_buffer[tuple(indices)]= signal

            with self.full_buffer_lock:
                # current implementation is to put each channel into the queue as a separate frame
                # and announce each as such. PYME.Acquire.cyztc.XYZTCAcquisition will handle the rest.
                for ind in range(self.n_channels):
                    self.full_frames.put(self._scan_buffer[:,:,ind])
                    self.n_full += 1
        finally:
            # re-zero the offsets
            for ind in range(len(self.axes_order)):
                self._handlers[ind].setValue(0)
    
    def wait_for_finished_buffer(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.n_full > 0:
                with self.full_buffer_lock:
                    return self.full_frames.get()
            time.sleep(0.05)
        raise TimeoutError('Timed out waiting for scanner buffer')
    
    @property
    def voxel_dwell_time(self):
        return self.signal_provider.voxel_dwell_time