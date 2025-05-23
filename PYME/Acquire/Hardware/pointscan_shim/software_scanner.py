
import time
import numpy as np
from PYME.Acquire.Hardware.pointscan_shim import pointscan_camera
import queue
import threading

class VoxelSignalProvider(object):
    """defines an interface for IO when using software-based scanning. 

    Attributes
    ----------
    n_channels : int
        the number of signal channels, i.e. the dimension of the signal 
        recorded at each scan position
    """
    n_channels = 1
    dtype = 'float64'
    def read(*args, **kwargs):
        raise NotImplementedError
    
    @property
    def voxel_dwell_time(self):
        raise NotImplementedError


class TestSignalProvider(VoxelSignalProvider):
    """uses the software scanner position to generate a 'signal'
    """
    n_channels=2  # set up for x, y scanner
    dtype = 'float64'
    def __init__(self, x_positioner, y_positioner):
        super().__init__()
        self.x_positioner = x_positioner
        self.y_positioner = y_positioner
    
    def read(self):
        return [self.x_positioner.GetPos(), self.y_positioner.GetPos()]

    @property
    def voxel_dwell_time(self):
        return 0.01


class LockInSignalProvider(VoxelSignalProvider):
    n_channels = 2
    dtype = 'float64'
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

class SoftwareStageScanner(pointscan_camera.BaseScanner):
    def __init__(self, x_positioner, y_positioner, signal_provider, kwargs=None):
        """

        Parameters
        ----------
        x_positioner :
            the x-axis positioner (PiezoBase subclass, possibly wrapped in SingleAxisWrapper)
        y_positioner :
            the y-axis positioner (PiezoBase subclass, possibly wrapped in SingleAxisWrapper)
        signal_provider : VoxelSignalProvider
            the signal provider
        kwargs : dict
            allows construction with pre-set scan parameters

        """
        super().__init__(**kwargs)
        self.x_positioner = x_positioner
        self.y_positioner = y_positioner
        self.signal_provider = signal_provider
        
        self._scan_buffer = np.zeros((self.width, self.height, self.n_channels), dtype=self.dtype)
    
    @property
    def n_channels(self):
        return self.signal_provider.n_channels
    
    @property
    def dtype(self):
        return self.signal_provider.dtype
    
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
    
    def _scan(self):
            # note current center position
            x0 = self.x_positioner.GetPos()
            y0 = self.y_positioner.GetPos()

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
            
            self.current_scan_position = int(-1) # start at -1 so that the first call to next() returns the first position

            # start the scan
            for ind in range(self.n_scan_positions):
                self.current_scan_position += 1
                pos = self._scan_positions[self.current_scan_position, :]
                # pos = {axis: self._scan_positions[self.current_scan_position, ind] for ind, axis in enumerate(self.axes_order)}
                pos_dict = {axis: pos[ind] for ind, axis in enumerate(self.axes_order)}
                self.x_positioner.MoveTo(0, x0 + pos_dict['x'])
                self.y_positioner.MoveTo(0, y0 + pos_dict['y'])
                # return self._scan_axes_indices[self.current_scan_position, :]
                signal = self.signal_provider.read()
                # tuple to avoid "Advanced indexing", which we don't want here
                self._scan_buffer[tuple(self._scan_axes_indices[self.current_scan_position, :])]= signal
            
            # return to the original position
            self.x_positioner.MoveTo(0, x0)
            self.y_positioner.MoveTo(0, y0)

            # write the scan buffer to the full frame buffer
            for ind in range(self.n_channels):
                buf = self.free_buffers.get_nowait()
                buf[:] = self._scan_buffer[:,:,ind]
                with self.full_buffer_lock:
                    self.full_buffers.put(buf)
                    self.n_full += 1

    def scan(self):
            """raster-scan an image and add each channel as a separate frame to the full frame
            buffer

            """
            # t = threading.Thread(target=self._scan)
            # t.start()
            self._scan()
        
    def get_serial_number(self):
        return 'Number0'
    
    def stop(self):
        pass
