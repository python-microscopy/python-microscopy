## raster_scan_shim.py

from PYME.Acquire.Hardware.Camera import Camera
# from PYME.Acquire.Hardware.Piezos.offsetPiezoREST import OffsetPiezo
# from PYME.Acquire.Hardware.Piezos.base_piezo import SingleAxisWrapper
import time
from PYME.Acquire.microscope import StateHandler #, StateManager
from PYME.contrib import dispatch
import numpy as np


class StageScanner(object):
    """
    basePiezo position - offset = OffsetPiezo position

    Parameters
    ----------
    object : _type_
        _description_
    """
    def __init__(self, positioning, axes_order=('x', 'y', 'z')):
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
            z = base_piezo.SingleAxisWrapper(scope.stage, 1)
            z = base_piezo.OffsetPiezo(z)

            scope.register_piezo(x, 'x', needCamRestart=False, multiplier=-1)
            scope.register_piezo(y, 'y', needCamRestart=False, multiplier=1)
            scope.register_piezo(z, 'z', needCamRestart=False)
        axes_order : tuple, optional
            the scan order (fastest to slowest), by default ('x', 'y', 'z')
        """
        
        self._positioning = positioning  #  (piezo, channel, 1*multiplier*units_um)
        self._axes_order = axes_order
        self._handlers = self._gen_axes_handlers(self._axes_order)
        
        self.on_move = dispatch.Signal()
    
    @property
    def axes_order(self):
        return self._axes_order
    
    def _gen_axes_handlers(self, axes_order):
        _handlers = list()
        
        for ind in range(len(axes_order)):
            piezo, channel, multiplier = self._positioning[axes_order[ind]]
            # could be a good place to check type on piezo, ensuring it is offsetpiezo subclass
            _handlers.append(StateHandler(axes_order[ind], 
                                          piezo.GetOffset,  # GetTargetPos?? 
                                          piezo.SetOffset))
        return _handlers
    
    @axes_order.setter
    def axes_order(self, axes_order):
        
        _handlers = self._gen_axes_handlers(axes_order)
        
        self._axes_order = axes_order
        self._handlers = _handlers
    
    def update_position(self, pos):
        """Update the position of the stage. This function knows nothing about
        the offset, assumes it is moving the stage

        Parameters
        ----------
        pos : dict

        """
        for ind, axis in enumerate(self.axes_order):
            self._handlers[ind].setValue(pos[axis])
        self.on_move.send(self, position=pos)
    
    def prepare(self, scan_params):
        """re-zero the offsets so we can continue to scan at the same position
        """
        self._axis_positions = {}
        self.n_scan_positions = 1
        for axis in self.axes_order:
            n_pixels_key = f'n_pixels_{axis}'
            self.n_scan_positions *= scan_params[n_pixels_key]
            if scan_params[n_pixels_key] == 1:
                self._axis_positions[axis] = np.array([0])
            else:
                assert (scan_params[n_pixels_key] % 2 == 1)
                half_roi = scan_params[n_pixels_key] // 2
                pixel_size = scan_params[f'{axis}_step_size']
                self._axis_positions[axis] = np.concatenate([
                    np.arange(-half_roi * pixel_size, 0, pixel_size), # negative direction
                    np.array([0]),  # zero offset position
                    np.arange(pixel_size, half_roi * pixel_size + pixel_size, pixel_size)  # positive direction
                ])
        
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


class VoxelSignalProvider(object):
    data_len = 1
    dtype = float
    def read(*args, **kwargs):
        raise NotImplementedError
    
    @property
    def pixel_dwell_time(self):
        raise NotImplementedError


class TestSignalProvider(VoxelSignalProvider):
    dtype=float
    def __init__(self, scanner):
        super().__init__()
        self.scanner = scanner
    
    @property
    def data_len(self):
        return len(self.scanner._handlers)
    
    def read(self):
        return [self.scanner._handlers[ind].getValue() for ind in range(len(self.scanner._handlers))]
        # return [self.scanner._handlers[0].getValue(), 
                # self.scanner._handlers[1].getValue()]

    @property
    def pixel_dwell_time(self):
        return 0.01



class LockInSignalProvider(VoxelSignalProvider):
    data_len = 2
    dtype = float
    # noise properties?
    def __init__(self, lockin, n_tc_delay=3):
        super().__init__()
        self.lockin = lockin
        self.n_tc_delay = n_tc_delay
    
    def read(self):
        # ensure lock-in has settled
        time.sleep(self.n_tc_delay * self.lockin.time_constant)
        return self.lockin.snap()
    
    @property
    def pixel_dwell_time(self):
        return self.lockin.time_constant * self.n_tc_delay


class RasterscanCameraShim(Camera):
    """Class to permit a stage/mirror/etc-scanning instrument to act like a 
    camera-based system in PYMEAcquire.

    """
    def __init__(self, position_scanner, signal_provider, *args, **kwargs):
        super().__init__()
        self._scan_params = {
            'n_pixels_x': 1,  # [px]
            'n_pixels_y': 1,  # [px]
            'n_pixels_z': 1,  # [px]
            'x_step_size': 1,  # [um]
            'y_step_size': 1,  # [um]
            'z_step_size': 1,  # [um]
            'pixel_clock_freq': 1,  # [Hz]
            'pixel_integration_time': 1,  # [s]
        }

        self.scanner = position_scanner
        self.signal_provider = signal_provider
        self.set_scan_params(kwargs)
    
    def set_scan_params(self, scan_params):
        self._scan_params.update(scan_params)
        self._image_buffer = np.zeros([self._scan_params['n_pixels_x'], 
                                       self._scan_params['n_pixels_y'],
                                       self._scan_params['n_pixels_z'], 
                                       self.signal_provider.data_len],
                                       dtype=self.signal_provider.dtype)
    
    def scan(self):
        """Scan the stage/mirror/etc. and return the signal.

        Returns
        -------
        signal : np.ndarray
            the signal from the scan
        """
        self.scanner.prepare(self._scan_params)

        for ind in range(self.scanner.n_scan_positions):
            indices = self.scanner.next()  # gets the scan axes indices
            signal = self.signal_provider.read()
            # tuple to avoid "Advanced indexing", which we don't want here
            self._image_buffer[tuple(indices)]= signal
        
        return self._image_buffer
    
    def frame_rate(self):
        return self._scan_params['n_pixels_x'] * self._scan_params['n_pixels_y'] \
              * self._scan_params['n_pixels_z'] / self.signal_provider.pixel_dwell_time
    
    def GetIntegTime(self):
        return self.scanner.frame_rate()
        