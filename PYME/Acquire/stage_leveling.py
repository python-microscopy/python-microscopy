
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class StageLeveler(object):
    def __init__(self, scope, offset_piezo, pause_on_relocate=0.25, 
                 focus_lock=None):
        """
        Allows semi-automated mapping of coverslip positions are various lateral positions using a focus lock/offset
        piezo to determine the offset.

        Usage after initialization
        --------------------------
        1 Manually, e.g. with joystick, move the stage to various positions and call the `add_position` method at each.
        2 Run `measure_offsets` to automatically visit each of these positions and log the offset.
        3 Call `plot` to have a look at the map and get a feel for any adjustments you need to make on the hardware

        Parameters
        ----------
        scope: PYME.Acquire.microscope.Microscope
        offset_piezo: PYME.Acquire.Hardware.offsetPiezoREST.OffsetPiezo
        pause_on_relocate: float
            [optional] time to pause during measure loop after moving to a new location and before measuring offset.
        focus_lock : PYME.Acquire.Hardware.focus_locks

        Notes
        -----
        Units are derived from PYME.Acquire.microscope.Microscope.GetPos and SetPos and should be in micrometers.

        Attributes
        ----------
        _positions : list
            list of position dictionaries with 'x' and 'y' keys in units of micrometers.
        _scans: list of dict
            list of measurement iterations. Each scan is a dict with 'x', 'y', and 'offset' keys which each return 1d
            arrays, all in units of micrometers.

        """
        self._scope = scope
        self._offset_piezo = offset_piezo
        self._focus_lock = focus_lock
        if self._focus_lock == None:
            try:
                self._focus_lock = scope.focus_lock
            except AttributeError:
                pass
        self._positions = []
        self._scans = []
        self._pause_on_relocate = pause_on_relocate

    def clear_positions(self):
        """
        Removes all positions stored by the stage leveler (positions to visit, not scans of visited positions)
        """
        self._positions = []

    def add_position(self):
        """
        Add the current microscope position to the list of positions to be scanned on subsequent calls to
        measure_offsets.
        """
        self._positions.append(self._scope.GetPos())

    def add_grid(self, x_length, y_length, x_spacing, y_spacing, center):
        """
        Add a grid of set spacings to the list of positions to scan when measuring offsets.

        Parameters
        x_length : float
            approximate x length to span in the grid, units of micrometers.
        y_length : float
            approximate y length to span in the grid, units of micrometers.
        x_spacing : float
            x grid spacing, micrometers.
        y_spacing : float
            y grid spacing, micrometers.'
        center : bool
            center the grid on the current position (True) or take the current
            scope position to be the smallest x, y position to queue (False)
        """

        x = np.arange(0, x_length, x_spacing, dtype=float)
        y = np.arange(0, y_length, y_spacing, dtype=float)

        if center:
            x = x - (0.5 * x.max())
            y = y - (0.5 * y.max())
        
        current = self._scope.GetPos()
        x += current['x']
        y += current['y']
        
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

        self._positions.extend([{'x': xi, 'y': yi} for xi, yi in zip(x_grid.ravel(), y_grid.ravel())])
    
    def add_96wp_positions(self, short='x'):
        """Shortcut for queueing center positions on a 96-well plate from 
        minimum x, y well. x (8 well) should be short axis, y (12 well) long

        Parameters
        ----------
        short: str
            stage dimension of the short axis (8 wells) of the plate. Defaults
            to x.
        """
        if short=='x':
            self.add_grid(9000 * 8, 9000 * 12, 9000, 9000, center=False)
        elif short=='y':
            self.add_grid(9000 * 12, 9000 * 8, 9000, 9000, center=False)
        else:
            logger.error('short axes must be "x" or "y"')
    
    def add_ibidi8wellslide_positions(self, short='x'):
        """Shortcut for queueing center positions on a ibidi 8 well slide from
        minimum x, y well. x (2 well) should be short axis, y (4 well) long

        Parameters
        ----------
        short: str
            stage dimension of the short axis (2 wells) of the plate. Defaults
            to x.
        """
        if short=='x':
            self.add_grid(11.2 * 1e3 * 2, 12.5 * 1e3 * 4, 
                          11.2 * 1e3, 12.5 * 1e3, center=False)
        elif short=='y':
            self.add_grid(12.5 * 1e3 * 4, 11.2 * 1e3 * 2,
                          12.5 * 1e3, 11.2 * 1e3, center=False)
        else:
            logger.error('short axes must be "x" or "y"')

    def measure_offsets(self, optimize_path=True, use_previous_scan=True):
        """
        Visit each position and log the offset

        Parameters
        ----------
        optimize_path : bool
            Flag to toggle visiting the positions in an order which minimizes the path relative to the microscope
            starting position.

        """
        from PYME.Analysis.points.traveling_salesperson import sort
        n_positions = len(self._positions)
        offset = np.zeros(len(self._positions), dtype=float)
        lock_ok = np.ones(len(self._positions), dtype=bool)
        x, y = np.zeros_like(offset), np.zeros_like(offset)
        positions = np.zeros((n_positions, 2), dtype=float)
        for ind in range(n_positions):
            positions[ind, :] = (self._positions[ind]['x'], self._positions[ind]['y'])

        if optimize_path:
            current_pos = self._scope.GetPos()
            positions = sort.tsp_sort(positions, start=(current_pos['x'], current_pos['y']))

        for ind in range(n_positions):
            self._scope.SetPos(x=positions[ind, 0], y=positions[ind, 1])
            
            time.sleep(self._pause_on_relocate)
            if hasattr(self, '_focus_lock') and not self._focus_lock.LockOK():
                logger.debug('focus lock not OK, scanning offset')
                if use_previous_scan:
                    try:
                        start_at = self.lookup_offset(positions[ind, 0],
                                                      positions[ind, 1])
                    except:
                        start_at = -25
                else:
                    start_at = -25
                self._focus_lock.ReacquireLock(start_at=start_at)
                time.sleep(1.)

                if self._focus_lock.LockOK():
                    time.sleep(1.)
            actual = self._scope.GetPos()
            x[ind], y[ind] = actual['x'], actual['y']
            offset[ind] = self._offset_piezo.GetOffset()
            try:
                lock_ok[ind] = self._focus_lock.LockOK()
                logger.debug('lock OK %s, x %.1f, y %.1f, offset %.1f' % (lock_ok[ind],
                                                                            x[ind], y[ind],
                                                                            offset[ind]))
            except AttributeError:
                logger.debug('x %.1f, y %.1f, offset %.1f' % (x[ind], y[ind], 
                                                              offset[ind]))

        self._scans.append({
            'x': x[lock_ok], 'y': y[lock_ok], 'offset': offset[lock_ok]
        })

    @staticmethod
    def plot_scan(scan, interpolation_factor=50):
        """

        Parameters
        ----------
        scan: dict
            x: x positions
            y: y positions
            offset: offset positions
        interpolation_factor: int
            number of points along x and y to interpolate the offset map.

        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata

        x_min, x_max = np.min(scan['x']), np.max(scan['x'])
        y_min, y_max = np.min(scan['y']), np.max(scan['y'])
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, interpolation_factor),
                                     np.linspace(y_min, y_max, interpolation_factor),
                                     indexing='ij')
        x_spacing = x_grid[1, 0] - x_grid[0, 0]
        y_spacing = y_grid[0, 1] - y_grid[0, 0]

        scan_map = griddata(np.stack([scan['x'], scan['y']], axis=1), scan['offset'], (x_grid, y_grid))

        plt.figure()
        plt.imshow(scan_map, interpolation='nearest', origin='lower', extent=(y_min - 0.5 * y_spacing, y_max + 0.5 * y_spacing,
                                                                              x_min - 0.5 * x_spacing, x_max + 0.5 * x_spacing))
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Offset [um]')
        plt.plot(scan['y'], scan['x'], 'k--')
        plt.scatter(scan['y'], scan['x'], marker='x', label='measured')
        plt.xlabel('y [um]')
        plt.ylabel('x [um]')
        plt.title('basePiezo position - offset = OffsetPiezo position')
        plt.legend()
        plt.axes().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.show()

    def plot(self, index=-1, interpolation_factor=50):
        if len(self._scans) < 1:
            raise UserWarning('no scans available, call StageLeveler.measure_offsets() first')
        StageLeveler.plot_scan(self._scans[index], interpolation_factor=interpolation_factor)

    def store_scan(self, index=-1):
        self._current_scan = self._scans[index]

    @property
    def current_scan(self):
        try:
            return self._current_scan
        except AttributeError:
            if len(self._scans) > 0:
                return self._scans[-1]

    def lookup_offset(self, x, y, default=0):
        """use a stored scan to estimate what the z offset should be at a given
        xy position

        Parameters
        ----------
        x : float
            x position in micrometers
        y : float
            y position in micrometers

        Returns
        -------
        float
            offset at xy from interpolated scan
        """
        from scipy.interpolate import interp2d
        try:
            scan = self.current_scan
        except (IndexError, ValueError):
            logger.error('no scan, returning %f for offset lookup' % default)
            return default
        f = interp2d(scan['x'], scan['y'], scan['offset'])
        return f(x, y)[0]
    
    def acquire_focus_lock(self):
        self._focus_lock.EnableLock()
        if self._focus_lock.LockOK():
            return
        time.sleep(1)
        if not self._focus_lock.LockOK():
            p = self._scope.GetPos()
            self._scope.focus_lock.ReacquireLock(self.lookup_offset(p['x'], p['y']))
