
import numpy as np
import time

class StageLeveler(object):
    def __init__(self, scope, offset_piezo, pause_on_relocate=0.25):
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
        scope: PYME.Acquire.microscope.microscope
        offset_piezo: PYME.Acquire.Hardware.offsetPiezoREST.OffsetPiezo
        pause_on_relocate: float
            [optional] time to pause during measure loop after moving to a new location and before measuring offset.

        Notes
        -----
        Units are derived from PYME.Acquire.microscope.microscope.GetPos and SetPos and should be in micrometers.

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

    def add_grid(self, x_length, y_length, x_spacing, y_spacing, center=True):
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

    def measure_offsets(self, optimize_path=True):
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
            actual = self._scope.GetPos()
            x[ind], y[ind] = actual['x'], actual['y']
            offset[ind] = self._offset_piezo.GetOffset()

        self._scans.append({
            'x': x, 'y': y, 'offset': offset
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
        plt.scatter(scan['y'], scan['x'], marker='x', label='measured')
        plt.xlabel('y [um]')
        plt.ylabel('x [um]')
        plt.legend()
        plt.axes().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.show()

    def plot(self, index=-1, interpolation_factor=50):
        if len(self._scans) < 1:
            raise UserWarning('no scans available, call StageLeveler.measure_offsets() first')
        StageLeveler.plot_scan(self._scans[index], interpolation_factor=interpolation_factor)
