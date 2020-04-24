

class VoxelSize(object):
    _CONVERSION_DICT = {
        'm': {
            'm': 1.0, 'mm': 1e3, 'um': 1e6, 'nm': 1e9
        },
        'mm': {
            'm': 1e-3, 'mm': 1.0, 'um': 1e3, 'nm': 1e6
        },
        'um': {
            'm': 1e-6, 'mm': 1e-3, 'um': 1.0, 'nm': 1e3
        },
        'nm': {
            'm': 1e-9, 'mm': 1e-6, 'um': 1e-3, 'nm': 1.0
        }
    }

    def __init__(self, x, y, z=0.1, units_x='um', units_y='um', units_z='um'):
        self._x = x
        self._y = y
        self._z = z

        self._units_x = units_x
        self._units_y = units_y
        self._units_z = units_z

    def __getitem__(self, item):
        return (self.x_um, self.y_um, self.z_um)[item]

    def x(self, units='um'):
        """

        Parameters
        ----------
        units: str
            Units to return size in, must be 'm', 'mm', 'um', or 'nm'. Defaults to 'um'.

        Returns
        -------
        voxelsize_x: float
            x pixel size in specified units

        """
        return self._x * self._CONVERSION_DICT[self._units_x][units]

    def y(self, units='um'):
        """

        Parameters
        ----------
        units: str
            Units to return size in, must be 'm', 'mm', 'um', or 'nm. Defaults to 'um'.

        Returns
        -------
        voxelsize_y: float
            y pixel size in specified units

        """
        return self._y * self._CONVERSION_DICT[self._units_y][units]

    def z(self, units='um'):
        """

        Parameters
        ----------
        units: str
            Units to return size in, must be 'm', 'mm', 'um', or 'nm. Defaults to 'um'.

        Returns
        -------
        voxelsize_z: float
            z voxel size in specified units
        """
        return self._z * self._CONVERSION_DICT[self._units_z][units]

    @property
    def x_um(self):
        """
        Returns
        -------
        voxelsize_x: float
            x pixel size in units of micrometers
        """
        return self.x()

    @property
    def y_um(self):
        """
        Returns
        -------
        voxelsize_y: float
            y pixel size in units of micrometers
        """
        return self.y()

    @property
    def z_um(self):
        """
        Returns
        -------
        voxelsize_z: float
            z pixel size in units of micrometers
        """
        return self.z()

    def as_metadata_handler(self, units='um'):
        """
        Create a metadata handler containing all of the voxelsize information
        Parameters
        ----------
        units: str
            Units to return size in, must be 'm', 'mm', 'um', or 'nm. Defaults to 'um'.

        Returns
        -------
        mdh: PYME.IO.MetaDataHandler
            metadata handler containing all voxelsize information.
        """
        from PYME.IO.MetaDataHandler import NestedClassMDHandler

        mdh = NestedClassMDHandler()

        mdh['voxelsize.x'] = self.x(units)
        mdh['voxelsize.y'] = self.y(units)
        mdh['voxelsize.z'] = self.z(units)
        mdh['voxelsize.units'] = units

        return mdh
