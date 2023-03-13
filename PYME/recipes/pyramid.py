
from .base import ModuleBase, register_module, register_legacy_module, OutputModule
from .traits import Input, Output, CStr, Int
from PYME.IO import tabular
import warnings
import logging

logger = logging.getLogger(__name__)


@register_module('SupertilePhysicalCoords')
@register_legacy_module('TilePhysicalCoords', 'supertile')
class SupertilePhysicalCoords(ModuleBase):
    """
    Adds x_um, y_um, x_px, and y_px columns to input measurements performed on an Supertile image sequence, mapping
    the x and y values to physical coordinates.

    Parameters
    ----------
    input_measurements: PYME.IO.tabular.TabularBase
        dict-like containing 'x' and 'y' coordinates to map into the physical reference frame of the supertile
    input_supertile: PYME.IO.ImageStack
        MUST wrap have a `PYME.IO.DataSources.SupertileDatasource` instance as the `data` attribute
    measurement_units: str
        One of 'um' for micrometers, 'nm' for nanometers, or 'px' for pixels. Has to match the units of 'x' and 'y' from
        `inputMeasurements`
    """

    input_measurements = Input('measurements')
    input_supertile = Input('input')
    measurement_units = CStr('nm')
    output_name = Output('meas_physical_coords')
    
    def run(self, input_measurements, input_supertile):
        from PYME.IO import MetaDataHandler
        meas = input_measurements
        img = input_supertile
        
        out = tabular.MappingFilter(meas)
        
        x_frame_um, y_frame_um =img.data.tile_coords_um[meas['t']].T
        x_frame_px, y_frame_px = img.data.tile_coords[meas['t']].T

        if self.measurement_units == 'um':
            x_to_micron, y_to_micron = 1, 1
            x_to_pixels, y_to_pixels = 1 / meas.mdh['voxelsize.x'], 1 / meas.mdh['voxelsize.y']
        elif self.measurement_units == 'nm':
            x_to_micron, y_to_micron = 1e-3, 1e-3
            x_to_pixels, y_to_pixels = 1 / (1e3 * meas.mdh['voxelsize.x']), 1 / (1e3 * meas.mdh['voxelsize.y'])
        elif self.measurement_units == 'px':
            x_to_micron, y_to_micron = meas.mdh['voxelsize.x'], meas.mdh['voxelsize.y']
            x_to_pixels, y_to_pixels = 1, 1
        else:
            raise RuntimeError("Supported units include 'um', 'nm', and 'px'")

        out.addColumn('x_um', x_frame_um + meas['x'] * x_to_micron)
        out.addColumn('y_um', y_frame_um + meas['y'] * y_to_micron)
        out.addColumn('x_px', x_frame_px + meas['x'] * x_to_pixels)
        out.addColumn('y_px', y_frame_px + meas['y'] * y_to_pixels)
        
        out.mdh = MetaDataHandler.DictMDHandler(meas.mdh)
        
        return out


@register_legacy_module('Supertile', 'supertile')
class _Supertile(ModuleBase):
    """
    DO NOT USE: Create a tile pyramid on disk (optionally using the TilePyramid
    output module in a separate recipe) and open this using a SUPERTILE: URI
    instead.
    
    This module will be removed.
    
    Take an input series acquired ay many positions, tiling an image, and 
    insert those images into the proper position, averaging with any overlapping
    images. Output is a series of sub-frames of this nicely coalesced overview,
    at a given zoom (with a given overlap between these sub-frames).

    Parameters
    ----------
    input_name : str
        `PYME.IO.image.ImageStack` of a tiled series.
    base_tile_size : int
        number of pixels on each side of the base level (level 0) tiles. Output
        series will be a multiple of this size as set by stride. By default, 256
    level : int
        the number of times to zoom-out by a factor of two from the original
        series pixel size. `level` 0 corresponds to the same pixel size as the
        raw input series, while each increasing level doubles the pixel size
        used to create the output `ImageStack`.
    stride : int
        size of output `ImageStack` frames, as a multiple of `base_tile_size`.
        By default, 3.
    overlap : int
        overlap between spatially adjacent output frames, in units of 
        `base_tile_size`. By default, 1.
    output_name : str
        `PYME.IO.image.ImageStack` wrapping a `SupertileDatasource`.
    
    Notes
    -----
    The output of this module can be saved with `output.ImageOutput` as normal,
    however the raw pyramid tiles are saved in a temporary directory which will
    be automatically deleted at the end of the recipe.

    .. seealso:: modules :py:mod:`PYME.Analysis.tile_pyramid`.
    .. seealso:: modules :py:mod:`PYME.IO.Datasources.SupertileDatasource`.
    """

    input_name = Input('input')
    base_tile_size = Int(256)
    level = Int(0)
    stride = Int(3)
    overlap = Int(1)
    output_name = Output('supertile')

    def run(self, input_name):
        from PYME.IO.DataSources.SupertileDatasource import SupertileDataSource
        from PYME.IO.image import ImageStack
        from PYME.Analysis import tile_pyramid
        from tempfile import TemporaryDirectory
        
        warnings.warn(DeprecationWarning('Supertile will be removed, create a pyramid on disk and use a SUPERTILE URI against that instead'))
        logger.warning('Supertile is deprecated and will be removed, create a pyramid on disk and use a SUPERTILE URI against that instead')
            
        x, y = tile_pyramid.get_position_from_events(input_name.events, input_name.mdh)
        
        p = tile_pyramid.tile_pyramid(TemporaryDirectory(), input_name.data, x, y, 
                                      input_name.mdh, 
                                      pyramid_tile_size=self.base_tile_size)
            
        datasource = SupertileDataSource(p, self.level, self.stride, self.overlap)
        
        return ImageStack(data=datasource, 
                                                 mdh=datasource.mdh, 
                                                 haveGUI=False)


@register_module('TilePyramid')
class TilePyramid(OutputModule):
    """
    Create a tile pyramid from an input series containing x-y moves.
     
    Parameters
    ----------
    inputName : str
        `PYME.IO.image.ImageStack` of a tiled series.
    
    tile_size : int
        Size, in pixels, of the tiles in the pyramid. By default, 256
    
    filePattern : basestring
        a pattern through which the output filename is generated by variable substitution (using `str.format`)

    scheme : enum
        The storage method, one of 'File', 'pyme-cluster://' or 'pyme-cluster:// - aggregate`. File is the default
        and saves to a file on disk.
    
    Notes
    -----

    Requires events to determine frame x/y positions, so should be directly connected to input (events generally don't
    propagate within recipes). Most likely to be used in a simple, one-module, conversion recipe.

    **pyme-cluster awareness**

    Pyramid output is semi cluster-aware. Selecting a scheme of `pyme-cluster://` will save within the cluster root on the
    current machine, so that the results are accessible through the cluster. **NOTE:** this will **ONLY** work if the processing
    node is also running a dataserver [TODO - make this more robust] .

    `pyme-cluster:// - aggregate` is not supported.
    
    """
    
    inputName = Input('input')
    tile_size = Int(256)
    filePattern = '{output_dir}/{file_stub}_pyramid/'

    def save(self, namespace, context={}):
        """
        Save series as a tile pyramid

        Parameters
        ----------
        namespace : dict
            The recipe namespace
        context : dict
            Information about the source file to allow pattern substitution to generate the output name. At least
            'basedir' (which is the fully resolved directory name in which the input file resides) and
            'file_stub' (which is the filename without any extension) should be resolved.

        Returns
        -------

        """
        from PYME.Analysis import tile_pyramid
        from .output import _ensure_output_directory
        out_dirname = self._schemafy_filename(self.filePattern.format(**context))
        _ensure_output_directory(out_dirname)

        stack = namespace[self.inputName]
        x, y = tile_pyramid.get_position_from_events(stack.events, stack.mdh)

        tile_pyramid.tile_pyramid(out_dirname, stack.data, x, y,stack.mdh,pyramid_tile_size=self.tile_size)
    