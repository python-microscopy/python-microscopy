
from .base import ModuleBase, register_module, OutputModule
from .traits import Input, Output, CStr, Int
from PYME.IO import tabular
import logging

logger = logging.getLogger(__name__)

@register_module('TilePhysicalCoords')
class TilePhysicalCoords(ModuleBase):
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
    
    def execute(self, namespace):
        meas = namespace[self.input_measurements]
        img = namespace[self.input_supertile]
        
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
        
        out.mdh = meas.mdh
        
        namespace[self.output_name] = out


@register_module('Supertile')
class Supertile(ModuleBase):
    """
    FIXME: This should be 2 separate modules (pyramid creation and supertile creation)
    WARNING: This module will likely dissappear / change extensively. Do not expect this
    to be present in future versions of the code.
    
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

    def execute(self, namespace):
        from PYME.IO.DataSources.SupertileDatasource import SupertileDataSource
        from PYME.IO.image import ImageStack
        from PYME.Analysis import tile_pyramid
        from tempfile import TemporaryDirectory
        
        stack = namespace[self.input_name]
            
        x, y = tile_pyramid.get_position_from_events(stack.events, stack.mdh)
        
        p = tile_pyramid.tile_pyramid(TemporaryDirectory(), stack.data, x, y, 
                                      stack.mdh, 
                                      pyramid_tile_size=self.base_tile_size)
            
        datasource = SupertileDataSource(p, self.level, self.stride, self.overlap)
        
        namespace[self.output_name] = ImageStack(data=datasource, 
                                                 mdh=datasource.mdh, 
                                                 haveGUI=False)
