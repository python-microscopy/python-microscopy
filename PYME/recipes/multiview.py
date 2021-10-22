from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float, CStr, Bool, Int, FileOrURI
import numpy as np
from PYME.IO import tabular
from PYME.IO import MetaDataHandler
from PYME.Analysis.points import multiview


@register_module('Fold')
class Fold(ModuleBase):
    """
    Fold localizations from images which have been taken with an image splitting device but analysed without channel
    awareness. Images taken in this fashion will have the channels side by side. This module folds the x co-ordinate to
    overlay the different channels, using the image metadata to determine the appropriate ROI boundaries.

    The current implementation is somewhat limited as it only handles folding along the x axis, and assumes that ROI
    sizes and spacings are completely uniform.

    Parameters
    ----------
    input_name : traits.Input
        localizations as PYME.IO.Tabular types

    Returns
    -------
    output_name : traits.Output
        localizations as PYME.IO.Tabular type with localizations folded

    Notes
    -----

    """
    input_name = Input('localizations')
    output_name = Output('folded')

    def execute(self, namespace):

        inp = namespace[self.input_name]

        if 'mdh' not in dir(inp):
            raise RuntimeError('Unfold needs metadata')

        mapped = multiview.foldX(inp, inp.mdh)
        mapped.mdh = inp.mdh

        namespace[self.output_name] = mapped


@register_module('ShiftCorrect')
class ShiftCorrect(ModuleBase):
    """
    Applies chromatic shift correction to folded localization data that was acquired with an image splitting device,
    but localized without splitter awareness.

    Parameters
    ----------
    input_name : traits.Input
        localizations as PYME.IO.Tabular types
    shift_map_path : traits.File
        file path or URL of shift map to be applied

    Returns
    -------
    output_name : traits.Output
        localizations as PYME.IO.Tabular type with clustered localizations typically denoted by the key 'clumpIndex'

    Notes
    -----
    """
    input_name = Input('folded')
    shift_map_path = FileOrURI('')
    output_name = Output('registered')

    def execute(self, namespace):

        inp = namespace[self.input_name]

        if 'mdh' not in dir(inp):
            raise RuntimeError('ShiftCorrect needs metadata')

        if self.shift_map_path == '':  # grab shftmap from the metadata
            loc = inp.mdh['Shiftmap']
        else:
            loc = self.shift_map_path

        shift_map = multiview.load_shiftmap(self.shift_map_path)

        mapped = tabular.MappingFilter(inp)

        multiview.apply_shifts_to_points(mapped, shift_map)
        # propagate metadata
        mapped.mdh = MetaDataHandler.NestedClassMDHandler(inp.mdh) #copy as we are going to modify
        mapped.mdh['Multiview.shift_map.location'] = loc

        namespace[self.output_name] = mapped


@register_module('FindClumps')
class FindClumps(ModuleBase):
    """

    Cluster localizations which, e.g. represent the same fluorophore, with the option to only clump localizations
    if they are in the same color channel.

    Parameters
    ----------
    input_name : traits.Input
        localizations as PYME.IO.Tabular types
    time_gap_tolerance : traits.Int
        Number of frames which a localizations is allowed to be missing and still be considered the same molecule if it
        reappears
    radius_scale : traits.Float
        Factor by which the localization precision is multiplied to determine the search radius for clustering. The
        default of 2 sigma means that we link ~95% of the points which should be linked (if Gaussian statistics hold)
    radius_offset : traits.Float
        Extra offset (in nanometers) for cases where we want to link localizations despite poor channel alignment
    probe_aware : traits.Bool
        If False, clumps molecules regardless of probe/colorchannel. If True, the key 'probe' should be present in the
        input datasource in order to determine which color channel a localization resides in.

    Returns
    -------
    output_name : traits.Output
        localizations as PYME.IO.Tabular type with clustered localizations typically denoted by the key 'clumpIndex'

    Notes
    -----

    """
    input_name = Input('registered')
    time_gap_tolerance = Int(1, desc='Number of off-frames allowed to still be a single clump')
    radius_scale = Float(2.0,
                        desc='Factor by which error_x is multiplied to detect clumps. The default of 2-sigma means we link ~95% of the points which should be linked')
    radius_offset = Float(0.,
                          desc='Extra offset (in nm) for cases where we want to link despite poor channel alignment')
    probe_aware = Bool(False, desc='''Use probe-aware clumping. NB this option does not work with standard methods of colour
                                             specification, and splitting by channel and clumping separately is preferred''')
    output_name = Output('clumped')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.input_name]

        if self.probe_aware and 'probe' in inp.keys():  # special case for using probe aware clumping NB this is a temporary fudge for non-standard colour handling
            mapped = multiview.find_clumps_within_channel(inp, self.time_gap_tolerance, self.radius_scale, self.radius_offset)
        else:  # default
            mapped = multiview.find_clumps(inp, self.time_gap_tolerance, self.radius_scale, self.radius_offset)

        if 'mdh' in dir(inp):
            mapped.mdh = inp.mdh

        namespace[self.output_name] = mapped


@register_module('MergeClumps')
class MergeClumps(ModuleBase):
    """

    Coalesces paired localizations that appeared on the same frame as determined by multiview.FindClumps

    Parameters
    ----------
    input_name : traits.Input
        localizations as PYME.IO.Tabular types
    labelKey : traits.CStr
        key to column of tabular input which contains clump ID of each localization. Localizations sharing the same
        clump ID will be coalesced.

    Returns
    -------
    output_name : traits.Output
        localizations as PYME.IO.Tabular type with clustered localizations coalesced into individual localizations

    Notes
    -----

    """
    input_name = Input('clumped')
    output_name = Output('merged')
    labelKey = CStr('clumpIndex')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.input_name]

        try:
            grouped = multiview.merge_clumps(inp, inp.mdh.getOrDefault('Multiview.NumROIs', 0), labelKey=self.labelKey)
            grouped.mdh = inp.mdh
        except AttributeError:
            grouped = multiview.merge_clumps(inp, numChan=0, labelKey=self.labelKey)

        namespace[self.output_name] = grouped


@register_module('MapAstigZ')
class MapAstigZ(ModuleBase):
    """

    Uses astigmatism calibration (widths of PSF along each dimension as a function of z) to determine the z-position of
    localizations relative to the focal plane of the frame during which it was imaged.

    Parameters
    ----------
    input_name : traits.Input
        localizations as PYME.IO.Tabular types
    astigmatism_calibration_location : traits.File
        file path or URL to astigmatism calibration file

    Returns
    -------
    output_name : traits.Output
        output is a json wrapped by PYME.IO.ragged.RaggedCache

    Notes
    -----

    """
    input_name = Input('merged')

    astigmatism_calibration_location = FileOrURI('')
    rough_knot_spacing = Float(50.)

    output_name = Output('zmapped')

    def execute(self, namespace):
        from PYME.Analysis.points.astigmatism import astigTools
        from PYME.IO import unifiedIO
        import json

        inp = namespace[self.input_name]

        if 'mdh' not in dir(inp):
            raise RuntimeError('MapAstigZ needs metadata')

        if self.astigmatism_calibration_location == '':  # grab calibration from the metadata
            calibration_location = inp.mdh['Analysis.AstigmatismMapID']
        else:
            calibration_location = self.astigmatism_calibration_location

        s = unifiedIO.read(calibration_location)

        astig_calibrations = json.loads(s)

        mapped = tabular.MappingFilter(inp)

        z, zerr = astigTools.lookup_astig_z(mapped, astig_calibrations, self.rough_knot_spacing, plot=False)

        mapped.addColumn('astigmatic_z', z)
        mapped.addColumn('astigmatic_z_lookup_error', zerr)
        mapped.setMapping('z', 'astigmatic_z + z')

        mapped.mdh = MetaDataHandler.NestedClassMDHandler(inp.mdh)
        mapped.mdh['Analysis.astigmatism_calibration_used'] = calibration_location

        namespace[self.output_name] = mapped

# ---------- calibration generation ----------

@register_module('CalibrateShifts')
class CalibrateShifts(ModuleBase):
    """

    Generates multiview shiftmaps from (folded) bead-data localizations. Only beads which show up in all channels are
    used to generate the shiftmaps

    Parameters
    ----------
    input_name : traits.Input
        localizations as PYME.IO.Tabular types
    search_radius_nm : traits.Float
        radius within which bead localizations should be clumped if the bead appears in all channels. Units of nm.

    Returns
    -------
    output_name : traits.Output
        output is a PYME.IO.tabular type with some relevant information stored in the metadata

    Notes
    -----

    """
    input_name = Input('folded')

    search_radius_nm = Float(250.)

    output_name = Output('shiftmap')

    def execute(self, namespace):
        from PYME.Analysis.points import twoColour
        from PYME.Analysis.points import multiview
        from PYME.IO.MetaDataHandler import NestedClassMDHandler

        inp = namespace[self.input_name]

        try:  # make sure we're looking at multiview data
            n_chan = inp.mdh['Multiview.NumROIs']
        except AttributeError:
            raise AttributeError('multiview metadata is missing or incomplete')

        # sort in frame order
        I = inp['tIndex'].argsort()
        x_sort, y_sort = inp['x'][I], inp['y'][I]
        chan_sort = inp['multiviewChannel'][I]

        clump_id, keep = multiview.pair_molecules(inp['tIndex'][I], x_sort, y_sort, chan_sort,
                                                  self.search_radius_nm * np.ones_like(x_sort),
                                                  appear_in=np.arange(n_chan), n_frame_sep=inp['tIndex'].max(),
                                                  pix_size_nm=inp.mdh.voxelsize_nm.x)

        # only look at the clumps which showed up in all channels
        x = x_sort[keep]
        y = y_sort[keep]
        chan = chan_sort[keep]
        clump_id = clump_id[keep]

        # Generate raw shift vectors (map of displacements between channels) for each channel
        mol_list = np.unique(clump_id)
        n_mols = len(mol_list)
        if n_mols < 3:
            raise ValueError('Need at 3 clusters containing points from each channel - try increasing search radius')

        dx = np.zeros((n_chan - 1, n_mols))
        dy = np.zeros_like(dx)
        dx_err = np.zeros_like(dx)
        dy_err = np.zeros_like(dx)
        x_clump, y_clump, x_std, y_std, x_shifted, y_shifted = [], [], [], [], [], []

        shift_map_dtype = [('mx', '<f4'), ('mx2', '<f4'), ('mx3', '<f4'),  # x terms
                           ('my', '<f4'), ('my2', '<f4'), ('my3', '<f4'),  # y terms
                           ('mxy', '<f4'), ('mx2y', '<f4'), ('mxy2', '<f4'),  # cross terms
                           ('x0', '<f4')]  # 0th order shift

        shift_maps = np.zeros(2*(n_chan - 1), dtype=shift_map_dtype)
        mdh = NestedClassMDHandler(inp.mdh)
        mdh['Multiview.shift_map.legend'] = {}

        for ii in range(n_chan):
            chan_mask = (chan == ii)
            x_chan = np.zeros(n_mols)
            y_chan = np.zeros(n_mols)
            x_chan_std = np.zeros(n_mols)
            y_chan_std = np.zeros(n_mols)

            for ind in range(n_mols):
                # merge clumps within channels
                clump_mask = np.where(np.logical_and(chan_mask, clump_id == mol_list[ind]))
                x_chan[ind] = x[clump_mask].mean()
                y_chan[ind] = y[clump_mask].mean()
                x_chan_std[ind] = x[clump_mask].std()
                y_chan_std[ind] = y[clump_mask].std()

            x_clump.append(x_chan)
            y_clump.append(y_chan)
            x_std.append(x_chan_std)
            y_std.append(y_chan_std)

            if ii > 0:
                dx[ii - 1, :] = x_clump[0] - x_clump[ii]
                dy[ii - 1, :] = y_clump[0] - y_clump[ii]
                dx_err[ii - 1, :] = np.sqrt(x_std[ii] ** 2 + x_std[0] ** 2)
                dy_err[ii - 1, :] = np.sqrt(y_std[ii] ** 2 + y_std[0] ** 2)
                # generate shiftmap between ii-th channel and the 0th channel
                dxx, dyy, spx, spy, good = twoColour.genShiftVectorFieldQ(x_clump[0], y_clump[0], dx[ii - 1, :],
                                                                          dy[ii - 1, :], dx_err[ii - 1, :],
                                                                          dy_err[ii - 1, :])
                # store shiftmaps in structured array
                mdh['Multiview.shift_map.legend']['Chan0%s.X' % ii] = 2*(ii - 1)
                mdh['Multiview.shift_map.legend']['Chan0%s.Y' % ii] = 2*(ii - 1) + 1
                for ki in range(len(shift_map_dtype)):
                    k = shift_map_dtype[ki][0]
                    shift_maps[2*(ii - 1)][k] = spx.__getattribute__(k)
                    shift_maps[2*(ii - 1) + 1][k] = spy.__getattribute__(k)


                # shift_maps['Chan0%s.X' % ii], shift_maps['Chan0%s.Y' % ii] = spx.__dict__, spy.__dict__

        mdh['Multiview.shift_map.model'] = '.'.join([spx.__class__.__module__, spx.__class__.__name__])

        namespace[self.output_name] = tabular.RecArraySource(shift_maps)
        namespace[self.output_name].mdh = mdh


@register_module('ExtractMultiviewChannel')
class ExtractMultiviewChannel(ModuleBase):
    """Extract a single multiview channel

    Parameters
    ----------
    input_name : PYME.IO.image.ImageStack
        input, with multiview metadata
    view_number : int
        which multiview view to extract for the new ImageStack. Number should
        match the multiview ROI number, not the number within the subset of
        active views. By default, 0
    output_name : PYME.IO.image.ImageStack
        image cropped to contain a single multiview channel
    
    Notes
    -----
    Multiview metadata of the output image will not be updated other than to
    note which channel has been extracted. All downstream analyses should be
    not be multiview specific.
    
    """
    input_name = Input('input')
    view_number = Int(0)
    output_name = Output('extracted')

    def execute(self, namespace):
        from PYME.IO.DataSources.CropDataSource import _DataSource as DataSource #TODO - fix to use new crop data source
        from PYME.IO.MetaDataHandler import DictMDHandler
        from PYME.IO.image import ImageStack

        source = namespace[self.input_name]
        roi_size = source.mdh['Multiview.ROISize']
        ind = np.argwhere(np.asarray(source.mdh['Multiview.ActiveViews']) == self.view_number)[0][0]
        x_i, x_f = int(ind * roi_size[0]), int((ind + 1 ) * roi_size[0])
        extracted = DataSource(source.data, (x_i, x_f))

        mdh = DictMDHandler(source.mdh)
        mdh['Multiview.Extracted'] = ind
        namespace[self.output_name] = ImageStack(data=extracted, mdh=mdh, 
                                                 events=source.events)
