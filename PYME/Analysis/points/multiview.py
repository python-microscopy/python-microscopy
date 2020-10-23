
import numpy as np
from PYME.Analysis.points.DeClump import pyDeClump

def load_shiftmap(uri):
    """
    helper function to handle I/O of two versions of shiftmaps. Note that HDF is prefered
    :param uri: str
        path or url to shiftmap-containing file (hdf, or [less ideal] json)
    :return: dict
        shiftmap
    """
    from PYME.IO import unifiedIO, tabular
    from PYME.IO.MetaDataHandler import HDFMDHandler
    import tables
    import json

    try:  # try loading shift map as hdf file
        with unifiedIO.local_or_temp_filename(uri) as f:
            t = tables.open_file(f)
            shift_map_source = tabular.HDFSource(t, 'shift_map')  # todo - is there a cleaner way to do this?
            shift_map_source.mdh = HDFMDHandler(t)

        # build dict of dicts so we can easily rebuild shiftfield objects in multiview.calc_shifts_for_points
        shift_map = {'shiftModel': shift_map_source.mdh['Multiview.shift_map.model']}
        legend = shift_map_source.mdh['Multiview.shift_map.legend']
        for l in legend.keys():
            keys = shift_map_source.keys()
            shift_map[l] = dict(zip(keys, [shift_map_source[k][legend[l]] for k in keys]))

        t.close()
    except tables.HDF5ExtError:  # file is probably saved as json (legacy)
        s = unifiedIO.read(uri)
        shift_map = json.loads(s)

    return shift_map


def coalesce_dict_sorted(inD, assigned, keys, weights_by_key,
                         discard_trivial=False):
    """
    Agregates clumps to a single event

    Parameters
    ----------
    inD : dict
        input dictionary containing fit results
    assigned : ndarray
        clump assignments to be coalesced. Cluster assignment index can start
        at 0 or 1 (the latter following PYME cluster labeling convention),
        however any index present will be coalesced (including the 0 cluster,
        if present).
    keys : list
        elements are strings corresponding to keys to be copied from the input
        to output dictionaries
    weights_by_key : dict
        maps weighting keys to coalescing weights or coalescing style (mean,
        min, sum).
    discard_trivial : bool
        by default, the output of aggregation/coalescing is indexable by the
        original clump ID and contains entries for the unassigned (0) clump
        and any other clumps which might have been lost to filtering. Setting
        discard_trial to true returns only those clumps for which  idx >=1 and
        which contain at least 1 point.

    Returns
    -------
    fres: dict
        coalesced results

    Notes
    -----
    This will evaluate the lazy pipeline events and add them into the dict as
    an array, not a code object.

    """
    from PYME.Analysis.points.DeClump import deClump

    NClumps = int(np.max(assigned) + 1)  # yes this is weird, but look at the C code

    clumped = {}
    
    if discard_trivial:
        non_trivial = np.unique(assigned[assigned >= 1]).astype('i')

    # loop through keys
    for rkey in keys:
        weights = weights_by_key.get(rkey, 'mean')

        if weights == 'mean':
            # if single value is given as weight, take an unweighted mean
            var = deClump.aggregateMean(NClumps, assigned.astype('i'), inD[rkey].astype('f'))
        elif weights == 'min':
            # if None has been passed as the weight for this key, take the minimum
            var = deClump.aggregateMin(NClumps, assigned.astype('i'), inD[rkey].astype('f'))
        elif weights == 'sum':
            var = deClump.aggregateSum(NClumps, assigned.astype('i'), inD[rkey].astype('f'))
        else:
            # if weights is an array, take weighted average
            var, errVec = deClump.aggregateWeightedMean(NClumps, assigned.astype('i'), inD[rkey].astype('f'), inD[weights].astype('f'))
            clumped[weights] = errVec[non_trivial] if discard_trivial else errVec

        if discard_trivial:
            clumped[rkey] = var[non_trivial]
        else:
            #default
            clumped[rkey] = var

    return clumped


def foldX(datasource, mdh, inject=False, chroma_mappings=False):
    """

    At this point the origin of x should be the corner of the concatenated frame

    Args:
        datasource:

    Returns: nothing
        Adds folded x-coordinates to the datasource
        Adds channel assignments to the datasource

    """
    from PYME.IO import tabular
    if not inject:
        datasource = tabular.MappingFilter(datasource)

    roiSizeNM = (mdh['Multiview.ROISize'][1]*mdh.voxelsize_nm.x)

    numChans = mdh.getOrDefault('Multiview.NumROIs', 1)
    color_chans = np.array(mdh.getOrDefault('Multiview.ChannelColor', np.zeros(numChans, 'i'))).astype('i')

    datasource.addVariable('roiSizeNM', roiSizeNM)
    datasource.addVariable('numChannels', numChans)

    # NB - this assumes that 'Multiview.ActiveViews' is sorted the same way that the views are concatenated (probably a safe assumption)
    active_rois = np.asarray(mdh.getOrDefault('Multiview.ActiveViews', 
                                              list(range(numChans))))
    multiview_channel = np.clip(np.floor(datasource['x'] / roiSizeNM), 
                                0, numChans - 1)
    multiview_channel = active_rois[multiview_channel.astype(int)]
    datasource.addColumn('multiviewChannel', multiview_channel)
    
    if chroma_mappings:
        datasource.addColumn('chromadx', 0 * datasource['x'])
        datasource.addColumn('chromady', 0 * datasource['y'])

        datasource.setMapping('x', 'x%roiSizeNM + chromadx')
        datasource.setMapping('y', 'y + chromady')
    else:
        datasource.setMapping('x', 'x%roiSizeNM')

    probe = color_chans[datasource['multiviewChannel']] #should be better performance
    datasource.addColumn('probe', probe)

    # add separate sigmaxy columns for each plane
    for chan in range(numChans):
        chan_mask = datasource['multiviewChannel'] == chan
        datasource.addColumn('chan%d' % chan, chan_mask)

        #mappings are cheap if we don't evaluate them
        datasource.setMapping('sigmax%d' % chan, 'chan%d*fitResults_sigmax' % chan)
        datasource.setMapping('sigmay%d' % chan, 'chan%d*fitResults_sigmay' % chan)
        datasource.setMapping('error_sigmax%d' % chan,
                                               'chan%(chan)d*fitError_sigmax - 1e4*(1-chan%(chan)d)' % {'chan': chan})
        datasource.setMapping('error_sigmay%d' % chan,
                                               'chan%(chan)d*fitError_sigmay - 1e4*(1-chan%(chan)d)' % {'chan': chan})

        #lets add some more that might be useful
        #datasource.setMapping('A%d' % chan, 'chan%d*A' % chan)

    return datasource

def correlative_shift(x0, y0, which_channel, pix_size_nm=115.):
    """
    Laterally shifts all channels to the first using cross correlations

    Parameters
    ----------
    x0 : ndarray
        array of localization x positions; not yet registered
    y0 : ndarray
        array of localization y positions; not yet registered
    which_channel : ndarray
        contains the channel ID for each localization
    pix_size_nm : float
        size of pixels to be used in generating 2D histograms which the correlations are then performed on

    Returns
    -------
    x : ndarray
        array of localization x positions registered to the first channel
    y : ndarray
        array of localization y positions registered to the first channel
    """
    from scipy.signal import fftconvolve  # , correlate2d
    from skimage import filters

    x, y = np.copy(x0), np.copy(y0)
    # determine number of ~pixel size bins for histogram
    bin_count = round((x.max() - x.min()) / pix_size_nm)  # assume square FOV (NB - after folding)
    # make sure bin_count is odd so its possible to have zero shift
    if bin_count % 2 == 0:
        bin_count += 1
    center = np.floor(float(bin_count) / 2)

    # generate first channel histogram
    channels = iter(np.unique(which_channel))
    first_chan = next(channels)
    mask = which_channel == first_chan
    first_channel, r_bins, c_bins = np.histogram2d(x[mask], y[mask], bins=(bin_count, bin_count))
    first_channel = first_channel >= filters.threshold_otsu(first_channel)
    first_channel = filters.gaussian(first_channel.astype(float))

    # loop through channels, skipping the first
    for chan in channels:
        # generate 2D histogram
        mask = which_channel == chan
        counts = np.histogram2d(x[mask], y[mask], bins=(r_bins, c_bins))[0]
        counts = counts >= filters.threshold_otsu(counts)
        counts = filters.gaussian(counts.astype(float))


        # cross-correlate this channel with the first, make it binary
        # cross_cor = correlate2d(first_channel, counts, mode='same', boundary='symm')
        cross_cor = fftconvolve(first_channel, counts[::-1, ::-1], mode='same')

        r_off, c_off = np.unravel_index(np.argmax(cross_cor), cross_cor.shape)

        # shift r and c positions
        r_shift = (center - r_off) * pix_size_nm
        c_shift = (center - c_off) * pix_size_nm

        x[mask] -= c_shift
        y[mask] -= r_shift

    return x, y

def pair_molecules(t_index, x0, y0, which_chan, delta_x=[None], appear_in=np.arange(4), n_frame_sep=5,
                   return_paired=True, pix_size_nm=115.):
    """
    pair_molecules uses pyDeClump functions to group localization clumps into molecules for registration.

    Parameters
    ----------
    t_index: from fitResults
    x0: ndarray
        x positions of localizations AFTER having been folded into the first channel
    y0: ndarray
        y positions of localizations
    which_chan: ndarray
        contains channel assignments for each localization
    delta_x: list
        distance within which neighbors will be clumped is set by 2*delta_x[i])**2. If None, will default to 100 nm
    appear_in: list
        a clump must have localizations in each of these channels in order to be a keep-clump
    n_frame_sep: int
        number of frames a molecule is allowed to blink off and still be clumped as the same molecule
    return_paired: bool
        flag to return a boolean array where True indicates that the molecule is a member of a
        clump whose members span the appearIn channels.

    Returns
    -------
    assigned: ndarray
        clump assignments for each localization. Note that molecules whose which_chan entry is set to a
        negative value will not be clumped, i.e. they will have a unique value in assigned.
    keep: ndarray
        a boolean vector encoding which molecules are in kept clumps

    Notes
    -----
    Outputs are of length #molecules, and the keep vector that is returned needs to be applied as:
    x_kept = x[keep] in order to only look at kept molecules.

    """
    # take out any large linear shifts for the sake of easier pairing
    x, y = correlative_shift(x0, y0, which_chan, pix_size_nm)
    # group within a certain distance, potentially based on localization uncertainty
    if not delta_x[0]:
        delta_x = 100.*np.ones_like(x)
    # group localizations
    assigned = pyDeClump.findClumps(t_index.astype(np.int32), x, y, delta_x, n_frame_sep)
    # print assigned.min()

    # only look at clumps with localizations from each channel
    clumps = np.unique(assigned)

    # Note that this will never be a keep clump if an ignore channel is present...
    kept_clumps = [np.array_equal(np.unique(which_chan[assigned == clumps[ii]]), appear_in) for ii in range(len(clumps))]

   # don't clump molecules from the wrong channel (done by parsing modified whichChan to this function)
    ignore_chan = which_chan < 0
    n_clump = np.max(assigned)
    ig_vec = np.arange(n_clump + 1, n_clump + 1 + sum(ignore_chan))
    # give ignored channel localizations unique clump assignments
    assigned[ignore_chan] = ig_vec

    if return_paired:
        kept_mols = []
        # TODO: speed up following loop - quite slow for large N
        for elem in assigned:
            kept_mols.append(elem in clumps[np.where(kept_clumps)])
        keep = np.where(kept_mols)
        return assigned, keep
    else:
        return assigned

def calc_shifts_for_points(datasource, shiftWallet):
    import importlib
    model = shiftWallet['shiftModel'].split('.')[-1]
    shiftModule = importlib.import_module(shiftWallet['shiftModel'].split('.' + model)[0])
    shiftModel = getattr(shiftModule, model)

    numChan = np.sum([(k.startswith('Chan') and k.endswith('.X')) for k in shiftWallet.keys()])

    x, y = datasource['x'], datasource['y']

    # FIXME: the camera roi positions below would not account for the multiview data source
    #x = x + pipeline.mdh['Camera.ROIX0']*pipeline.mdh['voxelsize.x']*1.0e3
    #y = y + pipeline.mdh['Camera.ROIY0']*pipeline.mdh['voxelsize.y']*1.0e3
    chan = datasource['multiviewChannel']

    dx = 0
    dy = 0
    for ii in range(1, numChan + 1):
        chanMask = chan == ii
        dx = dx + chanMask * shiftModel(dict=shiftWallet['Chan0%s.X' % ii]).ev(x, y)
        dy = dy + chanMask * shiftModel(dict=shiftWallet['Chan0%s.Y' % ii]).ev(x, y)

    return dx, dy

def apply_shifts_to_points(datasource, shiftWallet):  # FIXME: add metadata for camera roi positions
    """
    applyShiftmaps loads multiview shiftmap parameters from multiviewMapper.shiftWallet, reconstructs the shiftmap
    objects, applies them to the multiview data, and maps the positions registered to the first channel to the pipeline

    Args:
        x: vector of localization x-positions
        y: vector of localization y-positions
        numChan: number of multiview channels

    Returns: nothing
        Adds shifts into the pipeline which will then be applied automatically by the mappingFilter (see foldX)

    """
    dx, dy = calc_shifts_for_points(datasource, shiftWallet)

    datasource.addColumn('chromadx', dx)
    datasource.addColumn('chromady', dy)

    datasource.setMapping('x', 'x + chromadx')
    datasource.setMapping('y', 'y + chromady')


def find_clumps(datasource, gap_tolerance, radius_scale, radius_offset, inject=False):
    from PYME.Analysis.points.DeClump import deClump
    from PYME.IO import tabular
    t = datasource['t'] #OK as int
    clumps = np.zeros(len(t), 'i')
    I = np.argsort(t)
    t = t[I].astype('i')
    x = datasource['x'][I].astype('f4')
    y = datasource['y'][I].astype('f4')

    deltaX = (radius_scale*datasource['error_x'][I] + radius_offset).astype('f4')

    assigned = deClump.findClumpsN(t, x, y, deltaX, gap_tolerance)
    clumps[I] = assigned

    if not inject:
        datasource = tabular.MappingFilter(datasource)

    datasource.addColumn('clumpIndex', clumps)

    return datasource

def find_clumps_within_channel(datasource, gap_tolerance, radius_scale, radius_offset, inject=False):
    """

    Args:
        datasource: PYME datasource object - dictionary-like object with addColumn method
        gap_tolerance: number of frames acceptable for a molecule to go MIA and still be called the same molecule when
            it returns
        radius_scale: multiplicative factor applied to the error_x term in deciding search radius for pairing
        radius_offset: term added to radius_scale*error_x to set search radius

    Returns:
        Nothing, but adds clumpIndex column to datasource input
        
    FIXME: This function should probably not exist as channel handling should ideally only be in one place within the code base. A prefered solution would be to split using a colour filter, clump
    each channel separately, and then merge channels.

    """
    from PYME.Analysis.points.DeClump import deClump
    from PYME.IO import tabular
    t = datasource['t'] #OK as int
    clumps = np.zeros(len(t), 'i')
    I = np.argsort(t)
    t = t[I].astype('i')
    x = datasource['x'][I].astype('f4')
    y = datasource['y'][I].astype('f4')

    deltaX = (radius_scale*datasource['error_x'][I] + radius_offset).astype('f4')

    # extract color channel information
    uprobe = np.unique(datasource['probe'])
    probe = datasource['probe'][I]


    # only clump within color channels
    assigned = np.zeros_like(clumps)
    startAt = 0
    for pi in uprobe:
        pmask = probe == pi
        pClumps = deClump.findClumpsN(t[pmask], x[pmask], y[pmask], deltaX, gap_tolerance) + startAt
        # throw all unclumped into the 0th clumpID, and preserve pClumps[-1] of the last iteration
        pClumps[pClumps == startAt] = 0
        # patch in assignments for this color channel
        assigned[pmask] = pClumps
        startAt = np.max(assigned)
    clumps[I] = assigned

    if not inject:
        datasource = tabular.MappingFilter(datasource)

    datasource.addColumn('clumpIndex', clumps)

    return datasource

def merge_clumps(datasource, numChan, labelKey='clumpIndex'):
    from PYME.IO.tabular import DictSource

    keys_to_aggregate = ['x', 'y', 'z', 't', 'A', 'probe', 'tIndex', 'multiviewChannel', labelKey, 'focus', 'LLH']
    keys_to_aggregate += ['sigmax%d' % chan for chan in range(numChan)]
    keys_to_aggregate += ['sigmay%d' % chan for chan in range(numChan)]

    ds_keys = datasource.keys()
    keys_to_aggregate = [k for k in keys_to_aggregate if k in ds_keys] #discard any keys which are not in the underlying datasource

    all_keys = list(keys_to_aggregate) #this should be a copy otherwise we end up adding the weights to our list of stuff to aggregate

    # pair fit results and errors for weighting
    aggregation_weights = {k: 'error_' + k for k in keys_to_aggregate if 'error_' + k in datasource.keys()}
    all_keys += aggregation_weights.values()

    aggregation_weights['A'] = 'sum'

    I = np.argsort(datasource[labelKey])
    sorted_src = {k: datasource[k][I] for k in all_keys}

    grouped = coalesce_dict_sorted(sorted_src, sorted_src[labelKey], keys_to_aggregate, aggregation_weights, discard_trivial=True)
    return DictSource(grouped)



