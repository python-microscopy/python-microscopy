import numpy as np

def coalesceDictSorted(inD, assigned, keys, weights_by_key):  # , notKosher=None):
    """
    Agregates clumps to a single event
    Note that this will evaluate the lazy pipeline events and add them into the dict as an array, not a code
    object.
    Also note that copying a large dictionary can be rather slow, and a structured ndarray approach may be preferable.
    DB - we should never have a 'large' dictionary (ie there will only ever be a handful of keys)

    Args:
        inD: input dictionary containing fit results
        assigned: clump assignments to be coalesced
        keys: list whose elements are strings corresponding to keys to be copied from the input to output dictionaries
        weights_by_key: dictionary of weights.

    Returns:
        fres: output dictionary containing the coalesced results

    """
    from PYME.Analysis.points.DeClump import deClump

    NClumps = int(np.max(assigned) + 1)  # len(np.unique(assigned))  #

    clumped = {}

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
            clumped[weights] = errVec

        clumped[rkey] = var

    return clumped


def foldX(datasource, mdh):
    """

    At this point the origin of x should be the corner of the concatenated frame

    Args:
        datasource:

    Returns: nothing
        Adds folded x-coordinates to the datasource
        Adds channel assignments to the datasource

    """
    roiSizeNM = (mdh['Multiview.ROISize'][1]*mdh['voxelsize.x']*1000)  # voxelsize is in um

    numChans = mdh.getOrDefault('Multiview.NumROIs', 1)
    color_chans = np.array(mdh.getOrDefault('Multiview.ChannelColor', np.zeros(numChans, 'i'))).astype('i')

    datasource.addVariable('roiSizeNM', roiSizeNM)
    datasource.addVariable('numChannels', numChans)

    datasource.addColumn('chromadx', 0*datasource['x'])
    datasource.addColumn('chromady', 0*datasource['y'])

    #FIXME - cast to int should probably happen when we use multiViewChannel, not here (because we might have saved and reloaded in between)
    datasource.setMapping('multiviewChannel', 'clip(floor(x/roiSizeNM), 0, numChannels - 1).astype(int)')
    datasource.setMapping('x', 'x%roiSizeNM + chromadx')
    datasource.setMapping('y', 'y + chromady')

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

def applyShiftmaps(datasource, shiftWallet):  # FIXME: add metadata for camera roi positions
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
        dx = dx + chanMask*shiftModel(dict=shiftWallet['Chan0%s.X' % ii]).ev(x, y)
        dy = dy + chanMask*shiftModel(dict=shiftWallet['Chan0%s.Y' % ii]).ev(x, y)

    datasource.addColumn('chromadx', dx)
    datasource.addColumn('chromady', dy)


def findClumps(datasource, gap_tolerance, radius_scale, radius_offset):
    from PYME.Analysis.points.DeClump import deClump
    t = datasource['t'] #OK as int
    clumps = np.zeros(len(t), 'i')
    I = np.argsort(t)
    t = t[I].astype('i')
    x = datasource['x'][I].astype('f4')
    y = datasource['y'][I].astype('f4')

    deltaX = (radius_scale*datasource['error_x'][I] + radius_offset).astype('f4')

    assigned = deClump.findClumpsN(t, x, y, deltaX, gap_tolerance)
    clumps[I] = assigned

    datasource.addColumn('clumpIndex', clumps)


def mergeClumps(datasource, numChan):
    from PYME.IO.tabular import cachingResultsFilter

    keys_to_aggregate = ['x', 'y', 'z', 't', 'A', 'probe', 'tIndex', 'multiviewChannel', 'clumpIndex']
    keys_to_aggregate += ['sigmax%d' % chan for chan in range(numChan)]
    keys_to_aggregate += ['sigmay%d' % chan for chan in range(numChan)]

    all_keys = list(keys_to_aggregate) #this should be a copy otherwise we end up adding the weights to our list of stuff to aggregate

    # pair fit results and errors for weighting
    aggregation_weights = {k: 'error_' + k for k in keys_to_aggregate if 'error_' + k in datasource.keys()}
    all_keys += aggregation_weights.values()

    aggregation_weights['A'] = 'sum'

    I = np.argsort(datasource['clumpIndex'])
    sorted_src = {k: datasource[k][I] for k in all_keys}

    grouped = coalesceDictSorted(sorted_src, sorted_src['clumpIndex'], keys_to_aggregate, aggregation_weights)
    return cachingResultsFilter(grouped)




