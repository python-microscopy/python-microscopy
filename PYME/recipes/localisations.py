from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr

import numpy as np
import pandas as pd
from PYME.IO import tabular
from PYME.LMVis import renderers


@register_module('ExtractTableChannel')
class ExtractTableChannel(ModuleBase):
    """Create and return a ColourFilter which has filtered out one colour channel from a table of localizations."""
    inputName = Input('measurements')
    channel = CStr('everything')
    outputName = Output('filtered')

    def execute(self, namespace):
        inp = namespace[self.inputName]

        map = tabular.colourFilter(inp, currentColour=self.channel)

        if 'mdh' in dir(inp):
            map.mdh = inp.mdh

        namespace[self.outputName] = map

    @property
    def _colour_choices(self):
        #try and find the available column names
        try:
            return tabular.colourFilter.get_colour_chans(self._parent.namespace[self.inputName])
        except:
            return []

    @property
    def pipeline_view(self):
        from traitsui.api import View, Group, Item
        from PYME.ui.custom_traits_editors import CBEditor

        modname = ','.join(self.inputs) + ' -> ' + self.__class__.__name__ + ' -> ' + ','.join(self.outputs)

        return View(Group(Item('channel', editor=CBEditor(choices=self._colour_choices)), label=modname))

    @property
    def default_view(self):
        from traitsui.api import View, Group, Item
        from PYME.ui.custom_traits_editors import CBEditor

        return View(Item('inputName', editor=CBEditor(choices=self._namespace_keys)),
                    Item('_'),
                    Item('channel', editor=CBEditor(choices=self._colour_choices)),
                    Item('_'),
                    Item('outputName'), buttons=['OK'])


@register_module('DensityMapping')
class DensityMapping(ModuleBase):
    """ Use density estimation methods to generate an image from localizations

     """
    inputLocalizations = Input('localizations')
    outputImage = Output('output')
    renderingModule = Enum(renderers.RENDERERS.keys())

    pixelSize = Float(5)
    jitterVariable = CStr('1.0')
    jitterScale = Float(1.0)
    jitterVariableZ = CStr('1.0')
    jitterScaleZ = Float(1.0)
    MCProbability = Float(1.0)
    numSamples = Int(10)
    colours = List(['none'])
    zBounds = ListFloat([-500, 500])
    zSliceThickness = Float(50.0)
    softRender = Bool(True)

    def execute(self, namespace):
        from PYME.IO.image import ImageBounds
        inp = namespace[self.inputLocalizations]
        if not isinstance(inp, tabular.colourFilter):
            cf = tabular.colourFilter(inp, None)
            cf.mdh = inp.mdh
        else:
            cf = inp

        cf.imageBounds = ImageBounds.estimateFromSource(inp)

        renderer = renderers.RENDERERS[str(self.renderingModule)](None, cf)

        namespace[self.outputImage] = renderer.Generate(self.get())

@register_module('AddPipelineDerivedVars')
class Pipelineify(ModuleBase):
    inputFitResults = Input('FitResults')
    inputDriftResults = Input('')
    inputEvents = Input('')
    outputLocalizations = Output('localizations')

    pixelSizeNM = Float(1)


    def execute(self, namespace):
        from PYME.LMVis import pipeline
        fitResults = namespace[self.inputFitResults]
        mdh = fitResults.mdh

        mapped_ds = tabular.mappingFilter(fitResults)


        if not self.pixelSizeNM == 1: # TODO - check close instead?
            mapped_ds.addVariable('pixelSize', self.pixelSizeNM)
            mapped_ds.setMapping('x', 'x*pixelSize')
            mapped_ds.setMapping('y', 'y*pixelSize')

        #extract information from any events
        events = namespace.get(self.inputEvents, None)
        if isinstance(events, tabular.TabularBase):
            events = events.to_recarray()

        ev_maps, ev_charts = pipeline._processEvents(mapped_ds, events, mdh)
        pipeline._add_missing_ds_keys(mapped_ds, ev_maps)

        #Fit module specific filter settings
        if 'Analysis.FitModule' in mdh.getEntryNames():
            fitModule = mdh['Analysis.FitModule']

            if 'LatGaussFitFR' in fitModule:
                mapped_ds.addColumn('nPhotons', pipeline.getPhotonNums(mapped_ds, mdh))

        mapped_ds.mdh = mdh

        namespace[self.outputLocalizations] = mapped_ds


@register_module('MultiviewFold') #FIXME - move to multi-view specific module and potentially rename
class MultiviewFold(ModuleBase):
    """Fold localizations from images which have been taken with an image splitting device but analysed without channel
    awareness.

    Images taken in this fashion will have the channels side by side. This module folds the x co-ordinate to overlay the
    different channels, using the image metadata to determine the appropriate ROI boundaries. The current implementation
    is somewhat limited as it only handles folding along the x axis, and assumes that ROI sizes and spacings are completely
    uniform.
    """
    inputName = Input('localizations')
    outputName = Output('folded')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.inputName]

        if 'mdh' not in dir(inp):
            raise RuntimeError('Unfold needs metadata')

        mapped = multiview.foldX(inp, inp.mdh)
        mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped


@register_module('MultiviewShiftCorrect') #FIXME - move to multi-view specific module and rename OR make consistent with existing shift correction
class MultiviewShiftCorrect(ModuleBase):
    """Applies chromatic shift correction to folded localization data that was acquired with an image splitting device,
    but localized without splitter awareness."""
    inputName = Input('folded')
    shiftMapLocation = CStr('') #FIXME - change name to indicate that this is a filename/path/URL. Should probably be a File trait (or derived class which deals with clusterIO)
    outputName = Output('registered')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview
        from PYME.IO import unifiedIO
        import json

        inp = namespace[self.inputName]

        if 'mdh' not in dir(inp):
            raise RuntimeError('ShiftCorrect needs metadata')

        if self.shiftMapLocation == '':  # grab shftmap from the metadata
            s = unifiedIO.read(inp.mdh['Shiftmap'])
        else:
            s = unifiedIO.read(self.shiftMapLocation)

        shiftMaps = json.loads(s)

        mapped = tabular.mappingFilter(inp)

        dx, dy = multiview.calcShifts(mapped, shiftMaps)
        mapped.addColumn('chromadx', dx)
        mapped.addColumn('chromady', dy)

        mapped.setMapping('x', 'x + chromadx')
        mapped.setMapping('y', 'y + chromady')

        mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped


@register_module('FindClumps') #FIXME - move to multi-view specific module and rename OR make consistent with existing clumping
class FindClumps(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = Input('registered')
    gapTolerance = Int(1, desc='Number of off-frames allowed to still be a single clump')
    radiusScale = Float(2.0)
    radius_offset_nm = Float(150., desc='[nm]')
    probeAwareClumping = Bool(False, desc='''Use probe-aware clumping. NB this option does not work with standard methods of colour 
                                             specification, and splitting by channel and clumping separately is preferred''') 
    outputName = Output('clumped')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.inputName]

        if self.probeAwareClumping and 'probe' in inp.keys(): #special case for using probe aware clumping NB this is a temporary fudge for non-standard colour handling
            mapped = multiview.probeAwareFindClumps(inp, self.gapTolerance, self.radiusScale, self.radius_offset_nm)
        else: #default
            mapped = multiview.findClumps(inp, self.gapTolerance, self.radiusScale, self.radius_offset_nm)
        
        if 'mdh' in dir(inp):
            mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped


@register_module('MergeClumps') #FIXME - move to multi-view specific module and rename OR make consistent with existing clumping
class MergeClumps(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = Input('clumped')
    outputName = Output('merged')
    labelKey = CStr('clumpIndex')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.inputName]

        #mapped = tabular.mappingFilter(inp)

        if 'mdh' not in dir(inp):
            raise RuntimeError('MergeClumps needs metadata')

        grouped = multiview.mergeClumps(inp, inp.mdh.getOrDefault('Multiview.NumROIs', 0), labelKey=self.labelKey)

        grouped.mdh = inp.mdh

        namespace[self.outputName] = grouped


@register_module('MapAstigZ') #FIXME - move to multi-view specific module and rename
class MapAstigZ(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = Input('merged')
    astigmatismMapLocation = CStr('') #FIXME - rename and possibly change type
    outputName = Output('zmapped')

    def execute(self, namespace):
        from PYME.Analysis.points.astigmatism import astigTools
        from PYME.IO import unifiedIO
        import json

        inp = namespace[self.inputName]

        if 'mdh' not in dir(inp):
            raise RuntimeError('MapAstigZ needs metadata')

        if self.astigmatismMapLocation == '':  # grab calibration from the metadata
            s = unifiedIO.read(inp.mdh['Analysis.AstigmatismMapID'])
        else:
            s = unifiedIO.read(self.astigmatismMapLocation)

        astig_calibrations = json.loads(s)

        mapped = tabular.mappingFilter(inp)

        z, zerr = astigTools.lookup_astig_z(mapped, astig_calibrations, plot=False)

        mapped.addColumn('astigZ', z)
        mapped.addColumn('zLookupError', zerr)
        mapped.setMapping('z', 'astigZ + z')

        mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped

@register_module('IDTransientFrames')
class IDTransientFrames(ModuleBase): #FIXME - move to multi-view specific module and potentially rename (depending on whether we introduce scoping)
    """
    Adds an 'isTransient' column to the input datasource so that one can filter localizations that are from frames
    acquired during z-translation
    """
    inputName = Input('zmapped')
    inputEvents = Input('Events')
    framesPerStep = Float()
    outputName = Output('transientFiltered')

    def execute(self, namespace):
        from PYME.experimental import zMotionArtifactUtils

        inp = namespace[self.inputName]

        mapped = tabular.mappingFilter(inp)

        if 'mdh' not in dir(inp):
            if self.framesPerStep <= 0:
                raise RuntimeError('idTransientFrames needs metadata')
            else:
                fps = self.framesPerStep
        else:
            fps = inp.mdh['StackSettings.FramesPerStep']

        mask = zMotionArtifactUtils.flagMotionArtifacts(mapped, namespace[self.inputEvents], fps)
        mapped.addColumn('piezoUnstable', mask)

        mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped

@register_module('DBSCANClustering')
class DBSCANClustering(ModuleBase):
    """
    Performs DBSCAN clustering on input dictionary

    Parameters
    ----------

        searchRadius: search radius for clustering
        minPtsForCore: number of points within SearchRadius required for a given point to be considered a core point

    Notes
    -----

    See `sklearn.cluster.dbscan` for more details about the underlying algorithm and parameter meanings.

    """
    inputName = Input('filtered')

    columns = ListStr(['x', 'y', 'z'])
    searchRadius = Float()
    minClumpSize = Int()

    outputName = Output('dbscanClustered')

    def execute(self, namespace):
        from sklearn.cluster import dbscan

        inp = namespace[self.inputName]
        mapped = tabular.mappingFilter(inp)

        # Note that sklearn gives unclustered points label of -1, and first value starts at 0.
        core_samp, dbLabels = dbscan(np.vstack([inp[k] for k in self.columns]).T,
                                     self.searchRadius, self.minClumpSize)

        # shift dbscan labels up by one to match existing convention that a clumpID of 0 corresponds to unclumped
        mapped.addColumn('dbscanClumpID', dbLabels + 1)

        # propogate metadata, if present
        try:
            mapped.mdh = inp.mdh
        except AttributeError:
            pass

        namespace[self.outputName] = mapped

    @property
    def hide_in_overview(self):
        return ['columns']

@register_module('ClusterCountVsImagingTime')
class ClusterCountVsImagingTime(ModuleBase):
    """
    ClusterCountVsImagingTime iteratively filters a dictionary-like object on t, and at each step counts the number of
    labeled objects (e.g. DBSCAN clusters) which contain at least N-points. It does this for two N-points, so one can be
    set according to density with all frames included, and the other can be set for one of the earlier frame-counts.

    args:
        stepSize: number of frames to add in on each iteration
        labelsKey: key containing labels for each localization
        lowerMinPtsPerCluster:
        higherMinPtsPerCluster:

    returns:
        dictionary-like object with the following keys:
            t: upper bound on frame number included in calculations on each iteration.
            N_labelsWithLowMinPoints:
            N_labelsWithHighMinPoints:

    From wikipedia: "While minPts intuitively is the minimum cluster size, in some cases DBSCAN can produce smaller
    clusters. A DBSCAN cluster consists of at least one core point. As other points may be border points to more than
    one cluster, there is no guarantee that at least minPts points are included in every cluster."
    """
    inputName = Input('input')

    labelsKey = CStr('dbscanClumpID')
    lowerMinPtsPerCluster = Int(3)
    higherMinPtsPerCluster = Int(6)
    stepSize = Int(3000)

    outputName = Output('incremented')

    def execute(self, namespace):
        from PYME.IO import tabular

        if self.lowerMinPtsPerCluster > self.higherMinPtsPerCluster:
            print('Swapping low and high MinPtsPerCluster - input was reversed')
            temp = self.lowerMinPtsPerCluster
            self.lowerMinPtsPerCluster = self.higherMinPtsPerCluster
            self.higherMinPtsPerCluster = temp

        iters = (int(np.max(namespace[self.inputName]['t']))/int(self.stepSize)) + 2

        # other counts
        lowDensMinPtsClumps = np.empty(iters)
        lowDensMinPtsClumps[0] = 0
        hiDensMinPtsClumps = np.empty(iters)
        hiDensMinPtsClumps[0] = 0
        t = np.empty(iters)
        t[0] = 0

        inp = tabular.mappingFilter(namespace[self.inputName])

        for ind in range(1, iters):  # start from 1 since t=[0,0] will yield no clumps
            # filter time
            inc = tabular.resultsFilter(inp, t=[0, self.stepSize*ind])
            t[ind] = np.max(inc['t'])

            cid, counts = np.unique(inc[self.labelsKey], return_counts=True)
            # cmask = np.in1d(inc['DBSCAN_allFrames'], cid)

            cidL = cid[counts >= self.lowerMinPtsPerCluster]
            lowDensMinPtsClumps[ind] = np.sum(cidL != -1)  # ignore unclumped in count
            cid = cid[counts >= self.higherMinPtsPerCluster]
            hiDensMinPtsClumps[ind] = np.sum(cid != -1)  # ignore unclumped in count


        res = tabular.resultsFilter({'t': t,
                                     'N_labelsWithLowMinPoints': lowDensMinPtsClumps,
                                     'N_labelsWithHighMinPoints': hiDensMinPtsClumps})

        # propagate metadata, if present
        try:
            res.mdh = namespace[self.inputName].mdh
        except AttributeError:
            pass

        namespace[self.outputName] = res



@register_module('RadiusOfGyration')
class RadiusOfGyration(ModuleBase):
    """
    Parameters
    ----------

        labelsKey: array of label assignments. Radius of gyration will be calculated for each label.

    Notes
    -----

    """
    inputName = Input('dbscanClustered')
    labelsKey = CStr('dbscanClumpID')
    outputName = Output('gyrationRadii')

    def execute(self, namespace):

        inp = namespace[self.inputName]

        # sort input according to label key
        all_keys = inp.keys()
        I = np.argsort(inp[self.labelsKey])
        mapped = tabular.mappingFilter({k: inp[k][I] for k in all_keys})
        uni, counts = np.unique(mapped[self.labelsKey], return_counts=True)

        numLabs = len(uni)
        rg = np.empty(numLabs, dtype=float)
        # loop over labels, recall that input is now sorted, and we know how many points are in each label
        indi = 0
        for li in range(numLabs):
            indf = indi + counts[li]
            x, y, z = mapped['x'][indi:indf], mapped['y'][indi:indf], mapped['z'][indi:indf]
            com = np.array([x.mean(), y.mean(), z.mean()])
            disp = np.linalg.norm(np.hstack([x, y, z]) - com, axis=0)
            rg[li] = np.sqrt((1/float(counts[li]))*np.sum(disp**2))
            indi = indf

        mapped.addColumn('GyrationRadius', rg)

        try:
            mapped.mdh = namespace[self.inputName].mdh
        except AttributeError:
            pass

        namespace[self.outputName] = mapped
