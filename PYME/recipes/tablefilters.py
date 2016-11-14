from .base import register_module, ModuleBase, Filter, Float, Enum, CStr, Bool, Int, View, Item, List#, Group
from traits.api import DictStrStr, DictStrList, ListFloat
import numpy as np
import pandas as pd
from PYME.IO import tabular
from PYME.LMVis import renderers

@register_module('Mapping')
class Mapping(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = CStr('measurements')
    mappings = DictStrStr()
    outputName = CStr('mapped')

    def execute(self, namespace):
        inp = namespace[self.inputName]

        map = tabular.mappingFilter(inp, **self.mappings)

        if 'mdh' in dir(inp):
            map.mdh = inp.mdh

        namespace[self.outputName] = map


@register_module('Filter')
class Filter(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = CStr('measurements')
    filters = DictStrList()
    outputName = CStr('filtered')

    def execute(self, namespace):
        inp = namespace[self.inputName]

        map = tabular.resultsFilter(inp, **self.filters)

        if 'mdh' in dir(inp):
            map.mdh = inp.mdh

        namespace[self.outputName] = map

@register_module('DensityMapping')
class DensityMapping(ModuleBase):
    """ Use density estimation methods to generate an image from localizations - or more specifically a colour filter"""
    inputLocalizations = CStr('localizations')
    outputImage = CStr('output')
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
    inputFitResults = CStr('FitResults')
    inputDriftResults = CStr('')
    inputEvents = CStr('')
    outputLocalizations = CStr('localizations')

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
        ev_maps, ev_charts = pipeline._processEvents(mapped_ds, namespace.get(self.inputEvents, None), mdh)
        pipeline._add_missing_ds_keys(mapped_ds, ev_maps)

        #Fit module specific filter settings
        if 'Analysis.FitModule' in mdh.getEntryNames():
            fitModule = mdh['Analysis.FitModule']

            if 'LatGaussFitFR' in fitModule:
                mapped_ds.addColumn('nPhotons', pipeline.getPhotonNums(mapped_ds, mdh))

        mapped_ds.mdh = mdh

        namespace[self.outputLocalizations] = mapped_ds


@register_module('Fold')
class Fold(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = CStr('localizations')
    outputName = CStr('folded')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.inputName]

        if 'mdh' not in dir(inp):
            raise RuntimeError('Unfold needs metadata')

        mapped = tabular.mappingFilter(inp)

        multiview.foldX(mapped, inp.mdh)
        mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped


@register_module('ShiftCorrect')
class ShiftCorrect(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = CStr('folded')
    inputShiftMap = CStr('')
    outputName = CStr('registered')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview
        from PYME.IO import unifiedIO
        import json

        inp = namespace[self.inputName]

        if 'mdh' not in dir(inp):
            raise RuntimeError('ShiftCorrect needs metadata')

        if self.inputShiftMap == '':  # grab shftmap from the metadata
            s = unifiedIO.read(inp.mdh['Shiftmap'])
        else:
            s = unifiedIO.read(self.inputShiftMap)

        shiftMaps = json.loads(s)

        mapped = tabular.mappingFilter(inp)

        multiview.applyShiftmaps(mapped, shiftMaps)  # FIXME: parse mdh for camera.ROIX

        mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped


@register_module('FindClumps')
class FindClumps(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = CStr('registered')
    gapTolerance = Int(1, desc='Number of off-frames allowed to still be a single clump')
    radiusScale = Float(2.0)
    radius_offset_nm = Float(150., desc='[nm]')
    outputName = CStr('clumped')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.inputName]


        #    raise RuntimeError('Unfold needs metadata')

        mapped = tabular.mappingFilter(inp)

        multiview.findClumps(mapped, self.gapTolerance, self.radiusScale, self.radius_offset_nm)

        if 'mdh' in dir(inp):
            mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped


@register_module('MergeClumps')
class MergeClumps(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = CStr('clumped')
    outputName = CStr('merged')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.inputName]

        mapped = tabular.mappingFilter(inp)

        if 'mdh' not in dir(inp):
            raise RuntimeError('MergeClumps needs metadata')

        grouped = multiview.mergeClumps(mapped, inp.mdh.getOrDefault('Multiview.NumROIs', 0))

        grouped.mdh = inp.mdh

        namespace[self.outputName] = grouped


@register_module('MapAstigZ')
class MapAstigZ(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = CStr('merged')
    AstigmatismMapID = CStr('')
    outputName = CStr('zmapped')

    def execute(self, namespace):
        from PYME.Analysis.points.astigmatism import astigTools
        from PYME.IO import unifiedIO
        import json

        inp = namespace[self.inputName]

        if 'mdh' not in dir(inp):
            raise RuntimeError('MapAstigZ needs metadata')

        if self.AstigmatismMapID == '':  # grab calibration from the metadata
            s = unifiedIO.read(inp.mdh['Analysis.AstigmatismMapID'])
        else:
            s = unifiedIO.read(self.AstigmatismMapID)

        astig_calibrations = json.loads(s)

        mapped = tabular.mappingFilter(inp)

        z, zerr = astigTools.lookup_astig_z(mapped, astig_calibrations, plot=False)

        mapped.addColumn('astigZ', z)
        mapped.addColumn('zLookupError', zerr)
        mapped.setMapping('z', 'astigZ + z')

        mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped

@register_module('idTransientFrames')
class idTransientFrames(ModuleBase):
    """
    Adds an 'isTransient' column to the input datasource so that one can filter localizations that are from frames
    acquired during z-translation
    """
    inputName = CStr('zmapped')
    inputEvents = CStr('Events')
    framesPerStep = Float()
    outputName = CStr('transientFiltered')

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

@register_module('dbscanClustering')
class dbscanClustering(ModuleBase):
    """
    Performs DBSCAN clustering on input dictionary
    Args:
        searchRadius: search radius for clustering
        minPtsForCore: number of points within SearchRadius required for a given point to be considered a core point
    """
    inputName = CStr('filtered')
    xKey = CStr('x')
    yKey = CStr('y')
    zKey = CStr('z')
    searchRadius = Float()
    minPtsForCore = Int()

    outputName = CStr('dbscanClustered')

    def execute(self, namespace):
        from sklearn.cluster import dbscan

        inp = namespace[self.inputName]
        mapped = inpFilt.mappingFilter(inp)

        # Note that sklearn gives unclustered points label of -1, and first value starts at 0.
        core_samp, dbLabels = dbscan(np.vstack([inp[self.xKey], inp[self.yKey], inp[self.zKey]]).T,
                                     self.searchRadius, self.minPtsForCore)

        # shift dbscan labels up by one, so that the 0th cluster doesn't get lost when pipeline fills in filtered points
        # but keep the unclustered points at -1
        dbLabels[dbLabels == -1] = -2
        mapped.addColumn('dbscanClumpID', dbLabels + 1)

        # propogate metadata, if present
        try:
            mapped.mdh = inp.mdh
        except AttributeError:
            pass

        namespace[self.outputName] = mapped


