from .base import register_module, ModuleBase, Filter, Float, Enum, CStr, Bool, Int, View, Item, List#, Group
from traits.api import DictStrStr, DictStrList
import numpy as np
import pandas as pd
from PYME.LMVis import inpFilt
from PYME.LMVis import renderers

@register_module('Mapping')
class Mapping(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = CStr('measurements')
    mappings = DictStrStr()
    outputName = CStr('mapped')

    def execute(self, namespace):
        inp = namespace[self.inputName]

        map = inpFilt.mappingFilter(inp, **self.mappings)

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

        map = inpFilt.resultsFilter(inp, **self.filters)

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
    zBounds = List([0,0])
    zSliceThickness = Float(50.0)
    softRender = Bool(True)

    def execute(self, namespace):
        from PYME.IO.image import ImageBounds
        inp = namespace[self.inputLocalizations]
        if not isinstance(inp, inpFilt.colourFilter):
            cf = inpFilt.colourFilter(inp, None)
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

        mapped_ds = inpFilt.mappingFilter(fitResults)


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


