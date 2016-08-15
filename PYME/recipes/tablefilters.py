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
        inp = namespace[self.inputLocalizations]
        if not isinstance(inp, inpFilt.colourFilter):
            inp = inpFilt.colourFilter(inp)

        renderer = renderers.RENDERERS[str(self.renderingModule)](None, inp)

        namespace[self.outputImage] = renderer.Generate(self)

