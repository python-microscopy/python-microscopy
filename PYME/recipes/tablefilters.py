from .base import register_module, ModuleBase, Filter, Float, Enum, CStr, Bool, Int, View, Item#, Group
from traits.api import DictStrStr
import numpy as np
import pandas as pd
from PYME.LMVis import inpFilt

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

