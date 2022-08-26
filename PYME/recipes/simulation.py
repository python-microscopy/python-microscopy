from .base import register_module, ModuleBase,Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, DictStrFloat, DictStrBool, on_trait_change

import numpy as np


@register_module('RandomPoints')
class RandomPoints(ModuleBase):
    output = Output('points')

    def execute(self, namespace):
        from PYME.IO import tabular
        namespace[self.output]= tabular.RandomSource(1000, 1000, 1000)