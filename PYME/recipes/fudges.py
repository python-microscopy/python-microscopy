"""
This module contains fudges and workarounds - e.g. to deal with poorly formatted input data. They should be regarded as
temporary hacks until the problems in the upstream data that necessitated them can be fixed.

TLDR - If you need to use modules from this file you're doing it wrong.

"""

from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr

import numpy as np
from PYME.IO import tabular

@register_module('EstimateFrameZFromStepMetadata')
class EstimateFrameZFromStepMetadata(ModuleBase):
    """
    Estimate the z position of a frame from the stepping information in the metadata. Only really useful for HT data
    streamed from Labview with broken events & even then treat with caution.
    
    """
    inputImage = Input('input')
    outputName = Output('z_positions')
    
    def execute(self, namespace):
        from PYME.experimental import labview_spooling_hacks
        from PYME.Analysis import piecewiseMapping
        
        im = namespace[self.inputImage]
        
        position, frames = labview_spooling_hacks.spoof_focus_from_metadata(im.mdh)
        
        frame_nums = np.arange(im.data.shape[2])

        zm = piecewiseMapping.piecewiseMap(0, frames, position, im.mdh['Camera.CycleTime'], xIsSecs=False)
        
        z = zm[frame_nums]
        
        namespace[self.outputName] = tabular.ColumnSource(z=z)
        
        