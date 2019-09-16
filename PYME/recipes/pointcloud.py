from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr

import numpy as np
from PYME.IO import tabular



@register_module('Octree')
class Octree(ModuleBase):
    input_localizations = Input('input')
    output_octree = Output('output')
    
    minimum_pixel_size = Float(5)
    max_depth = Int(20)
    samples_per_node = Int(1)
    
    #bounds_mode = Enum(['Auto', 'Manual'])
    #manual_bounds = ListFloat([0,0,0, 5e3,5e3, 5e3])
    
    def execute(self, namespace):
        from PYME.experimental.octree import gen_octree_from_points
        inp = namespace[self.input_localizations]

        ot = gen_octree_from_points(inp, min_pixel_size=self.minimum_pixel_size, max_depth=self.max_depth, samples_per_node=self.samples_per_node)
        
        namespace[self.output_octree] = ot