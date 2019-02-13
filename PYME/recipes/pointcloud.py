from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr

import numpy as np
from PYME.IO import tabular



@register_module('Octree')
class Octree(ModuleBase):
    input_localizations = Input('input')
    output_octree = Output('output')
    
    minimum_pixel_size = Float(5)
    
    #bounds_mode = Enum(['Auto', 'Manual'])
    #manual_bounds = ListFloat([0,0,0, 5e3,5e3, 5e3])
    
    def execute(self, namespace):
        from PYME.experimental.octree import Octree
        inp = namespace[self.input_localizations]

        pts = np.vstack([inp['x'], inp['y'], inp['z']]).T.astype('f4')

        r_min = pts.min(axis=0)
        r_max = pts.max(axis=0)

        bbox_size = (r_max - r_min).max()

        bb_max = r_min + bbox_size

        max_depth = np.log2(bbox_size / self.minimum_pixel_size) + 1

        ot = Octree([r_min[0], bb_max[0], r_min[1], bb_max[1], r_min[2], bb_max[2]], maxdepth=max_depth)
        ot.add_points(pts)
        
        namespace[self.output_octree] = ot