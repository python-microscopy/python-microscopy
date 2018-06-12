from PYME.LMVis.layers.octree import OctreeRenderLayer
from PYME.experimental.octree import Octree
import numpy as np


def gen_octree_layer_from_points(visFr):
    MIN_PIXEL_SIZE=5
    
    pts = np.vstack([visFr.pipeline['x'], visFr.pipeline['y'], visFr.pipeline['z']]).T.astype('f4')
    
    r_min = pts.min(axis=0)
    r_max = pts.max(axis=0)
    
    bbox_size = (r_max - r_min).max()
    
    bb_max = r_min + bbox_size
    
    max_depth = np.log2(bbox_size/MIN_PIXEL_SIZE) + 1
    
    ot = Octree([r_min[0], bb_max[0], r_min[1], bb_max[1], r_min[2], bb_max[2]], maxdepth=max_depth)
    ot.add_points(pts)
    
    l = OctreeRenderLayer(visFr.pipeline, 'flat', ot, depth=0, alpha=1.0)
    
    visFr.add_layer(l)


def Plug(visFr):
    visFr.AddMenuItem('View', 'Add Octree Layer', lambda e : gen_octree_layer_from_points(visFr))