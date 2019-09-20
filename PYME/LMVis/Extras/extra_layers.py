import numpy as np



# def _gen_octree_layer_from_points(visFr):
#     MIN_PIXEL_SIZE=5
#
#     pts = np.vstack([visFr.pipeline['x'], visFr.pipeline['y'], visFr.pipeline['z']]).T.astype('f4')
#
#     r_min = pts.min(axis=0)
#     r_max = pts.max(axis=0)
#
#     bbox_size = (r_max - r_min).max()
#
#     bb_max = r_min + bbox_size
#
#     max_depth = np.log2(bbox_size/MIN_PIXEL_SIZE) + 1
#
#     ot = Octree([r_min[0], bb_max[0], r_min[1], bb_max[1], r_min[2], bb_max[2]], maxdepth=max_depth)
#     ot.add_points(pts)
#
#     l = OctreeRenderLayer(visFr.pipeline, 'flat', ot, depth=0, alpha=1.0)
#
#     visFr.add_layer(l)


def gen_octree_layer_from_points(visFr):
    # create the octree once and add to the pipeline as a new data source, rather than adding the
    # octree module to the pipeline - this avoids
    # regenerating everytime we change something with the points
    # TODO - expose octree parameters
    
    from PYME.recipes.pointcloud import Octree
    from PYME.LMVis.layers.octree import OctreeRenderLayer
    
    otm = Octree()
    if True:#otm.configure_traits(kind='modal'):
        visFr.pipeline.dataSources['octree'] = otm.apply_simple(visFr.pipeline)
        
        print('Octree created, adding layer')
        
        l = OctreeRenderLayer(visFr.pipeline, 'flat', 'octree', depth=0, alpha=1.0)
        visFr.add_layer(l)
        
        print('Octree layer added')

def gen_isosurface(visFr):
    from PYME.LMVis.layers.triangle_mesh import TriangleRenderLayer
    from PYME.recipes.surface_fitting import DualMarchingCubes
    
    if not 'octree' in visFr.pipeline.dataSources.keys():
        gen_octree_layer_from_points(visFr)

    
    recipe = visFr.pipeline.recipe
    dmc = DualMarchingCubes(recipe, invalidate_parent=False, input='octree',output='surf')
    
    if dmc.configure_traits(kind='modal'):
        recipe.add_module(dmc)
        recipe.execute()

        print('Isosurface generated, adding layer')
        layer = TriangleRenderLayer(visFr.pipeline, dsname='surf')
        visFr.add_layer(layer)
        dmc._invalidate_parent = True
        print('Isosurface layer added')
    


def Plug(visFr):
    visFr.AddMenuItem('View', 'Add Octree Layer', lambda e : gen_octree_layer_from_points(visFr))
    visFr.AddMenuItem('View', 'Generate Isosurface', lambda e: gen_isosurface(visFr))