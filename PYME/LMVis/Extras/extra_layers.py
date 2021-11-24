import numpy as np
import wx
import os



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


def add_octree_layer(visFr):
    # create the octree once and add to the pipeline as a new data source, rather than adding the
    # octree module to the pipeline - this avoids
    # regenerating everytime we change something with the points
    # TODO - expose octree parameters
    from PYME.LMVis.layers.octree import OctreeRenderLayer
    
    if not 'octree0' in visFr.pipeline.dataSources.keys():
        gen_octree_from_points(visFr)
        
    l = OctreeRenderLayer(visFr.pipeline, 'flat', 'octree0', depth=0, alpha=1.0)
    visFr.add_layer(l)
        
    print('Octree layer added')
        
def gen_octree_from_points(visFr):
    from PYME.recipes.pointcloud import Octree
    
    pipeline = visFr.pipeline
    oc_name = pipeline.new_ds_name('octree')
    
    colour_chans = pipeline.colourFilter.getColourChans()
    current_colour = pipeline.colourFilter.currentColour
    if len(colour_chans) > 1:
        channel_choices = colour_chans + ['<all>',]

        dlg = wx.SingleChoiceDialog(visFr, "Which colour channel do you want to skin?", "Multiple colour channels detected", channel_choices)

        if not dlg.ShowModal():
            dlg.Destroy()
            return
        else:
            chan = dlg.GetStringSelection()
            dlg.Destroy()
            
            if chan == '<all>':
                chan = None
                
            pipeline.colourFilter.setColour(chan)
    
    otm = Octree()
    pipeline.dataSources[oc_name] = otm.apply_simple(pipeline)
    print('Octree  (%s) created' % oc_name)

    pipeline.colourFilter.setColour(current_colour)
    
    return oc_name

def gen_isosurface(visFr):
    from PYME.LMVis.layers.mesh import TriangleRenderLayer
    from PYME.recipes.surface_fitting import DualMarchingCubes
    
    oc_name = gen_octree_from_points(visFr)
    surf_name, surf_count = visFr.pipeline.new_ds_name('surf', return_count=True)
    
    recipe = visFr.pipeline.recipe
    dmc = DualMarchingCubes(recipe, invalidate_parent=False, input=oc_name,output=surf_name)
    
    if dmc.configure_traits(kind='modal'):
        recipe.add_modules_and_execute([dmc,])

        print('Isosurface generated, adding layer')
        layer = TriangleRenderLayer(visFr.pipeline, dsname=surf_name, method='shaded', cmap = ['C', 'M', 'Y', 'R', 'G', 'B'][surf_count % 6])
        visFr.add_layer(layer)
        dmc._invalidate_parent = True
        print('Isosurface layer added')
        
def open_surface(visFr):
    import wx
    # from PYME.experimental import triangle_mesh
    from PYME.experimental import _triangle_mesh as triangle_mesh
    from PYME.LMVis.layers.mesh import TriangleRenderLayer
    
    filename = wx.FileSelector("Choose a file to open",
                                   default_extension='stl',
                                   wildcard='STL mesh (*.stl)|*.stl|PLY mesh (*.ply)|*.ply')
    #print filename
    if not filename == '':
        tail = os.path.split(filename)[-1]
        base_name = tail.split('.')[0]
        surf_name, surf_count = base_name, 0
        while surf_name in visFr.pipeline.dataSources.keys():
            surf_count += 1
            surf_name = '%s%d' % (base_name, surf_count)

        ext = tail.split('.')[-1]
        if ext == 'stl':
            visFr.pipeline.dataSources[surf_name] = triangle_mesh.TriangleMesh.from_stl(filename)
        elif ext == 'ply':
            visFr.pipeline.dataSources[surf_name] = triangle_mesh.TriangleMesh.from_ply(filename)
        else:
            raise ValueError('Invalid file extension .' + str(ext))
        layer = TriangleRenderLayer(visFr.pipeline, dsname=surf_name, method='shaded')
        visFr.add_layer(layer)
        
def save_surface(visFr):
    import wx
    from PYME.experimental import _triangle_mesh as triangle_mesh
    
    surf_keys = [key for key, mesh in visFr.pipeline.dataSources.items() if isinstance(mesh, triangle_mesh.TriangleMesh)]
    
    if len(surf_keys) == 0:
        raise RuntimeError('No surfaces present')
    
    if len(surf_keys) == 1:
        key = surf_keys[0]
    else:
        dlg = wx.SingleChoiceDialog(visFr, "Which surface do you want to save?", "Choose a surface to save", surf_keys)
        
        if not dlg.ShowModal():
            dlg.Destroy()
            return
        else:
            key = dlg.GetStringSelection()
            dlg.Destroy()

    filename = wx.FileSelector('Save surface as...',
                               default_extension='stl',
                               wildcard='STL mesh (*.stl)|*.stl|PLY mesh (*.ply)|*.ply',
                               flags=wx.FD_SAVE)

    if not filename == '':
        ext = filename.split('.')[-1]
        if ext == 'stl':
            visFr.pipeline.dataSources[key].to_stl(filename)
        elif ext == 'ply':
            colors = None
            # Check if we've rendered this data source 
            layer_list = [x.dsname for x in visFr.glCanvas.layers]
            if key in layer_list:
                # If we have, save the PLY with its colors
                layer = visFr.glCanvas.layers[layer_list.index(key)]
                # Construct a re-indexing for non-negative vertices
                live_vertices = np.flatnonzero(visFr.pipeline.dataSources[key]._vertices['halfedge'] != -1)
                new_vertex_indices = np.arange(live_vertices.shape[0])
                vertex_lookup = np.zeros(visFr.pipeline.dataSources[key]._vertices.shape[0])
                
                vertex_lookup[live_vertices] = new_vertex_indices

                # Grab the faces and vertices we want
                faces = vertex_lookup[visFr.pipeline.dataSources[key].faces]

                colors = np.zeros((live_vertices.size, 3), dtype=np.ubyte)
                colors[faces.ravel().astype(np.int)] = np.floor(layer._colors[:,:3]*255).astype(np.ubyte)
                
            visFr.pipeline.dataSources[key].to_ply(filename, colors)
        else:
            raise ValueError('Invalid file extension .' + str(ext))
    
def distance_to_surface(visFr):
    from PYME.recipes.surface_fitting import DistanceToMesh
    from PYME.experimental._triangle_mesh import TriangleMesh

    pipeline = visFr.pipeline

    dist_name = visFr.pipeline.new_ds_name('distance')

    mesh_names = [k for k, v in pipeline.dataSources.items() if isinstance(v, TriangleMesh)]
    
    dlg = wx.SingleChoiceDialog(visFr, "Measure distance to which mesh?", "Choose a mesh", mesh_names)

    if dlg.ShowModal() == wx.ID_OK:
        surf_name = dlg.GetStringSelection()
        dlg.Destroy()
    else:
        dlg.Destroy()
        return

    recipe = visFr.pipeline.recipe
    dts = DistanceToMesh(recipe, input_mesh=surf_name, input_points=pipeline.selectedDataSourceKey, output=dist_name)

    recipe.add_modules_and_execute([dts,])
    visFr.pipeline.selectDataSource(dist_name)
 
def estimate_density(visFr):
    from PYME.recipes.pointcloud import LocalPointDensity
    dens_name = visFr.pipeline.new_ds_name('dense')

    recipe = visFr.pipeline.recipe
    dmc = LocalPointDensity(recipe, invalidate_parent=False, input=visFr.pipeline.selectedDataSourceKey, output=dens_name)

    if dmc.configure_traits(kind='modal'):
        recipe.add_modules_and_execute([dmc,])
        dmc._invalidate_parent = True
        visFr.pipeline.selectDataSource(dens_name)
        
        
def estimate_circumcentre_densities(visFr):
    from PYME.recipes.pointcloud import LocalPointDensity, DelaunayCircumcentres

    recipe = visFr.pipeline.recipe
    
    tess_name = add_tesselation(visFr)
    if tess_name is None:
        return
        
    cc_name = visFr.pipeline.new_ds_name('circumcentres')
    cc = DelaunayCircumcentres(recipe, input=tess_name, output=cc_name)
    #recipe.add_module(cc)
    
    dens_name = visFr.pipeline.new_ds_name('dense')
    dmc = LocalPointDensity(recipe, invalidate_parent=False, input=visFr.pipeline.selectedDataSourceKey, input_sample_locations=cc_name, output=dens_name)

    if dmc.configure_traits(kind='modal'):
        #recipe.add_module(dmc)
        #recipe.execute()
        recipe.add_modules_and_execute([cc, dmc])
        dmc._invalidate_parent = True
        visFr.pipeline.selectDataSource(dens_name)
    
    
def add_tesselation(visFr):
    from PYME.recipes.pointcloud import DelaunayTesselation

    pipeline = visFr.pipeline
    surf_name = pipeline.new_ds_name('delaunay')

    colour_chans = pipeline.colourFilter.getColourChans()
    current_colour = pipeline.colourFilter.currentColour
    if len(colour_chans) > 1:
        channel_choices = colour_chans + ['<all>', ]
    
        dlg = wx.SingleChoiceDialog(visFr, "Which colour channel do you want to skin?",
                                    "Multiple colour channels detected", channel_choices)
    
        if not dlg.ShowModal():
            dlg.Destroy()
            return
        else:
            chan = dlg.GetStringSelection()
            dlg.Destroy()
        
            if chan == '<all>':
                chan = None
        
            pipeline.colourFilter.setColour(chan)

    dt = DelaunayTesselation()
    if dt.configure_traits(kind='modal'):
        pipeline.dataSources[surf_name] = dt.apply_simple(pipeline)
        print('Delaunay tesselation  (%s) created' % surf_name)
        pipeline.colourFilter.setColour(current_colour)
        return surf_name
        
    else:
        pipeline.colourFilter.setColour(current_colour)
        return None
    
def add_tesselation_layer(visFr):
    from PYME.LMVis.layers.mesh import TriangleRenderLayer
    surf_name = add_tesselation(visFr)
    if surf_name is not None:
        layer = TriangleRenderLayer(visFr.pipeline, dsname=surf_name, method='flat',
                                    cmap='hot')
        visFr.add_layer(layer)
        print('Tesselation layer added')

def gen_isosurface_from_tesselation(visFr):
    from PYME.LMVis.layers.mesh import TriangleRenderLayer
    from PYME.recipes.surface_fitting import DelaunayMarchingTetrahedra

    if 'delaunay0' not in visFr.pipeline.dataSources.keys():
        del_name = add_tesselation(visFr)
    else:
        _, c = visFr.pipeline.new_ds_name('delaunay', return_count=True)
        del_name = 'delaunay{}'.format(c-1)

    if 'dn' not in visFr.pipeline.dataSources[del_name].keys():
        # visFr.pipeline.selectDataSource(del_name)
        estimate_density(visFr)
    
    surf_name, surf_count = visFr.pipeline.new_ds_name('surf', return_count=True)
    
    recipe = visFr.pipeline.recipe
    mt = DelaunayMarchingTetrahedra(recipe, invalidate_parent=False, input=del_name, output=surf_name)
    
    if mt.configure_traits(kind='modal'):
        recipe.add_modules_and_execute([mt,])

        print('Isosurface generated, adding layer')
        layer = TriangleRenderLayer(visFr.pipeline, dsname=surf_name, method='shaded', cmap = ['C', 'M', 'Y', 'R', 'G', 'B'][surf_count % 6])
        visFr.add_layer(layer)
        mt._invalidate_parent = True
        print('Isosurface layer added')


def Plug(visFr):
    visFr.AddMenuItem('View', 'Add Octree Layer', lambda e : add_octree_layer(visFr))
    visFr.AddMenuItem('View', 'Estimate density', lambda e: estimate_density(visFr))
    visFr.AddMenuItem('View', 'Estimate density [circumcentres]', lambda e: estimate_circumcentre_densities(visFr))
    visFr.AddMenuItem('View', 'Create Delaunay Tesselation', lambda e: add_tesselation_layer(visFr))
    visFr.AddMenuItem('View', 'Generate Isosurface from Delaunay Tesselation', lambda e: gen_isosurface_from_tesselation(visFr))
    visFr.AddMenuItem('Mesh', 'Generate Isosurface', lambda e: gen_isosurface(visFr))
    visFr.AddMenuItem('Mesh', 'Load mesh', lambda e: open_surface(visFr))
    visFr.AddMenuItem('Mesh', 'Save mesh', lambda e: save_surface(visFr))
    visFr.AddMenuItem('Mesh>Analysis', 'Distance to mesh', lambda e: distance_to_surface(visFr))
