from .base import EngineLayer
from .mesh import TriangleRenderLayer  #, ENGINES

from PYME.experimental._octree import Octree

from PYME.recipes.traits import CStr, Float, Enum, ListFloat, List, Int
# from pylab import cm
from PYME.misc.colormaps import cm
import numpy as np


from OpenGL.GL import *


# OCT_SHIFT is just _octant_sign from _octree
OCT_SHIFT = np.zeros((8,3))
for n in range(8):
    OCT_SHIFT[n,0] = 2*(n&1) - 1
    OCT_SHIFT[n,1] = (n&2) -1
    OCT_SHIFT[n,2] = (n&4)/2.0 -1

class OctreeRenderLayer(TriangleRenderLayer):
    """
    Layer for viewing octrees. Takes in an octree, splits the faces into triangles, and then uses the rendering engines
    from PYME.LMVis.layers.triangle_mesh.
    """
    # Additional properties (the rest are inherited from TriangleRenderLayer)
    depth = Int(3, desc='Depth at which to render Octree. Set to -1 for dynamic depth rendering.')
    density = Float(0.0, desc='Minimum density of octree node to display.')
    min_points = Int(10, desc='Number of points/node to truncate octree at')

    def __init__(self, pipeline, method='wireframe', dsname='', context=None, **kwargs):
        TriangleRenderLayer.__init__(self, pipeline, method, dsname, context, **kwargs)

        self.on_trait_change(self.update, 'depth')
        self.on_trait_change(self.update, 'density')
        self.on_trait_change(self.update, 'min_points')
        
    @property
    def _ds_class(self):
        from PYME.experimental import octree
        return (octree.Octree, octree.PyOctree)

    def update_from_datasource(self, ds):
        """
        Opens an octree. Subdivides the faces into triangles. Feeds the triangle points/normals to update_data.

        Parameters
        ----------
        ds :
            Octree (see PYME.experimental.octree)

        Returns
        -------
        None
        """
        nodes = ds._nodes[ds._nodes[ds._nodes['parent']]['nPoints'] >= float(self.min_points)]
        
        if self.depth > 0:
            # Grab the nodes at the specified depth
            nodes = nodes[nodes['depth'] == self.depth]
            box_sizes = np.ones((nodes.shape[0], 3))*ds.box_size(self.depth)
            node_density = 1.*nodes['nPoints']/np.prod(box_sizes,axis=1)
            nodes = nodes[node_density >= self.density]
            box_sizes = np.ones((nodes.shape[0], 3))*ds.box_size(self.depth)
            alpha = nodes['nPoints']/box_sizes[:,0]
        elif self.depth == 0:
            # plot all bins
            nodes = nodes[nodes['nPoints'] >= 1]
            box_sizes = np.vstack(ds.box_size(nodes['depth'])).T

            alpha = nodes['nPoints'] * ((2 ** nodes['depth'])**3)
        else:
            # Plot leaf nodes
            nodes = nodes[(np.sum(nodes['children'],axis=1) == 0)&(nodes['depth'] > 0)]
            box_sizes = np.vstack(ds.box_size(nodes['depth'])).T
            
            alpha = nodes['nPoints']*((2.0**nodes['depth']))**3

        if len(nodes) > 0:
            c = nodes['centre']  # center
            shifts = (box_sizes[:,None]*OCT_SHIFT[None,:])*0.5
            v = (c[:,None,:] + shifts)

            #
            #     z
            #     ^
            #     |
            #    v4 ----------v6
            #    /|           /|
            #   / |          / |
            #  v5----------v7  |
            #  |  |    c    |  |
            #  | v0---------|-v2
            #  | /          | /
            #  v1-----------v3---> y
            #  /
            # x
            #
            # Now note that the counterclockwise triangles (when viewed straight-on) formed along the faces of the cube are:
            #
            # v0 v2 v1
            # v0 v1 v5
            # v0 v5 v4
            # v0 v6 v2
            # v0 v4 v6
            # v1 v2 v3
            # v1 v3 v7
            # v1 v7 v5
            # v2 v6 v7
            # v2 v7 v3
            # v4 v5 v6
            # v5 v7 v6

            # Counterclockwise triangles (when viewed straight-on) formed along
            # the faces of an octree box
            t0 = np.vstack(v[:,[0,0,0,0,0,1,1,1,2,2,4,5],:])
            t1 = np.vstack(v[:,[2,1,5,6,4,2,3,7,6,7,5,7],:])
            t2 = np.vstack(v[:,[1,5,4,2,6,3,7,5,7,3,6,6],:])

            x, y, z = np.hstack([t0,t1,t2]).reshape(-1, 3).T  # positions

            # Now we create the normal as the cross product
            tn = np.cross((t2-t1),(t0-t1))

            # We copy the normals 3 times per triangle to get 3x(3N) normals to match the vertices shape
            xn, yn, zn = np.repeat(tn.T, 3, axis=1)  # normals

            # Color is fixed constnat for octree
            c = np.ones(len(x))
            clim = [0, 1]
            
            alpha = self.alpha*alpha/alpha.max()
            alpha = (alpha[None,:]*np.ones(12)[:,None])
            alpha = np.repeat(alpha.ravel(), 3)
            print('Octree scaled alpha range: %g, %g' % (alpha.min(), alpha.max()))

            cmap = cm[self.cmap]

            # Do we have coordinates? Concatenate into vertices.
            if x is not None and y is not None and z is not None:
                vertices = np.vstack((x.ravel(), y.ravel(), z.ravel()))
                self._vertices = vertices.T.ravel().reshape(len(x.ravel()), 3)

                if not xn is None:
                    self._normals = np.vstack((xn.ravel(), yn.ravel(), zn.ravel())).T.ravel().reshape(len(x.ravel()), 3)
                else:
                    self._normals = -0.69 * np.ones(self._vertices.shape)

                self._bbox = np.array([x.min(), y.min(), z.min(), x.max(), y.max(), z.max()])
            else:
                self._bbox = None

            if clim is not None and c is not None and cmap is not None:
                cs_ = ((c - clim[0]) / (clim[1] - clim[0]))
                cs = cmap(cs_)

                if self.method in ['flat', 'tessel']:
                    alpha = cs_ * alpha
                
                cs[:, 3] = alpha
                
                if self.method == 'tessel':
                    cs = np.power(cs, 0.333)

                self._colors = cs.ravel().reshape(len(c), 4)
            else:
                # cs = None
                if not self._vertices is None:
                    self._colors = np.ones((self._vertices.shape[0], 4), 'f')

            print('Colors: {}'.format(self._colors))
                
            self._alpha = alpha
            self._color_map = cmap
            self._color_limit = clim
        else:
            print('No nodes for density {0}, depth {1}'.format(self.density, self.depth))
        

    @property
    def default_view(self):
        from traitsui.api import View, Item, Group, InstanceEditor, EnumEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor, CBEditor

        return View([#Group([
                     Item('dsname', label='Data', editor=EnumEditor(name='_datasource_choices')),
                     Item('method'),
                     Item('depth'),Item('min_points'),
                     #Item('vertexColour', editor=EnumEditor(name='_datasource_keys'), label='Colour'),
                     #Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata), show_label=False), ]),
                     Group([Item('cmap', label='LUT'), Item('alpha')])], )
        # buttons=['OK', 'Cancel'])

