from .base import EngineLayer
from .triangle_mesh import TriangleRenderLayer, ENGINES

from PYME.experimental._octree import Octree

from PYME.recipes.traits import CStr, Float, Enum, ListFloat, List, Int
from pylab import cm
import numpy as np


from OpenGL.GL import *


class OctreeRenderLayer(TriangleRenderLayer):
    """
    Layer for viewing octrees. Takes in an octree, splits the faces into triangles, and then uses the rendering engines
    from PYME.LMVis.layers.triangle_mesh.
    """
    # properties to show in the GUI. Note that we also inherit 'visible' from BaseLayer
    #vertexColour = CStr('', desc='Name of variable used to colour our points')
    #cmap = Enum(*cm.cmapnames, default='gist_rainbow', desc='Name of colourmap used to colour faces')
    #clim = ListFloat([0, 1], desc='How our variable should be scaled prior to colour mapping')
    #alpha = Float(1.0, desc='Face tranparency')
    depth = Int(3, desc='Depth at which to render Octree. Set to -1 for dynamic depth rendering.')
    #method = Enum(*ENGINES.keys(), desc='Method used to display faces')

    def __init__(self, pipeline, method='wireframe', datasource=None, **kwargs):
        TriangleRenderLayer.__init__(self, pipeline, method, datasource, **kwargs)

        self.on_trait_change(self.update, 'depth')

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
        
        if self.depth > 0:
            # Grab the nodes at the specified depth
            nodes = ds._nodes[ds._nodes['depth'] == self.depth]
            box_sizes = np.ones((nodes.shape[0], 3))*ds.box_size(self.depth)
            alpha = self.alpha*nodes['nPoints']
        elif self.depth == 0:
            # plot all bins
            nodes = ds._nodes[ds._nodes['nPoints'] >= 1]
            box_sizes = np.vstack(ds.box_size(nodes['depth'])).T

            alpha = self.alpha * nodes['nPoints'] * (8.0 ** nodes['depth'])
            
        else:
            # Follow the nodes until we reach a terminating node, then append this node to our list of nodes to render
            # Start at the 0th node
            # children = ds._nodes[0]['children'].tolist()
            # node_indices = []
            # box_sizes = []
            # # Do this until we've looked at the whole octree (the list of children is empty)
            # while children:
            #     # Check the child
            #     node_index = children.pop()
            #     curr_node = ds._nodes[node_index]
            #     # Is this a terminating node?
            #     if np.any(curr_node['children']):
            #         # It's not, so we'll add the children to the list
            #         new_children = curr_node['children'][curr_node['children'] > 0]
            #         children.extend(new_children)
            #     else:
            #         # Terminating node! We want to render this
            #         node_indices.append(node_index)
            #         box_sizes.append(ds.box_size(curr_node['depth']))
                    
            
            

            # We've followed the octree to the end, return the nodes and box sizes
            #nodes = ds._nodes[node_indices]
            #box_sizes = np.array(box_sizes)

            nodes = ds._nodes[ds._nodes['nPoints'] == 1]
            box_sizes = np.vstack(ds.box_size(nodes['depth'])).T
            
            alpha = self.alpha*nodes['nPoints']*(8.0**nodes['depth'])


        # First we need the vertices of the cube. We find them from the center c provided and the box size (lx, ly, lz)
        # provided by the octree:

        c = nodes['centre']  # center
        v0 = c + box_sizes * -1 / 2            # v0 = c - lx/2 - ly/2 - lz/2
        v1 = c + box_sizes * [-1, -1, 1] / 2   # v1 = c - lx/2 - ly/2 + lz/2
        v2 = c + box_sizes * [-1, 1, -1] / 2   # v2 = c - lx/2 + ly/2 - lz/2
        v3 = c + box_sizes * [-1, 1, 1] / 2    # v3 = c - lx/2 + ly/2 + lz/2
        v4 = c + box_sizes * [1, -1, -1] / 2   # v4 = c + lx/2 - ly/2 - lz/2
        v5 = c + box_sizes * [1, -1, 1] / 2    # v5 = c + lx/2 - ly/2 + lz/2
        v6 = c + box_sizes * [1, 1, -1] / 2    # v6 = c + lx/2 + ly/2 - lz/2
        v7 = c + box_sizes / 2                 # v7 = c + lx/2 + ly/2 + lz/2

        #
        #     z
        #     ^
        #     |
        #    v1 ----------v3
        #    /|           /|
        #   / |          / |
        #  v5----------v7  |
        #  |  |    c    |  |
        #  | v0---------|-v2
        #  | /          | /
        #  v4-----------v6---> y
        #  /
        # x
        #
        # Now note that the counterclockwise triangles (when viewed straight-on) formed along the faces of the cube are:
        #
        # v0 v2 v6
        # v0 v4 v5
        # v1 v3 v2
        # v2 v0 v1
        # v3 v1 v5
        # v3 v7 v6
        # v4 v6 v7
        # v5 v1 v0
        # v5 v7 v3
        # v6 v2 v3
        # v6 v4 v0
        # v7 v5 v4


        # Concatenate vertices, interleave, restore to 3x(3N) points (3xN triangles),
        # and assign the points to x, y, z vectors
        triangle_v0 = np.vstack((v0, v0, v1, v2, v3, v3, v4, v5, v5, v6, v6, v7))
        triangle_v1 = np.vstack((v2, v4, v3, v0, v1, v7, v6, v1, v7, v2, v4, v5))
        triangle_v2 = np.vstack((v6, v5, v2, v1, v5, v6, v7, v0, v3, v3, v0, v4))

        x, y, z = np.hstack((triangle_v0, triangle_v1, triangle_v2)).reshape(-1, 3).T

        # Now we create the normal as the cross product (triangle_v2 - triangle_v1) x (triangle_v0 - triangle_v1)
        triangle_normals = np.cross((triangle_v2 - triangle_v1), (triangle_v0 - triangle_v1))
        # We copy the normals 3 times per triangle to get 3x(3N) normals to match the vertices shape
        xn, yn, zn = np.repeat(triangle_normals.T, 3, axis=1)
        
        alpha = (alpha[None,:]*np.ones(12*3)[:,None]).ravel()
        print('Octree scaled alpha range: %g, %g' % (alpha.min(), alpha.max()))

        # Pass the restructured data to update_data
        self.update_data(x, y, z, cmap=getattr(cm, self.cmap), clim=self.clim, alpha=alpha, xn=xn, yn=yn, zn=zn)

    

    @property
    def default_view(self):
        from traitsui.api import View, Item, Group, InstanceEditor, EnumEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor, CBEditor

        return View([#Group([Item('datasource', label='Data', editor=EnumEditor(name='_datasource_choices')), ]),
                     Item('method'),
                     Item('depth'),
                     #Item('vertexColour', editor=EnumEditor(name='_datasource_keys'), label='Colour'),
                     #Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata), show_label=False), ]),
                     Group([Item('cmap', label='LUT'), Item('alpha'), Item('visible')])], )
        # buttons=['OK', 'Cancel'])

