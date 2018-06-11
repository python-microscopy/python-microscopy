from .base import BaseEngine, EngineLayer
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram
from PYME.LMVis.shader_programs.WireFrameShaderProgram import WireFrameShaderProgram
from PYME.LMVis.shader_programs.GouraudShaderProgram import GouraudShaderProgram

from PYME.recipes.traits import CStr, Float, Enum, ListFloat, List
from pylab import cm
import numpy as np
import dispatch

from OpenGL.GL import *


class WireframeEngine(BaseEngine):
    def __init__(self):
        BaseEngine.__init__(self)
        self.set_shader_program(WireFrameShaderProgram)

    def render(self, gl_canvas, layer):
        self._set_shader_clipping(gl_canvas)

        with self.shader_program:
            vertices = layer.get_vertices()
            n_vertices = vertices.shape[0]

            glVertexPointerf(vertices)
            glNormalPointerf(layer.get_normals())
            glColorPointerf(layer.get_colors())

            glDrawArrays(GL_TRIANGLES, 0, n_vertices)


class FlatFaceEngine(WireframeEngine):
    def __init__(self):
        BaseEngine.__init__(self)

    def render(self, gl_canvas, layer):
        self.set_shader_program(DefaultShaderProgram)
        WireframeEngine.render(self, gl_canvas, layer)


class ShadedFaceEngine(WireframeEngine):
    def __init__(self):
        BaseEngine.__init__(self)

    def render(self, gl_canvas, layer):
        self.set_shader_program(GouraudShaderProgram)
        WireframeEngine.render(self, gl_canvas, layer)

ENGINES = {
    'wireframe' : WireframeEngine,
    'monochrome_triangles' : FlatFaceEngine,
    'shaded_triangles' : ShadedFaceEngine,
}


class TriangleRenderLayer(EngineLayer):
    """
    Layer for viewing triangle meshes.
    """
    # properties to show in the GUI. Note that we also inherit 'visible' from BaseLayer
    vertexColour = CStr('', desc='Name of variable used to colour our points')
    cmap = Enum(*cm.cmapnames, default='gist_rainbow', desc='Name of colourmap used to colour faces')
    clim = ListFloat([0, 1], desc='How our variable should be scaled prior to colour mapping')
    alpha = Float(1.0, desc='Face tranparency')
    method = Enum(*ENGINES.keys(), desc='Method used to display faces')

    def __init__(self, pipeline, method='wireframe', datasource=None, **kwargs):
        self._pipeline = pipeline
        self.engine = None
        self.cmap = 'gist_rainbow'

        self.x_key = 'x'  # TODO - make these traits?
        self.y_key = 'y'
        self.z_key = 'z'

        self.xn_key = 'xn'
        self.yn_key = 'yn'
        self.zn_key = 'zn'

        self._bbox = None

        # define a signal so that people can be notified when we are updated (currently used to force a redraw when
        # parameters change)
        self.on_update = dispatch.Signal()

        # define responses to changes in various traits
        self.on_trait_change(self._update, 'vertexColour')
        self.on_trait_change(lambda: self.on_update.send(self), 'visible')
        self.on_trait_change(self.update, 'cmap, clim, alpha, datasource')
        self.on_trait_change(self._set_method, 'method')

        # update any of our traits which were passed as command line arguments
        self.set(**kwargs)

        # update datasource and method
        self.datasource = datasource
        if self.method == method:
            #make sure we still call _set_method even if we start with the default method
            self._set_method()
        else:
            self.method = method

        # if we were given a pipeline, connect ourselves to the onRebuild signal so that we can automatically update
        # ourselves
        if not self._pipeline is None:
            self._pipeline.onRebuild.connect(self.update)

    # @property
    # def datasource(self):
    #     """
    #     Return the datasource we are connected to (does not go through the pipeline for triangles_mesh).
    #     """
    #     #return self._pipeline.get_layer_data(self.dsname)
    #     return self.datasource

    def _set_method(self):
        self.engine = ENGINES[self.method]()
        self.update()

    def _get_cdata(self):
        try:
            cdata = self.datasource[self.vertexColour]
        except KeyError:
            cdata = np.array([0, 1])

        return cdata

    def _update(self, *args, **kwargs):
        cdata = self._get_cdata()
        self.clim = [float(cdata.min()), float(cdata.max())]
        # self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        if not (self.engine is None or self.datasource is None):
            print ('lw update')
            self.update_from_datasource(self.datasource)
            self.on_update.send(self)

    @property
    def bbox(self):
        return self._bbox

    def update_from_datasource(self, ds):
        """
        Pulls vertices/normals from a binary STL file. See PYME.IO.FileUtils.stl for more info. Calls update_data on the input.

        Parameters
        ----------
        ds :
            Binary STL file read in via load_stl_binary()

        Returns
        -------
        None
        """

        # Concatenate vertices, interleave, restore to 3x(3N) points (3xN triangles),
        # and assign the points to x, y, z vectors
        x, y, z = np.hstack((ds['vertex0'], ds['vertex1'], ds['vertex2'])).reshape(-1, 3).T

        # We copy the normals 3 times per triangle to get 3x(3N) normals to match the vertices shape
        xn, yn, zn = np.repeat(ds['normal'].T, 3, axis=1)
        #xn, yn, zn = np.repeat(ds['normal'], 3, axis=1).reshape(-1, 3).T

        # Pass the restructured data to update_data
        self.update_data(x, y, z, cmap=getattr(cm, self.cmap), clim=self.clim, alpha=self.alpha, xn=xn, yn=yn, zn=zn)

    def update_data(self, x=None, y=None, z=None, cmap=None, clim=None, alpha=1.0, xn=None, yn=None, zn=None):
        """
        Feeds new vertex, normal, color, and transparency information about mesh triangles to VisGUI.

        Parameters
        ----------
        x, y, z :
            Triangle vertex 3-coordinates
        cmap :
            Color map
        clim :
            Color map lower and upper limits
        alpha :
            Transparency of triangles (between 0 and 1)
        xn, yn, zn :
            Normal magnitudes in x, y, and z-direction

        Returns
        -------
        None
        """

        self._vertices = None
        self.normals = None
        self._colors = None
        self._color_map = None
        self._color_limit = 0
        self._alpha = 0

        # Do we have coordinates? Concatenate into vertices.
        if x is not None and y is not None and z is not None:
            vertices = np.vstack((x.ravel(), y.ravel(), z.ravel()))
            vertices = vertices.T.ravel().reshape(len(x.ravel()), 3)

            if not xn is None:
                normals = np.vstack((xn.ravel(), yn.ravel(), zn.ravel())).T.ravel().reshape(len(x.ravel()), 3)
            else:
                normals = -0.69 * np.ones(vertices.shape)

            self._bbox = np.array([x.min(), y.min(), z.min(), x.max(), y.max(), z.max()])
        else:
            vertices = None
            normals = None
            self._bbox = None

        # TODO: This temporarily sets all triangles to the color red. User should be able to select color.
        color = 255  # pink
        colors = np.ones(vertices.shape[0]) * color  # vector of pink

        if clim is not None and colors is not None:
            cs_ = ((colors - clim[0]) / (clim[1] - clim[0]))
            cs = cmap(cs_)
            cs[:, 3] = alpha

            cs = cs.ravel().reshape(len(colors), 4)
        else:
            # cs = None
            if not vertices is None:
                cs = np.ones((vertices.shape[0], 4), 'f')
            else:
                cs = None
            color_map = None
            color_limit = None

        print ('setting values')
        self.set_values(vertices, normals, cs, cmap, clim, alpha)

    def set_values(self, vertices=None, normals=None, colors=None, color_map=None, color_limit=None, alpha=None):
        if vertices is not None:
            self._vertices = vertices
        if normals is not None:
            self._normals = normals
        if color_map is not None:
            self._color_map = color_map
        if colors is not None:
            self._colors = colors
        if color_limit is not None:
            self._color_limit = color_limit
        if alpha is not None:
            self._alpha = alpha

    def get_vertices(self):
        return self._vertices

    def get_normals(self):
        return self._normals

    def get_colors(self):
        return self._colors

    def get_color_map(self):
        return self._color_map

    @property
    def colour_map(self):
        return self._color_map

    def get_color_limit(self):
        return self._color_limit

    @property
    def default_view(self):
        from traitsui.api import View, Item, Group, InstanceEditor, EnumEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor, CBEditor

        return View([Group([Item('datasource', label='Data', editor=EnumEditor(name='_datasource_choices')), ]),
                     Item('method'),
                     Item('vertexColour', editor=EnumEditor(name='_datasource_keys'), label='Colour'),
                     Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata), show_label=False), ]),
                     Group([Item('cmap', label='LUT'), Item('alpha'), Item('visible')])], )
        # buttons=['OK', 'Cancel'])

    def default_traits_view(self):
        return self.default_view