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
        self.set_shader_program(DefaultShaderProgram)

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
        self.set_shader_program(WireFrameShaderProgram)
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


class TrianglesRenderLayer(EngineLayer):
    """
    Layer for viewing triangle meshes.
    """
    # properties to show in the GUI. Note that we also inherit 'visible' from BaseLayer
    faceColour = CStr('', desc='Name of variable used to colour triangle faces')
    cmap = Enum(*cm.cmapnames, default='gist_rainbow', desc='Name of colourmap used to colour faces')
    clim = ListFloat([0, 1], desc='How our variable should be scaled prior to colour mapping')
    alpha = Float(1.0, desc='Face tranparency')
    method = Enum(*ENGINES.keys(), desc='Method used to display faces')
    dsname = CStr('output', desc='Name of the datasource within the pipeline to use as a source of points')
    _datasource_keys = List()
    _datasource_choices = List()

    def __init__(self, pipeline, method='points', dsname='', **kwargs):
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

        # define a signal so that people can be notified when we are updated (currently used to force a redraw when parameters change)
        self.on_update = dispatch.Signal()

        # define responses to changes in various traits
        self.on_trait_change(self._update, 'faceColour')
        self.on_trait_change(lambda: self.on_update.send(self), 'visible')
        self.on_trait_change(self.update, 'cmap, clim, alpha, dsname')
        self.on_trait_change(self._set_method, 'method')

        # update any of our traits which were passed as command line arguments
        self.set(**kwargs)

        # update datasource name and method
        self.dsname = dsname
        self.method = method

        # if we were given a pipeline, connect ourselves to the onRebuild signal so that we can automatically update ourselves
        if not self._pipeline is None:
            self._pipeline.onRebuild.connect(self.update)

    @property
    def datasource(self):
        """
        Return the datasource we are connected to (through our dsname property).
        """
        return self._pipeline.get_layer_data(self.dsname)

    def _set_method(self):
        self.engine = ENGINES[self.method]()
        self.update()

    def _get_cdata(self):
        try:
            cdata = self.datasource[self.faceColour]
        except KeyError:
            cdata = np.array([0, 1])

        return cdata

    def _update(self, *args, **kwargs):
        cdata = self._get_cdata()
        self.clim = [float(cdata.min()), float(cdata.max())]
        # self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        print('lw update')
        self._datasource_choices = self._pipeline.layer_data_source_names
        if not self.datasource is None:
            self._datasource_keys = self.datasource.keys()

        if not (self.engine is None or self.datasource is None):
            self.update_from_datasource(self.datasource)
            self.on_update.send(self)

    @property
    def bbox(self):
        return self._bbox

    def update_from_datasource(self, ds):
        x, y = ds[self.x_key], ds[self.y_key]

        if not self.z_key is None:
            z = ds[self.z_key]
        else:
            z = 0 * x

        if not self.vertexColour == '':
            c = ds[self.vertexColour]
        else:
            c = None

        if self.xn_key in ds.keys():
            xn, yn, zn = ds[self.xn_key], ds[self.yn_key], ds[self.zn_key]
            self.update_data(x, y, z, c, cmap=getattr(cm, self.cmap), clim=self.clim, alpha=self.alpha, xn=xn, yn=yn,
                             zn=zn)
        else:
            self.update_data(x, y, z, c, cmap=getattr(cm, self.cmap), clim=self.clim, alpha=self.alpha)

    def update_data(self, x=None, y=None, z=None, colors=None, cmap=None, clim=None, alpha=1.0, xn=None, yn=None,
                    zn=None):
        self._vertices = None
        self._normals = None
        self._colors = None
        self._color_map = None
        self._color_limit = 0
        self._alpha = 0
        if x is not None and y is not None and z is not None:
            vertices = np.vstack((x.ravel(), y.ravel(), z.ravel()))
            vertices = vertices.T.ravel().reshape(len(x.ravel()), 3)

            if not xn is None:
                normals = np.vstack((xn.ravel(), yn.ravel(), zn.ravel()))
            else:
                normals = -0.69 * np.ones(vertices.shape)

            self._bbox = np.array([x.min(), y.min(), z.min(), x.max(), y.max(), z.max()])
        else:
            vertices = None
            normals = None
            self._bbox = None

        if clim is not None and colors is not None and clim is not None:
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

        return View([Group([Item('dsname', label='Data', editor=EnumEditor(name='_datasource_choices')), ]),
                     Item('method'),
                     Item('faceColour', editor=EnumEditor(name='_datasource_keys'), label='Colour'),
                     Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata), show_label=False), ]),
                     Group([Item('cmap', label='LUT'), Item('alpha'), Item('visible')])], )
        # buttons=['OK', 'Cancel'])

    def default_traits_view(self):
        return self.default_view