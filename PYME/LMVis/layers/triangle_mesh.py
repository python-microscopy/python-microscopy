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