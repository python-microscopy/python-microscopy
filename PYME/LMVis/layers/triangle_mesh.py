from .base import BaseEngine, EngineLayer
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram
from PYME.LMVis.shader_programs.PointSpriteShaderProgram import PointSpriteShaderProgram
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

class FlatEngine(WireframeEngine):
    def __init__(self):
        BaseEngine.__init__(self)

    def render(self, gl_canvas, layer):
        glShadeModel(GL_FLAT)
        WireframeEngine.render(self, gl_canvas, layer)

class ShadedEngine(WireframeEngine):
    def __init__(self):
        BaseEngine.__init__(self)

    def render(self, gl_canvas, layer):
        glShadeModel(GL_SMOOTH)
        WireframeEngine.render(self, gl_canvas, layer)

class TrianglesRenderLayer(EngineLayer):
