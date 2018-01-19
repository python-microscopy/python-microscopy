#!/usr/bin/python

# Point3DRenderLayer.py
#
# Copyright Michael Graff
#   graff@hm.edu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from PYME.LMVis.layers.VertexRenderLayer import VertexRenderLayer
from OpenGL.GL import *

from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram
import warnings
from PYME.recipes.traits import Float


class Point3DRenderLayer(VertexRenderLayer):
    point_size = Float(5.0)

    def __init__(self,  x=None, y=None, z=None, colors=None, color_map=None, color_limit=None, alpha=1.0, point_size=5):
        VertexRenderLayer.__init__(self, x, y, z, colors, color_map, color_limit, alpha)
        self.point_size = point_size
        self.set_shader_program(DefaultShaderProgram)

    def render(self, gl_canvas):
        with self.shader_program:

            n_vertices = self.get_vertices().shape[0]

            glVertexPointerf(self.get_vertices())
            glNormalPointerf(self.get_normals())
            glColorPointerf(self.get_colors())

            if gl_canvas:
                if self.point_size == 0:
                    glPointSize(1 / gl_canvas.pixelsize)
                else:
                    glPointSize(self.point_size / gl_canvas.pixelsize)
            else:
                glPointSize(self.point_size)
            glDrawArrays(GL_POINTS, 0, n_vertices)

    def get_point_size(self):
        warnings.warn("use the point_size property instead", DeprecationWarning)
        return self.point_size

    def set_point_size(self, point_size):
        warnings.warn("use the point_size property instead", DeprecationWarning)
        self.point_size = point_size

    def view(self, ds_keys):
        from traitsui.api import View, Item, Group
        from PYME.ui.custom_traits_editors import CBEditor
    
        return View([Item('vertexColour', editor=CBEditor(choices=ds_keys), label='Colour'),
                     Item('point_size', label='Size [nm]')])
