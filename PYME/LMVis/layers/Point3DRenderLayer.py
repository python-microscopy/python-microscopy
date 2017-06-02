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


class Point3DRenderLayer(VertexRenderLayer):

    def __init__(self,  x=None, y=None, z=None, colors=None, color_map=None, color_limit=None, alpha=1.0, point_size=5):
        VertexRenderLayer.__init__(self, x, y, z, colors, color_map, color_limit, alpha)
        self._point_size = point_size
        self.set_shader_program(DefaultShaderProgram)

    def render(self, gl_canvas):
        with self.get_shader_program():

            n_vertices = self.get_vertices().shape[0]

            glVertexPointerf(self.get_vertices())
            glNormalPointerf(self.get_normals())
            glColorPointerf(self.get_colors())

            if gl_canvas:
                if self.get_point_size() == 0:
                    glPointSize(1 / gl_canvas.pixelsize)
                else:
                    glPointSize(self.get_point_size() / gl_canvas.pixelsize)
            else:
                glPointSize(self.get_point_size())
            glDrawArrays(GL_POINTS, 0, n_vertices)

    def get_point_size(self):
        return self._point_size

    def set_point_size(self, point_size):
        self._point_size = point_size
