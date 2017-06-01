#!/usr/bin/python

# QuadTreeRenderLayer.py
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


class QuadTreeRenderLayer(VertexRenderLayer):

    def __init__(self,  x, y, z, colors, color_map, color_limit, alpha):
        VertexRenderLayer.__init__(self, x, y, z, colors, color_map, color_limit, alpha)

    def render(self, gl_canvas):
        with self.get_shader_program():

            n_vertices = self.get_vertices().shape[0]

            glVertexPointerf(self.get_vertices())
            glNormalPointerf(self.get_normals())
            glColorPointerf(self.get_colors())

            glDrawArrays(GL_QUADS, 0, n_vertices)

