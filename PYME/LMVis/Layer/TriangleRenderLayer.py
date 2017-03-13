#!/usr/bin/python

# TriangleRenderLayer.py
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
from PYME.LMVis.Layer.RenderLayer import RenderLayer
from PYME.LMVis.ShaderProgram.GouraudShaderProgram import GouraudShaderProgram
from PYME.LMVis.ShaderProgram.WireFrameShaderProgram import WireFrameShaderProgram
from PYME.LMVis.gen3DTriangs import gen3DTriangs
from OpenGL.GL import *


class TriangleRenderLayer(RenderLayer):
    """
    This program draws a WareFrame of the given points. They are interpreted as triangles.
    """

    DRAW_MODE = GL_TRIANGLES

    def __init__(self, x, y, z, colors, color_map, size_cutoff, internal_cull, z_rescale, alpha, is_wire_frame=False):
        p, a, n = gen3DTriangs(x, y, z / z_rescale, size_cutoff, internalCull=internal_cull)
        if colors == 'z':
            colors = p[:, 2]
        else:
            colors = 1. / a
        color_limit = [colors.min(), colors.max()]
        super(TriangleRenderLayer, self).__init__(colors=colors, color_map=color_map, color_limit=color_limit,
                                                  alpha=alpha)
        self.set_values(p, n)
        if is_wire_frame:
            self.set_shader_program(WireFrameShaderProgram())
        else:
            self.set_shader_program(GouraudShaderProgram())

    def render(self, gl_canvas):
        """

        Parameters
        ----------
        gl_canvas
            nothing of the canvas is used. That's how it should be.
        Returns
        -------

        """
        with self.get_shader_program():
            n_vertices = self.get_vertices().shape[0]

            glVertexPointerf(self.get_vertices())
            glNormalPointerf(self.get_normals())
            glColorPointerf(self.get_colors())

            glPushMatrix()
            glDrawArrays(self.DRAW_MODE, 0, n_vertices)

            glPopMatrix()
