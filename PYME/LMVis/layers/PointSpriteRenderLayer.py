#!/usr/bin/python

# PointSpriteRenderLayer.py
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
from OpenGL.GL import *
from PYME.LMVis.layers.Point3DRenderLayer import Point3DRenderLayer
from PYME.LMVis.shader_programs.PointSpriteShaderProgram import PointSpriteShaderProgram


class PointSpritesRenderLayer(Point3DRenderLayer):
    """
    This class prepares OpenGL for displaying the point clouds using point sprites.

    """

    def __init__(self, x_values, y_values, z_values, colors, color_map, color_limit, alpha, point_size=5):
        """
        This constructor is only used to call the super constructor and set those parameters.
        Some of them may never be used.
        """
        Point3DRenderLayer.__init__(self, x_values, y_values, z_values, colors, color_map, color_limit, alpha,
                                    point_size)
        self.set_shader_program(PointSpriteShaderProgram)
        self.set_point_size(point_size)

    def set_point_size(self, point_size):
        Point3DRenderLayer.set_point_size(self, point_size * self.get_shader_program().get_size_factor())

    def render(self, gl_canvas=None):
        """
        The OpenGL context isn't available before the OnPaint method.
        That's why we can't initialize the program while initializing the class. Which would be the better fit.
        To solve this problem, we need to check if the program was already created and can be used.
        If not we will create it.


        Parameters
        ----------
        gl_canvas the scene is drawn into
            the pixel size is used to calculate the right point size to match nm,um etc.
        """

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
