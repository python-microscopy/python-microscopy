#!/usr/bin/python

# ScaleBarOverlayLayer.py
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
from PYME.LMVis.Layer.OverlayLayer import OverlayLayer
from OpenGL.GL import *
from PYME.LMVis.ShaderProgram.DefaultShaderProgram import DefaultShaderProgram


class ScaleBarOverlayLayer(OverlayLayer):

    _color = None
    _scale_bar_depth = 10.0
    _scale_bar_length = 1000

    def __init__(self, offset=None, color=None):
        super(ScaleBarOverlayLayer, self).__init__(offset)
        if not color:
            self._color = [1.0, 0.0, 0.0]
        self.set_shader_program(DefaultShaderProgram())

    def set_scale_bar_length(self, length):
        self._scale_bar_length = length

    def set_scale_bar_depth(self, depth):
        self._scale_bar_depth = depth

    def render(self, gl_canvas):
        # TODO remove this, when size changes are implemented using the setter.
        self._scale_bar_length = gl_canvas.scaleBarLength

        with self.get_shader_program():
            if self._scale_bar_length:
                view_size_x = gl_canvas.xmax - gl_canvas.xmin
                view_size_y = gl_canvas.ymax - gl_canvas.ymin

                sb_ur_x = -gl_canvas.xc + gl_canvas.xmax - self.get_offset()[0] * view_size_x / gl_canvas.Size[0]
                sb_ur_y = - gl_canvas.yc + gl_canvas.ymax - self.get_offset()[1] * view_size_y / gl_canvas.Size[1]
                sb_depth = self._scale_bar_depth * view_size_y / gl_canvas.Size[1]

                glDisable(GL_LIGHTING)

                glColor3fv(self._color)
                glBegin(GL_POLYGON)
                glVertex3f(sb_ur_x, sb_ur_y, 0)
                glVertex3f(sb_ur_x, sb_ur_y - sb_depth, 0)
                glVertex3f(sb_ur_x - self._scale_bar_length, sb_ur_y - sb_depth, 0)
                glVertex3f(sb_ur_x - self._scale_bar_length, sb_ur_y, 0)
                glEnd()
