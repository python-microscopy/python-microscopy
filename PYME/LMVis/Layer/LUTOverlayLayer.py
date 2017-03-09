#!/usr/bin/python

# LUTOverlayLayer.py
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
import numpy
import pylab

from PYME.LMVis.Layer.OverlayLayer import OverlayLayer
from OpenGL.GL import *
from PYME.LMVis.ShaderProgram.DefaultShaderProgram import DefaultShaderProgram


class LUTOverlayLayer(OverlayLayer):

    _color_map = None
    _size = 1000
    _scale_bar_depth = 10.0
    _color = [1, 1, 0]

    def __init__(self, size=100, offset=None, color_map=pylab.cm.hsv):
        if not offset:
            offset = [10, 10]
        OverlayLayer.__init__(self, offset)
        self._color_map = color_map
        self._offset = offset
        self._size = size
        self.set_shader_program(DefaultShaderProgram())

    def set_color_map(self, color_map):
        self._color_map = color_map

    def render(self, gl_canvas):
        with self.get_shader_program():
            view_size_x = gl_canvas.xmax - gl_canvas.xmin
            view_size_y = gl_canvas.ymax - gl_canvas.ymin

            #upper right x
            lb_ur_x = -gl_canvas.xc + gl_canvas.xmax - self.get_offset()[0] * view_size_x / gl_canvas.Size[0]
            #uper right y
            lb_ur_y = .4 * view_size_y

            #lower right y
            lb_lr_y = -.4 * view_size_y
            lb_width = self._scale_bar_depth * view_size_x / gl_canvas.Size[0]
            #upper left x
            lb_ul_x = lb_ur_x - lb_width

            lb_len = lb_ur_y - lb_lr_y

            glDisable(GL_LIGHTING)

            glBegin(GL_QUAD_STRIP)

            for i in numpy.arange(0, 1, .01):
                glColor3fv(self._color_map(i)[:3])
                glVertex2f(lb_ul_x, lb_lr_y + i * lb_len)
                glVertex2f(lb_ur_x, lb_lr_y + i * lb_len)

            glEnd()

            glBegin(GL_LINE_LOOP)
            glColor3f(.5, .5, 0)
            glVertex2f(lb_ul_x, lb_lr_y)
            glVertex2f(lb_ur_x, lb_lr_y)
            glVertex2f(lb_ur_x, lb_ur_y)
            glVertex2f(lb_ul_x, lb_ur_y)
            glEnd()
