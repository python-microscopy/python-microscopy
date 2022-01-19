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
from PYME.LMVis.layers.OverlayLayer import OverlayLayer
from OpenGL.GL import *
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram


class ScaleBarOverlayLayer(OverlayLayer):
    """
    This class creates a scale bar that indicate the size, depending on the zoom level
    """
    def __init__(self, offset=None, color=None, **kwargs):
        super(ScaleBarOverlayLayer, self).__init__(offset, **kwargs)
        if not color:
            self._color = [1.0, 1.0, 1.0]
        else:
            self._color = color
        self._scale_bar_depth = 10
        self._scale_bar_length = 1000
        self.set_shader_program(DefaultShaderProgram)

    def set_scale_bar_length(self, length):
        self._scale_bar_length = length

    def set_scale_bar_depth(self, depth):
        self._scale_bar_depth = depth

    def render(self, gl_canvas):
        """

        Parameters
        ----------
        gl_canvas
            the scaleBarLength is currently used, but should be passed with a setter in the future
            xmin, xmax, ymin, ymax and Size is used to get the total size of the canvas and to calculate
             the real size of the scale bar

        Returns
        -------

        """
        if not self.visible:
            return
        
        self._clear_shader_clipping(gl_canvas)
        with self.get_shader_program(gl_canvas):
            if self._scale_bar_length:
                view_size_x = gl_canvas.xmax - gl_canvas.xmin
                view_size_y = gl_canvas.ymax - gl_canvas.ymin

                sb_ur_x = -gl_canvas.view.translation[0] + gl_canvas.xmax - self.get_offset()[0] * view_size_x / gl_canvas.Size[0]
                #sb_ur_y = - gl_canvas.yc + gl_canvas.ymax - self.get_offset()[1] * view_size_y / gl_canvas.Size[1]
                sb_ur_y = -gl_canvas.view.translation[1] + gl_canvas.ymin + self.get_offset()[1] * view_size_y / gl_canvas.Size[1]
                sb_depth = self._scale_bar_depth * view_size_y / gl_canvas.Size[1]

                glDisable(GL_LIGHTING)

                glColor3fv(self._color)
                glBegin(GL_POLYGON)
                glVertex3f(sb_ur_x, sb_ur_y, 0)
                glVertex3f(sb_ur_x, sb_ur_y + sb_depth, 0)
                glVertex3f(sb_ur_x - self._scale_bar_length, sb_ur_y + sb_depth, 0)
                glVertex3f(sb_ur_x - self._scale_bar_length, sb_ur_y, 0)
                glEnd()
