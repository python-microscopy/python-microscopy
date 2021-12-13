#!/usr/bin/python

# SelectionOverlayLayer.py
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


class SelectionOverlayLayer(OverlayLayer):

    """
    This OverlayLayer creates the lines of a box.
    The dimensions of the box are determined by the selection_settings
    """
    def __init__(self, selection_settings, **kwargs):
        super(SelectionOverlayLayer, self).__init__([0, 0], **kwargs)
        self._selection_settings = selection_settings

    def render(self, gl_canvas):
        """

        Parameters
        ----------
        gl_canvas
            zc is used to set the z value of the Overlay
        Returns
        -------

        """
        if not self.visible:
            return
        
        self._clear_shader_clipping(gl_canvas)
        with self.get_shader_program(gl_canvas):
            if self._selection_settings.show:
                glDisable(GL_LIGHTING)
                x0, y0 = self._selection_settings.start
                x1, y1 = self._selection_settings.finish

                zc = gl_canvas.view.translation[2]

                glColor3fv(self._selection_settings.colour)
                glBegin(GL_LINE_LOOP)
                glVertex3f(x0, y0, zc)
                glVertex3f(x1, y0, zc)
                glVertex3f(x1, y1, zc)
                glVertex3f(x0, y1, zc)
                glEnd()
