#!/usr/bin/python

# AxesOverlayLayer.py
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


class AxesOverlayLayer(OverlayLayer):
    """
    This OverlayLayer produces axes and displays the orientation of the model.
    """

    def __init__(self, offset=None, size=1, **kwargs):
        if not offset:
            offset = [10, 10]
        super(AxesOverlayLayer, self).__init__(offset, **kwargs)
        self._size = size

    def render(self, gl_canvas):
        """

        Parameters
        ----------
        gl_canvas
            the rotation matrix is used to get the current orientation
            the size of the canvas is also used

        Returns
        -------

        """
        if not self.visible:
            return
        
        self._clear_shader_clipping(gl_canvas)
        with self.get_shader_program(gl_canvas):
            glDisable(GL_LIGHTING)
            glPushMatrix()

            view_ratio = float(gl_canvas.Size[1])/float(gl_canvas.Size[0])
            glTranslatef(.9, .9*view_ratio, 0)
            glScalef(.1, .1, .1)
            glLineWidth(3)
            glMultMatrixf(gl_canvas.object_rotation_matrix)

            glColor3fv([1, .5, .5])
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(self._size, 0, 0)
            glEnd()

            glColor3fv([.5, 1, .5])
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(0, self._size, 0)
            glEnd()

            glColor3fv([.5, .5, 1])
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, self._size)
            glEnd()

            glLineWidth(1)

            glPopMatrix()
            glEnable(GL_LIGHTING)
