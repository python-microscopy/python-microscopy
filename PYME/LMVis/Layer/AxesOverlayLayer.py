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
from PYME.LMVis.Layer.OverlayLayer import OverlayLayer
from OpenGL.GL import *


class AxesOverlayLayer(OverlayLayer):

    _size = 1

    def __init__(self, offset=None):
        if not offset:
            offset = [10, 10]
        super(AxesOverlayLayer, self).__init__(offset)

    def render(self, gl_canvas):
        with gl_canvas.defaultProgram:
            glDisable(GL_LIGHTING)
            glPushMatrix()

            view_ratio = float(gl_canvas.Size[1])/float(gl_canvas.Size[0])
            glTranslatef(.9, .1 - view_ratio, 0)
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
