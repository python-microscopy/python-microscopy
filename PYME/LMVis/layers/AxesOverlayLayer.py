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
import numpy as np

from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram, FatLineShaderProgram

class AxesOverlayLayer(OverlayLayer):
    """
    This OverlayLayer produces axes and displays the orientation of the model.
    """

    def __init__(self, offset=None, size=0.1, **kwargs):
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
        
        if gl_canvas.core_profile:
            sp = self.get_specific_shader_program(gl_canvas, FatLineShaderProgram)
        else:
            sp = self.get_specific_shader_program(gl_canvas, DefaultShaderProgram) 


        with sp:
            sp.clear_shader_clipping()

            view_ratio = float(gl_canvas.Size[1])/float(gl_canvas.Size[0])

            glEnable(GL_LINE_SMOOTH)

            #glDisable(GL_LIGHTING)
            if gl_canvas.core_profile:
                import PYME.LMVis.mv_math as mm
                mv = mm.translate(gl_canvas.mv, .88, .88*view_ratio, -0.5)
                mv = np.dot(mv, gl_canvas.object_rotation_matrix.T)
                mvp = np.dot(gl_canvas.proj, mv)

                sp.set_modelviewprojectionmatrix(np.array(mvp))

                glUniform1f(sp.get_uniform_location('line_width_px'), 3.0*gl_canvas.content_scale_factor)
                vp = glGetIntegerv(GL_VIEWPORT)
                glUniform2f(sp.get_uniform_location('viewport_size'), vp[2], vp[3])

            else:
                glPushMatrix()
                glTranslatef(.9, .9*view_ratio,0)
                #glScalef(.1, .1, .1)
                
                glMultMatrixf(gl_canvas.object_rotation_matrix)

            verts = np.array([[0, 0, 0], [self._size, 0, 0], 
                              [0,0,0], [0, self._size, 0], 
                              [0,0,0], [0, 0, self._size]], 'f')
            
            colors = np.array([[1, .5, .5, 1], [1, .5, .5, 1], 
                               [.5, 1, .5, 1], [.5, 1, .5, 1], 
                               [.5, .5, 1, 1], [.5, .5, 1, 1]], 'f')
            
            self._bind_data('axes', verts, 0*verts, colors, sp, core_profile=gl_canvas.core_profile)
            
            lw = min(3.0, glGetFloatv(GL_LINE_WIDTH_RANGE)[1])
            glLineWidth(lw)
            
            glDrawArrays(GL_LINES, 0, 6)
            glLineWidth(1.0)

            # glColor3fv([1, .5, .5])
            # glBegin(GL_LINES)
            # glVertex3f(0, 0, 0)
            # glVertex3f(self._size, 0, 0)
            # glEnd()

            # glColor3fv([.5, 1, .5])
            # glBegin(GL_LINES)
            # glVertex3f(0, 0, 0)
            # glVertex3f(0, self._size, 0)
            # glEnd()

            # glColor3fv([.5, .5, 1])
            # glBegin(GL_LINES)
            # glVertex3f(0, 0, 0)
            # glVertex3f(0, 0, self._size)
            # glEnd()

            

            if not gl_canvas.core_profile:
                glPopMatrix()

            #glEnable(GL_LIGHTING)
