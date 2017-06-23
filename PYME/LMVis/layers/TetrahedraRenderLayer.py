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
from PYME.Analysis.points.gen3DTriangs import gen3DTriangs
from PYME.LMVis.layers.VertexRenderLayer import VertexRenderLayer
from PYME.LMVis.shader_programs.GouraudShaderProgram import GouraudShaderProgram
from PYME.LMVis.shader_programs.WireFrameShaderProgram import WireFrameShaderProgram
from OpenGL.GL import *

import numpy as np
from PYME.recipes.traits import Float, Bool

class TetrahedraRenderLayer(VertexRenderLayer):
    """
    This program draws a WareFrame of the given points. They are interpreted as triangles.
    """
    z_rescale = Float(1.0)
    size_cutoff = Float(1000.)
    internal_cull = Bool(True)
    wireframe = Bool(False)

    DRAW_MODE = GL_TRIANGLES
    

    def __init__(self, x, y, z, colors, color_map, size_cutoff, internal_cull, z_rescale, alpha, is_wire_frame=False):
        p, a, n = gen3DTriangs(x, y, z / z_rescale, size_cutoff, internalCull=internal_cull)
        if colors == 'z':
            colors = p[:, 2]
        else:
            colors = 1. / a
        color_limit = [colors.min(), colors.max()]
        super(TetrahedraRenderLayer, self).__init__(colors=colors, color_map=color_map, color_limit=color_limit,
                                                    alpha=alpha)
        self.set_values(p, n)
        if is_wire_frame:
            self.set_shader_program(WireFrameShaderProgram)
        else:
            self.set_shader_program(GouraudShaderProgram)

    def update_from_datasource(self, ds, cmap=None, clim=None, alpha=1.0):
        x, y = ds[self.x_key], ds[self.y_key]
    
        if not self.z_key is None:
            z = ds[self.z_key]
        else:
            z = 0 * x

        p, a, n = gen3DTriangs(x, y, z / z_rescale, size_cutoff, internalCull=internal_cull)
    
        if not self.vertexColour == '':
            c = ds[self.vertexColour]
        else:
            c = None
    
        self.update_data(x, y, z, c, cmap=cmap, clim=clim, alpha=alpha)

    def update_data(self, x=None, y=None, z=None, colors=None, cmap=None, clim=None, alpha=1.0):
        self._vertices = None
        self._normals = None
        self._colors = None
        self._color_map = None
        self._color_limit = 0
        self._alpha = 0
        
        if x is not None and y is not None and z is not None:
            vertices = np.vstack((x.ravel(), y.ravel(), z.ravel()))
            vertices = vertices.T.ravel().reshape(len(x.ravel()), 3)
            normals = -0.69 * np.ones(vertices.shape)
        else:
            vertices = None
            normals = None
    
        if clim is not None and colors is not None and clim is not None:
            cs_ = ((colors - clim[0]) / (clim[1] - clim[0]))
            cs = cmap(cs_)
            cs[:, 3] = alpha
        
            cs = cs.ravel().reshape(len(colors), 4)
        else:
            #cs = None
            if not vertices is None:
                cs = np.ones((vertices.shape[0], 4), 'f')
            else:
                cs = None
            color_map = None
            color_limit = None
    
        self.set_values(vertices, normals, cs, cmap, clim, alpha)

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
