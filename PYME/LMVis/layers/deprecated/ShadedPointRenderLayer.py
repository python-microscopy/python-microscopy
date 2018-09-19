#!/usr/bin/python

# ShadedPointRenderLayer.py
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
import numpy as np

from PYME.LMVis.layers.Point3DRenderLayer import Point3DRenderLayer
from PYME.LMVis.shader_programs.GouraudShaderProgram import GouraudShaderProgram


class ShadedPointRenderLayer(Point3DRenderLayer):

    def __init__(self, x, y, z, normal_x, normal_y, normal_z, colors, color_map, color_limit, alpha, point_size=5):
        Point3DRenderLayer.__init__(self, x, y, z, colors, color_map, color_limit, alpha, point_size=point_size)
        self.set_shader_program(GouraudShaderProgram)
        self.set_values(normals=self.calculate_normals(normal_x, normal_y, normal_z))

    @staticmethod
    def calculate_normals(normal_x, normal_y, normal_z):
        normals = np.vstack((normal_x.ravel(), normal_y.ravel(), normal_z.ravel()))
        return normals.T.ravel().reshape(len(normal_x.ravel()), 3)
