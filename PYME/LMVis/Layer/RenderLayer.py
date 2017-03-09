#!/usr/bin/python

# RenderLayer.py
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
from abc import abstractmethod

import numpy
from PYME.LMVis.Layer.Layer import Layer


class RenderLayer(Layer):

    _vertices = None
    _normals = None
    _colors = None
    _color_map = None
    _color_limit = 0
    _alpha = 0

    def __init__(self, x, y, z, colors, color_map, color_limit, alpha, process=False):
        Layer.__init__(self)
        if process:
            vertices = numpy.vstack((x.ravel(), y.ravel(), z.ravel()))
            vertices = vertices.T.ravel().reshape(len(x.ravel()), 3)
            normals = -0.69 * numpy.ones(vertices.shape)
            cs_ = ((colors - color_limit[0]) / (color_limit[1] - color_limit[0]))
            cs = color_map(cs_)
            cs[:, 3] = alpha

            cs = cs.ravel().reshape(len(colors), 4)

            self.set_values(vertices, normals, cs, color_map, color_limit, alpha)

    @abstractmethod
    def render(self, gl_canvas):
        pass

    def set_values(self, vertices=None, normals=None, colors=None, color_map=None, color_limit=None, alpha=None):

        if vertices is not None:
            self._vertices = vertices
        if normals is not None:
            self._normals = normals
        if color_map is not None:
            self._color_map = color_map
        if colors is not None:
            self._colors = colors
        if color_limit is not None:
            self._color_limit = color_limit
        if alpha is not None:
            self._alpha = alpha

    def get_vertices(self):
        return self._vertices

    def get_normals(self):
        return self._normals

    def get_colors(self):
        return self._colors

    def get_color_map(self):
        return self._color_map

    def get_color_limit(self):
        return self._color_limit
