#!/usr/bin/python

# Point3DRenderLayer.py
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
from PYME.LMVis.Layer.RenderLayer import RenderLayer


class Point3DRenderLayer(RenderLayer):

    _point_size = 1

    def __init__(self,  x, y, z, colors, color_map, color_limit, alpha, point_size=5):
        RenderLayer.__init__(self, x, y, z, colors, color_map, color_limit, alpha)
        self._point_size = point_size

    def render(self, gl_canvas):
        pass

    def get_point_size(self):
        return self._point_size
