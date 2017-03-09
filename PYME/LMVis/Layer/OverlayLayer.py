#!/usr/bin/python

# OverlayLayer.py
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
import abc

from PYME.LMVis.Layer.Layer import Layer


class OverlayLayer(Layer):

    _offset = [10, 10]

    def __init__(self, offset):
        super(OverlayLayer, self).__init__()
        self._offset = offset

    def set_offset(self, offset):
        self._offset = offset

    def get_offset(self):
        return self._offset

    @abc.abstractmethod
    def render(self, gl_canvas):
        pass
