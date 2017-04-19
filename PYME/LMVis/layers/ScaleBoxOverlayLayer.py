#!/usr/bin/python

# ScaleBoxOverlayLayer.py
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
from math import floor

import numpy as np

from PYME.LMVis.Layer.OverlayLayer import OverlayLayer
from OpenGL.GL import *


class ScaleBoxOverlayLayer(OverlayLayer):
    def __init__(self, tick_distance, color=None, box_dimensions=(1.0, 1.0, 1.0)):
        """
        
        Parameters
        ----------
        color       [r, g, b] , rgb = float [0.0,1.0]
        """
        offset = None
        super(ScaleBoxOverlayLayer, self).__init__(offset)
        if not color:
            self._color = [1.0, 1.0, 1.0, 0.5]
        else:
            self._color = color

        self._tick_distance = tick_distance
        self._box_dimensions = None
        self._starts = (0.0, 0.0, 0.0)
        self.set_box_dimensions(box_dimensions)
        self._show = False

    def show(self, boolean):
        self._show = boolean

    def set_tick_distance(self, value):
        """
        Set the distance between two ticks in the box
        Parameters
        ----------
        value   float, >0

        Returns
        -------

        """
        if value is not None:
            self._tick_distance = value

    def get_tick_distance(self):
        return self._tick_distance

    def set_box_dimensions(self, box_dimensions):
        """
        
        Parameters
        ----------
        box_dimensions  (width, height, depth) or (x/y/z) in nm 
                        is smaller than tick size, tick size is used

        Returns
        -------

        """
        new_box_dimensions = np.zeros((3, 1))
        index = 0
        for value in box_dimensions:
            if value < self._tick_distance:
                new_box_dimensions[index] = self._tick_distance
            else:
                new_box_dimensions[index] = value
            index += 1

        self._box_dimensions = new_box_dimensions

    def set_starts(self, start_x, start_y, start_z):
        self._starts = (start_x, start_y, start_z)

    def render(self, gl_canvas):
        if self._show:
            with self.get_shader_program():
                glDisable(GL_LIGHTING)
                glColor4fv(self._color)

                start_x, start_y, start_z = self._starts
                delta_x, delta_y, delta_z = self._box_dimensions

                amount_of_lines_x = int(floor(delta_x / self._tick_distance)) + 1
                amount_of_lines_y = int(floor(delta_y / self._tick_distance)) + 1
                amount_of_lines_z = int(floor(delta_z / self._tick_distance)) + 1

                # wall xy
                glBegin(GL_LINES)
                # top
                glVertex(start_x + amount_of_lines_x * self._tick_distance,
                         start_y,
                         start_z)
                glVertex(start_x + amount_of_lines_x * self._tick_distance,
                         start_y + amount_of_lines_y * self._tick_distance,
                         start_z)
                # ticks
                for step in np.arange(amount_of_lines_y + 1):
                    glVertex(start_x,
                             start_y + step * self._tick_distance,
                             start_z)
                    glVertex(start_x + amount_of_lines_x * self._tick_distance,
                             start_y + step * self._tick_distance,
                             start_z)

                # bottom
                glVertex(start_x,
                         start_y,
                         start_z)
                glVertex(start_x,
                         start_y + amount_of_lines_y * self._tick_distance,
                         start_z)
                glEnd()

                # wall xz
                glBegin(GL_LINES)
                # top
                glVertex(start_x + amount_of_lines_x * self._tick_distance,
                         start_y,
                         start_z)
                glVertex(start_x + amount_of_lines_x * self._tick_distance,
                         start_y,
                         start_z + amount_of_lines_z * self._tick_distance)
                # ticks
                for step in np.arange(amount_of_lines_z + 1):
                    glVertex(start_x,
                             start_y,
                             start_z + step * self._tick_distance)
                    glVertex(start_x + amount_of_lines_x * self._tick_distance,
                             start_y,
                             start_z + step * self._tick_distance)

                # bottom
                glVertex(start_x,
                         start_y,
                         start_z)
                glVertex(start_x,
                         start_y,
                         start_z + amount_of_lines_z * self._tick_distance)
                glEnd()

                # wall yz
                glBegin(GL_LINES)
                # top
                glVertex(start_x,
                         start_y + amount_of_lines_y * self._tick_distance,
                         start_z)
                glVertex(start_x,
                         start_y + amount_of_lines_y * self._tick_distance,
                         start_z + amount_of_lines_z * self._tick_distance)
                # ticks
                for step in np.arange(amount_of_lines_z + 1):
                    glVertex(start_x,
                             start_y,
                             start_z + step * self._tick_distance)
                    glVertex(start_x,
                             start_y + amount_of_lines_y * self._tick_distance,
                             start_z + step * self._tick_distance)

                # bottom
                glVertex(start_x,
                         start_y,
                         start_z)
                glVertex(start_x,
                         start_y,
                         start_z + amount_of_lines_z * self._tick_distance)
                glEnd()

