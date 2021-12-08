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

from .OverlayLayer import OverlayLayer
from OpenGL.GL import *


class ScaleBoxOverlayLayer(OverlayLayer):
    def __init__(self, tick_distance=1000, color=None, **kwargs):
        """
        
        Parameters
        ----------
        color       [r, g, b] , rgb = float [0.0,1.0]
        """
        
        
        offset = None
        super(ScaleBoxOverlayLayer, self).__init__(offset, **kwargs)
        if not color:
            self._color = [1.0, 1.0, 1.0, 0.5]
        else:
            self._color = color

        self._tick_distance = tick_distance
        #self._box_dimensions = None
        #self._starts = [0.0, 0.0, 0.0]
        #self.set_box_dimensions(box_dimensions)
        self._show = False
        self._flips = [False, False, False]

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
        raise NotImplementedError('box dimensions are now inferred from the data bounding-box')
        
        new_box_dimensions = np.zeros((3, 1))
        index = 0
        for value in box_dimensions:
            if value < self._tick_distance:
                new_box_dimensions[index] = self._tick_distance
            else:
                new_box_dimensions[index] = value
            index += 1

        self._box_dimensions = new_box_dimensions

    def set_color(self, color):
        self._color = color

    def set_starts(self, start_x, start_y, start_z):
        self._starts = [start_x, start_y, start_z]

    def flip_starts(self, flip_x=None, flip_y=None, flip_z=None):
        """
        There's six possible sides of the bounding box. Only three are used
        to draw the grids. With this method you can change to the other ones.
        Parameters
        ----------
        flip_x      change grid position in the x plane
        flip_y      change grid position in the y plane
        flip_z      change grid position in the z plane

        Returns
        -------

        """

        if flip_x:
            self._flips[0] = not self._flips[0]

        if flip_y:
            self._flips[1] = not self._flips[1]

        if flip_z:
            self._flips[2] = not self._flips[2]

    def render(self, gl_canvas):
        if self._show and self.visible: #handle old _show as well as the standard .visible
            bbox = gl_canvas.bbox
    
            if bbox is None:
                return
            
            self._clear_shader_clipping(gl_canvas)
            with self.get_shader_program(gl_canvas):
                glDisable(GL_LIGHTING)
                glColor4fv(self._color)
                
               
                
                bb_0 = bbox[:3]
                bb_1 = bbox[3:]

                original_starts = np.copy(bb_0)
                
                bb_dims = bb_1 - bb_0
                
                delta_x, delta_y, delta_z = bb_dims

                for i in np.arange(len(self._flips)):
                    if self._flips[i]:
                        original_starts[i] += bb_dims[i]

                [start_x, start_y, start_z] = original_starts
                [direction_x, direction_y, direction_z] = (np.array(self._flips) - 0.5) * -2

                amount_of_lines_x = int(floor(delta_x / self._tick_distance)) + 1
                amount_of_lines_y = int(floor(delta_y / self._tick_distance)) + 1
                amount_of_lines_z = int(floor(delta_z / self._tick_distance)) + 1

                # wall xy
                glBegin(GL_LINES)
                # top
                glVertex(start_x + amount_of_lines_x * self._tick_distance * direction_x,
                         start_y,
                         start_z)
                glVertex(start_x + amount_of_lines_x * self._tick_distance * direction_x,
                         start_y + amount_of_lines_y * self._tick_distance * direction_y,
                         start_z)
                # ticks
                for step in np.arange(amount_of_lines_y + 1):
                    glVertex(start_x,
                             start_y + step * self._tick_distance * direction_y,
                             start_z)
                    glVertex(start_x + amount_of_lines_x * self._tick_distance * direction_x,
                             start_y + step * self._tick_distance * direction_y,
                             start_z)

                for step in np.arange(amount_of_lines_x + 1):
                    glVertex(start_x + step * self._tick_distance * direction_x,
                             start_y,
                             start_z)
                    glVertex(start_x + step * self._tick_distance * direction_x,
                             start_y + amount_of_lines_y * self._tick_distance * direction_y,
                             start_z)

                # bottom
                glVertex(start_x,
                         start_y,
                         start_z)
                glVertex(start_x,
                         start_y + amount_of_lines_y * self._tick_distance * direction_y,
                         start_z)
                glEnd()

                # wall xz
                glBegin(GL_LINES)
                # top
                glVertex(start_x + amount_of_lines_x * self._tick_distance * direction_x,
                         start_y,
                         start_z)
                glVertex(start_x + amount_of_lines_x * self._tick_distance * direction_x,
                         start_y,
                         start_z + amount_of_lines_z * self._tick_distance * direction_z)
                # ticks
                for step in np.arange(amount_of_lines_z + 1):
                    glVertex(start_x,
                             start_y,
                             start_z + step * self._tick_distance * direction_z)
                    glVertex(start_x + amount_of_lines_x * self._tick_distance * direction_x,
                             start_y,
                             start_z + step * self._tick_distance * direction_z)

                for step in np.arange(amount_of_lines_x + 1):
                    glVertex(start_x + step * self._tick_distance * direction_x,
                             start_y,
                             start_z)
                    glVertex(start_x + step * self._tick_distance * direction_x,
                             start_y,
                             start_z + amount_of_lines_z * self._tick_distance * direction_z)

                # bottom
                glVertex(start_x,
                         start_y,
                         start_z)
                glVertex(start_x,
                         start_y,
                         start_z + amount_of_lines_z * self._tick_distance * direction_z)
                glEnd()

                # wall yz
                glBegin(GL_LINES)
                # top
                glVertex(start_x,
                         start_y + amount_of_lines_y * self._tick_distance * direction_y,
                         start_z)
                glVertex(start_x,
                         start_y + amount_of_lines_y * self._tick_distance * direction_y,
                         start_z + amount_of_lines_z * self._tick_distance * direction_z)
                # ticks
                for step in np.arange(amount_of_lines_z + 1):
                    glVertex(start_x,
                             start_y,
                             start_z + step * self._tick_distance * direction_z)
                    glVertex(start_x,
                             start_y + amount_of_lines_y * self._tick_distance * direction_y,
                             start_z + step * self._tick_distance * direction_z)

                for step in np.arange(amount_of_lines_y + 1):
                    glVertex(start_x,
                             start_y + step * self._tick_distance * direction_y,
                             start_z)
                    glVertex(start_x,
                             start_y + step * self._tick_distance * direction_y,
                             start_z + amount_of_lines_z * self._tick_distance * direction_z)

                # bottom
                glVertex(start_x,
                         start_y,
                         start_z)
                glVertex(start_x,
                         start_y,
                         start_z + amount_of_lines_z * self._tick_distance * direction_z)
                glEnd()
