#!/usr/bin/python

# GLProgram.py
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
from OpenGL.GL import *
import numpy as np

class GLProgram(object):

    def __init__(self):
        self._shader_program = None

        self.xmin, self.xmax = [-1e6, 1e6]
        self.ymin, self.ymax = [-1e6, 1e6]
        self.zmin, self.zmax = [-1e6, 1e6]
        self.vmin, self.vmax = [-1e6, 1e6]
        self.v_matrix = np.eye(4, 4, dtype='f')

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(0)

    def set_shader_program(self, shader_program):
        self._shader_program = shader_program

    def get_shader_program(self):
        return self._shader_program

    def get_uniform_location(self, uniform_name):
        return self._shader_program.get_uniform_location(uniform_name)
