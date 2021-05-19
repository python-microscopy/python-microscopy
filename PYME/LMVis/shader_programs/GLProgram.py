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

from PYME.LMVis.shader_programs.shader_program import ShaderProgram
import os

class GLProgram(object):

    def __init__(self, vs_filename=None, fs_filename=None):
        self._shader_program = None

        self.xmin, self.xmax = [-1e6, 1e6]
        self.ymin, self.ymax = [-1e6, 1e6]
        self.zmin, self.zmax = [-1e6, 1e6]
        self.vmin, self.vmax = [-1e6, 1e6]
        self.v_matrix = np.eye(4, 4, dtype='f')
        
        if (vs_filename is not None) and (fs_filename is not None):
            self.create_and_set_shader_program(vs_filename, fs_filename)

        self._old_prog = 0

    @abc.abstractmethod
    def __enter__(self):
        self._old_prog = glGetInteger(GL_CURRENT_PROGRAM)

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(self._old_prog)

    def create_and_set_shader_program(self, vs_filename, fs_filename):
        shader_path = os.path.join(os.path.dirname(__file__), "shaders")
        shader_program = ShaderProgram(shader_path)
        shader_program.add_shader(vs_filename, GL_VERTEX_SHADER)
        shader_program.add_shader(fs_filename, GL_FRAGMENT_SHADER)
        shader_program.link()
        self.set_shader_program(shader_program)
        
    def set_shader_program(self, shader_program):
        self._shader_program = shader_program

    def get_shader_program(self):
        return self._shader_program

    def get_uniform_location(self, uniform_name):
        return self._shader_program.get_uniform_location(uniform_name)

    def set_clipping_uniforms(self):
        glUniform1f(self.get_uniform_location('x_min'), float(self.xmin))
        glUniform1f(self.get_uniform_location('x_max'), float(self.xmax))
        glUniform1f(self.get_uniform_location('y_min'), float(self.ymin))
        glUniform1f(self.get_uniform_location('y_max'), float(self.ymax))
        glUniform1f(self.get_uniform_location('z_min'), float(self.zmin))
        glUniform1f(self.get_uniform_location('z_max'), float(self.zmax))
        glUniform1f(self.get_uniform_location('v_min'), float(self.vmin))
        glUniform1f(self.get_uniform_location('v_max'), float(self.vmax))
        glUniformMatrix4fv(self.get_uniform_location('clip_rotation_matrix'), 1, GL_FALSE, self.v_matrix)