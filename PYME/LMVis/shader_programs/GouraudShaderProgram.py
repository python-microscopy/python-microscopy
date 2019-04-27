#!/usr/bin/python

# GouraudShaderProgram.py
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
import os
from PYME.LMVis.shader_programs.GLProgram import GLProgram
from OpenGL.GL import *
from PYME.LMVis.shader_programs.shader_program import ShaderProgram
import numpy as np

class GouraudShaderProgram(GLProgram):
    INPUT_LIGHT = b'inputLight'
    test_value = [1.0, 0.0, 1.0, 1.0]
    lights_vector = (0.0, 0.00, -1.0, 0.0)
    view_vector = (0.0, 0.0, -1.0, 0.0)

    LIGHT_PROPS = {
            'light_ambient': (0.1, 0.1, 0.1, 1.0),
            'light_diffuse': (0.6, 0.6, 0.6, 1.0),
            'light_specular': (0.3, 0.3, 0.3, 1.0),
            'light_position': lights_vector
    }

    def __init__(self):
        GLProgram.__init__(self)
        shader_path = os.path.join(os.path.dirname(__file__), "shaders")
        shader_program = ShaderProgram(shader_path)
        shader_program.add_shader("gouraud_vs.glsl", GL_VERTEX_SHADER)
        shader_program.add_shader("gouraud_fs.glsl", GL_FRAGMENT_SHADER)
        shader_program.link()
        self.set_shader_program(shader_program)
        self._shininess = 80
        

    def __enter__(self):
        self.get_shader_program().use()
        for name, value in GouraudShaderProgram.LIGHT_PROPS.items():
            location = self.get_uniform_location(name)
            glUniform4f(location, *value)
        location = self.get_uniform_location('shininess')
        glUniform1f(location, self._shininess)
        location = self.get_uniform_location('view_vector')
        glUniform4f(location, *self.view_vector)

        glUniform1f(self.get_uniform_location('x_min'), float(self.xmin))
        glUniform1f(self.get_uniform_location('x_max'), float(self.xmax))
        glUniform1f(self.get_uniform_location('y_min'), float(self.ymin))
        glUniform1f(self.get_uniform_location('y_max'), float(self.ymax))
        glUniform1f(self.get_uniform_location('z_min'), float(self.zmin))
        glUniform1f(self.get_uniform_location('z_max'), float(self.zmax))
        glUniform1f(self.get_uniform_location('v_min'), float(self.vmin))
        glUniform1f(self.get_uniform_location('v_max'), float(self.vmax))

        #glUniformMatrix4fv(self.get_uniform_location('clip_rotation_matrix'), 1, GL_FALSE, self.v_matrix)
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND) #turn off alpha - as we don't sort triangles we don't want to do anything fancy here
        #glEnable(GL_CULL_FACE)
        #glCullFace(GL_BACK)
        
        glDisable(GL_POINT_SMOOTH)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)
        #glEnable(GL_BLEND)

    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(0)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_POINT_SMOOTH)
