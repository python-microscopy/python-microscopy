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

from PYME.LMVis.ShaderProgram.GLProgram import GLProgram, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, \
     glUseProgram, glUniform4f, glUniform1f
from PYME.LMVis.ShaderProgram.ShaderProgram import ShaderProgram, os


class GouraudShaderProgram(GLProgram):
    INPUT_LIGHT = b'inputLight'
    test_value = [1.0, 0.0, 1.0, 1.0]
    lights_vector = (2.0, 2.00, 2.0, 0.0)
    view_vector = (0.0, 0.0, -1.0, 0.0)

    LIGHT_PROPS = {
            'light_ambient': (0.5, 0.5, 0.5, 1.0),
            'light_diffuse': (0.8, 0.8, 0.8, 1.0),
            'light_specular': (0.3, 0.3, 0.3, 1.0),
            'light_position': lights_vector
    }

    def __init__(self):
        super(GouraudShaderProgram, self).__init__()
        GLProgram.__init__(self)
        shader_path = os.path.join(os.path.dirname(__file__), "../shaders/")
        shader_program = ShaderProgram(shader_path)
        shader_program.add_shader("gouraud_vs.glsl", GL_VERTEX_SHADER)
        shader_program.add_shader("gouraud_fs.glsl", GL_FRAGMENT_SHADER)
        shader_program.link()
        self.set_shader_program(shader_program)
        self._shininess = 80

    def __enter__(self):
        self.get_shader_program().use()
        for name, value in GouraudShaderProgram.LIGHT_PROPS.iteritems():
            location = self.get_uniform_location(name)
            glUniform4f(location, *value)
        location = self.get_uniform_location('shininess')
        glUniform1f(location, self._shininess)
        location = self.get_uniform_location('view_vector')
        glUniform4f(location, *self.view_vector)

    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(0)
