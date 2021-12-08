#!/usr/bin/python

# WireFrameShaderProgram.py
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

from PYME.LMVis.shader_programs.GLProgram import GLProgram, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, glPolygonMode, \
    GL_FRONT_AND_BACK, GL_LINE, GL_FILL, glUseProgram, GL_BLEND,glDisable, glEnable, glDepthMask, GL_TRUE, GL_FALSE, \
    GL_DEPTH_TEST, glBlendFunc, GL_SRC_ALPHA, GL_ONE, glGetInteger, GL_CURRENT_PROGRAM
from PYME.LMVis.shader_programs.shader_program import ShaderProgram


class TesselShaderProgram(GLProgram):
    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(self._old_prog)
        pass

    def __enter__(self):
        self._old_prog = glGetInteger(GL_CURRENT_PROGRAM)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glDisable(GL_DEPTH_TEST)
        #glDepthFunc(GL_LEQUAL)
        #glEnable(GL_POINT_SMOOTH)
        self.get_shader_program().use()
        self.set_clipping_uniforms()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        return self

    def __init__(self):
        GLProgram.__init__(self)
        shader_path = os.path.join(os.path.dirname(__file__), "shaders")
        _shader_program = ShaderProgram(shader_path)
        _shader_program.add_shader("tessel_vs.glsl", GL_VERTEX_SHADER)
        _shader_program.add_shader("tessel_fs.glsl", GL_FRAGMENT_SHADER)
        _shader_program.link()
        self.set_shader_program(_shader_program)
