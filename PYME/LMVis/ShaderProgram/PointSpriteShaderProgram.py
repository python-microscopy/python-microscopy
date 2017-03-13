#!/usr/bin/python

# PointSpriteShaderProgram.py
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

from OpenGL.GL import *
from PYME.LMVis.ShaderProgram.GLProgram import GLProgram
from PYME.LMVis.ShaderProgram.ShaderProgram import ShaderProgram
from PYME.LMVis.gl_texture import GaussTexture


class PointSpriteShaderProgram(GLProgram):
    #    This attribute holds an instance of a texture class
    _texture = None
    #    This is the uniform location to pass to the fragment shader to locate the texture
    _uniform_tex_2d_id = 0

    def __init__(self):
        GLProgram.__init__(self)
        shader_path = os.path.join(os.path.dirname(__file__), "../shaders/")
        shader_program = ShaderProgram(shader_path)
        shader_program.add_shader("pointsprites_vs.glsl", GL_VERTEX_SHADER)
        shader_program.add_shader("pointsprites_fs.glsl", GL_FRAGMENT_SHADER)
        shader_program.link()
        self._texture = GaussTexture()
        self._texture.load_texture()
        self._uniform_tex_2d_id = glGetUniformLocation(shader_program.get_program(), b'tex2D')
        self.set_shader_program(shader_program)

    def __enter__(self):
        self.get_shader_program().use()
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        self._texture.enable_texture_2d()
        self._texture.bind_texture(self._shader_program.get_uniform_location(b'tex2D'))

    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(0)
        glDisable(GL_BLEND)
        glDisable(GL_PROGRAM_POINT_SIZE)
        glDisable(GL_POINT_SPRITE)
        glEnable(GL_DEPTH_TEST)

    def __del__(self):
        self._texture.delete_texture()
