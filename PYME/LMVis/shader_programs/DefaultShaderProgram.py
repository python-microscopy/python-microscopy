#!/usr/bin/python

# DefaultShaderProgram.py
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
import numpy as np

from PYME.LMVis.shader_programs.GLProgram import GLProgram
from OpenGL.GL import  GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, glUseProgram, \
    glPolygonMode, GL_FILL, GL_FRONT_AND_BACK, glEnable, GL_BLEND, GL_SRC_ALPHA, GL_DST_ALPHA, glBlendFunc, \
    glBlendEquation, GL_FUNC_ADD, GL_DEPTH_TEST, glDepthFunc, GL_LEQUAL, GL_POINT_SMOOTH, GL_ONE_MINUS_SRC_ALPHA, GL_PROGRAM_POINT_SIZE,\
    GL_TRUE, glDepthMask, glClearDepth, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, glDisable, GL_ONE, GL_ZERO, \
    glUniform4f, glUniform1f, glUniformMatrix4fv, GL_CURRENT_PROGRAM, glGetInteger, GL_POINT_SPRITE

from PYME.LMVis.shader_programs.shader_program import ShaderProgram


class DefaultShaderProgram(GLProgram):
    def __init__(self, **kwargs):
        GLProgram.__init__(self, vs_filename = 'default_vs.glsl', fs_filename='default_fs.glsl', **kwargs)  

    def __enter__(self):
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glDisable(GL_DEPTH_TEST)
        #glDepthFunc(GL_LEQUAL)
        try:
            glEnable(GL_POINT_SMOOTH)
        except:
            # not supported in core profile
            pass
        
        self._old_prog = glGetInteger(GL_CURRENT_PROGRAM)
        
        self.get_shader_program().use()
        self.set_clipping_uniforms()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(self._old_prog)
        glDisable(GL_BLEND)
        #glClearDepth(1.0)
        #glClear(GL_DEPTH_BUFFER_BIT)
        pass


class OpaquePointShaderProgram(DefaultShaderProgram):
    def __init__(self, **kwargs):
        GLProgram.__init__(self, vs_filename='default_vs.glsl', fs_filename='flatpoints_fs.glsl', **kwargs)

    def __enter__(self):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)


        glEnable(GL_PROGRAM_POINT_SIZE)
        
        #glDepthFunc(GL_LEQUAL)
        try:
            glEnable(GL_POINT_SMOOTH)
            glEnable(GL_POINT_SPRITE)
        except:
            # not supported in core profile
            pass
        self.get_shader_program().use()
        self.set_clipping_uniforms()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        return self
    
class TransparentPointShaderProgram(OpaquePointShaderProgram):
    def __enter__(self):
        super().__enter__()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        return self
    

class BigOpaquePointShaderProgram(DefaultShaderProgram):
    def __init__(self, **kwargs):
        GLProgram.__init__(self, vs_filename='default_vs.glsl', fs_filename='bigflatpoints_fs.glsl', gs_filename='bigpoints_gs.glsl', **kwargs)

    def __enter__(self):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

        try:
            glEnable(GL_POINT_SPRITE)
        except:
            # not supported in core profile
            pass

        self.get_shader_program().use()
        self.set_clipping_uniforms()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        return self
    
class BigTransparentPointShaderProgram(BigOpaquePointShaderProgram):
    def __enter__(self):
        super().__enter__()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        return self

class ImageShaderProgram(DefaultShaderProgram):
    def __init__(self, **kwargs):
        GLProgram.__init__(self, vs_filename='image_vs.glsl', fs_filename='image_fs.glsl', **kwargs)
        
class TextShaderProgram(DefaultShaderProgram):
    def __init__(self, **kwargs):
        GLProgram.__init__(self, vs_filename='text_vs.glsl', fs_filename='text_fs.glsl', **kwargs)

class FatLineShaderProgram(DefaultShaderProgram):
    def __init__(self, **kwargs):
        GLProgram.__init__(self, vs_filename='widelines_vs.glsl', fs_filename='widelines_fs.glsl', gs_filename='widelines_gs.glsl', **kwargs)

    def __enter__(self):
        self.get_shader_program().use()
        self.set_clipping_uniforms()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        return self
