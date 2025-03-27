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
from PYME.LMVis.shader_programs.GLProgram import GLProgram
from PYME.LMVis.shader_programs.shader_program import ShaderProgram
#from PYME.LMVis.gl_texture import GaussTexture

import numpy as np
from PYME.LMVis.rendGauss import gaussKernel

import warnings

class GaussTexture:

    # specific texture id of this texture
    _texture_id = None

    def __init__(self):
        pass

    def bind_texture(self, uniform_location):
        """
        This methods binds exactly one texture.
        Parameters
        ----------
        uniform_location: int, defines the uniform location, use shader_program.get_uniform_location

        Returns
        -------

        """
        glActiveTexture(GL_TEXTURE0)
        if self._texture_id is None:
            self.load_texture()
        else:
            glBindTexture(GL_TEXTURE_2D, self._texture_id)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glUniform1i(uniform_location, 0)

    def create_texture(self, size=31, sigma=5, normalize_sum=False):
        """
        This method create a gauss kernel texture and stores it on the graphics card.
        The max amplitude of the kernel is 1.
        Parameters
        ----------
        size size of the kernel
        sigma of the kernel

        Returns
        -------
        size correction factor is a factor that should be multiplied on the texture size.
        That way sprites don't appear smaller than opaque dots
        """
        data = gaussKernel(size, sigma)
        # index of the first element that is bigger than 0.5
        index = next(x[0] for x in enumerate(data[size // 2, :]) if x[1] > 0.5)
        size_correction_factor = (float(size) / 2) / float(size // 2 - index)

        if normalize_sum:
            data /= data.sum()

        self._data = data
        return size_correction_factor

    def load_texture(self):
        size = self._data.shape[0]
        self._texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, size, size, 0, GL_RED, GL_FLOAT, np.float16(self._data))

    def delete_texture(self):
        glDeleteTextures(1, self._texture_id)

    @staticmethod
    def enable_texture_2d():
        glEnable(GL_TEXTURE_2D)

    @staticmethod
    def disable_texture_2d():
        glDisable(GL_TEXTURE_2D)



class PointSpriteShaderProgram(GLProgram):
    #    This attribute holds an instance of a texture class
    _texture = None
    #    This is the uniform location to pass to the fragment shader to locate the texture
    #_uniform_tex_2d_id = 0

    def __init__(self, **kwargs):
        GLProgram.__init__(self, vs_filename='pointsprites_vs.glsl', fs_filename='pointsprites_fs.glsl', **kwargs)
        self._texture = GaussTexture()
        self.size_factor = self._texture.create_texture()
        

    def get_size_factor(self):
        warnings.warn("use size_factor attribute instead", DeprecationWarning)
        return self.size_factor

    def __enter__(self):
        self._old_prog = glGetInteger(GL_CURRENT_PROGRAM)
        self.get_shader_program().use()
        self.set_clipping_uniforms()
        
        try:
            glEnable(GL_POINT_SPRITE)
        except:
            # deprecated for core profile
            pass
        glEnable(GL_PROGRAM_POINT_SIZE)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        #glBlendEquation(GL_FUNC_ADD)
        #self._texture.enable_texture_2d()
        self._texture.bind_texture(self._shader_program.get_uniform_location(b'tex2D'))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(self._old_prog)
        glDisable(GL_BLEND)
        glDisable(GL_PROGRAM_POINT_SIZE)
        try:
            glDisable(GL_POINT_SPRITE)
        except:
            # deprecated for core profile
            pass
        glEnable(GL_DEPTH_TEST)

    def __del__(self):
        # self._texture.delete_texture()
        pass

class BigPointSpriteShaderProgram(PointSpriteShaderProgram):
    def __init__(self, **kwargs):
        GLProgram.__init__(self, vs_filename='default_vs.glsl', fs_filename='bigpointsprites_fs.glsl', gs_filename='bigpoints_gs.glsl', **kwargs)
        self._texture = GaussTexture()
        self.size_factor = self._texture.create_texture()
