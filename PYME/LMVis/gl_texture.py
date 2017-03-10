#!/usr/bin/python

# gl_texture.py
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
from OpenGL.GL import *
import numpy as np
from PYME.LMVis.rendGauss import gaussKernel


class GaussTexture:

    # specific texture id of this texture
    _texture_id = 0

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
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glUniform1i(uniform_location, 0)

    def load_texture(self, size=31, sigma=5):
        data = gaussKernel(size, sigma)
        glGenTextures(1, self._texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, size, size, 0, GL_LUMINANCE, GL_FLOAT, np.float16(data))

    def delete_texture(self):
        glDeleteTextures(1, self._texture_id)

    @staticmethod
    def enable_texture_2d():
        glEnable(GL_TEXTURE_2D)

    @staticmethod
    def disable_texture_2d():
        glDisable(GL_TEXTURE_2D)
