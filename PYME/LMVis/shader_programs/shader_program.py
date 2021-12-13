#!/usr/bin/python

##################
# ShaderProgram.py
#
# Copyright Michael Graff
# graff@hm.edu
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
##################

from OpenGL.GL import *

from PYME.LMVis.shader_programs.shader_loader import *


class ShaderProgram(object):
    """
    This class handles all  OpenGl-PROGRAM related stuff

    This includes loading and adding shaders, compiling them
    and handling or raising exceptions.

    """

#   that's the actual OpenGL program handle
    _program = 0
#   that's the path where the shader files are saved
    _shader_path = None

    _shaders = set()

    def __init__(self, shader_path="./shaders/"):
        """

        :param shader_path: the folder that contains the shader sources
                            ending with a '/'
                            It can be relative or absolute.
        :raises GLProgramException if glUseProgram is not successful
        """
        if not glUseProgram:
            raise GLProgramException('Missing Shader Objects!')
        self._shader_path = shader_path
        self._program = glCreateProgram()
        
        if self._program == 0:
            raise RuntimeError('glCreateProgram failed')

    def add_shader(self, shader_name, shader_type):
        """
        This method adds a new shader module to the program.

        :param shader_name: the filename of the shader that should be loaded
                            keep in mind that the shader_path was set in __init__
        :param shader_type: defines the stage the shader should be added
                            e.g. GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
        :return: nothing

        :raises: GLShaderLoadException if the shader could not be loaded
        :raises: GLProgramException if the shader could not be compiled

        """
        shader = glCreateShader(shader_type)
        code = read_shader(shader_name, self._shader_path)
        glShaderSource(shader, code)
        glCompileShader(shader)
        is_compiled = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if is_compiled != 1:
            error_log = glGetShaderInfoLog(shader)
            exception = GLProgramException("Shader " + shader_name + " didn't compile correctly", error_log)
            glDeleteShader(shader)
            raise exception
        glAttachShader(self._program, shader)
        self._shaders.add(shader)

    # def get_program(self):
    #     """
    #     Returns the real handle
    #     :return: the real handle to the OpenGL-program used by the OpenGL-Context
    #     """
    #     return self._program

    def link(self):
        glLinkProgram(self._program)

    def get_uniform_location(self, uniform_name):
        return glGetUniformLocation(self._program, uniform_name)

    def use(self):
        try:
            glUseProgram(self._program)
        except GLError:
            exception = GLProgramException(glGetProgramInfoLog(self._program))
            raise exception

    def __del__(self):
        # shaders seem to have been deleted already, since it throws errors
        # for shader in self._shaders:
        #     glDeleteShader(shader)
        # glDeleteProgram(self._program)
        pass


def read_shader(shader_name, path):
    """
    This method reads a given shader within a given file path

    :param shader_name: the name of the shader file
    :param path: the path where the shader is stored on disk
    :return: a string representing the shader
    :raises: GLShaderLoadException if the shader could not be loaded
    """
    return ShaderLoader().read_file_with_path(path, shader_name)


class GLProgramException(IOError):
    """
    This class is used if an error in the OpenGLProgramHandler occurs
    You usually only pass the text message to the constructor and raise the exception
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
