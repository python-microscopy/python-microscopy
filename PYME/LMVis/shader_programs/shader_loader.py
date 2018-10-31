#!/usr/bin/python

##################
# gl_shaderLoader.py
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

import os


class ShaderLoader(object):
    """
    This class is used to load shader source code and deliver it to programs that need it.


    It could be used to read any text based file. But this would be confusing and should't be done.
    """

    # The source code of the most frequent shader that has been read using this instance.
    # Don't access it directly, but use the given getter
    _code = 0

    def __init__(self):
        pass

    def read_file(self, path):
        """
        Read a shader source file.
        :param path: the directory and the filename as the shader is saved on the disk
                     it can be relative or absolute
        :return: a string representing the source code of the shader
        """
        try:
            with open(path, 'r') as f:
                self._code = f.read()
            return self._code
        except IOError as e:
            message = "File {} could not be read properly in folder: {}".format(path, os.getcwd())
            raise GLShaderLoadException(e.errno, e.strerror, message)

    def read_file_with_path(self, path, file_name):
        """
        Read a shader source file.
        :param path: the directory on the disk, including the last '/'
                     it can be relative or absolute
        :param file_name:  the filename of the shader source
        :return: a string representing the source code of the shader
        """
        return self.read_file(os.path.join(path, file_name))

    def get_code(self):
        return self._code

    def __del__(self):
        self._code = None


class GLShaderLoadException(Exception):
    """
    This class is used if an error in the ShaderLoader occurs
    You usually only pass the text message to the constructor and raise the exception
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

if __name__ == '__main__':
    s = ShaderLoader()
    print(s.read_file_with_path("./shaders/", "pointsprites_fs.glsl"))
