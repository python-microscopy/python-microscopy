#!/usr/bin/python

# layers.py
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
import warnings

from PYME.LMVis.shader_programs.ShaderProgramFactory import ShaderProgramFactory

from PYME.recipes.traits import HasTraits

class BaseLayer(HasTraits):
    """
    This class represents a layer that should be rendered.
    It also deals with the shader program that is used to render the layer
     with appropriate shaders.
    """
    def __init__(self):
        self._shader_program = None

    
    def set_shader_program(self, shader_program):
        self._shader_program = ShaderProgramFactory.get_program(shader_program)

    @property
    def shader_program(self):
        return self._shader_program
    
    def get_shader_program(self):
        warnings.warn("use the shader_program property instead", DeprecationWarning)
        return self.shader_program

    @abc.abstractmethod
    def render(self, gl_canvas):
        pass
