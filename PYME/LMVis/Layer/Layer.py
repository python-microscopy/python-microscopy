#!/usr/bin/python

# Layer.py
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


class Layer(object):

    _is_initialized = False

    _shader_program = None

    def __init__(self):
        pass

    def is_initialized(self):
        return self._is_initialized

    def set_shader_program(self, shader_program):
        self._shader_program = shader_program

    def get_shader_program(self):
        return self._shader_program

    def initialize(self):
        self._is_initialized = True

    @abc.abstractmethod
    def render(self, gl_canvas):
        pass
