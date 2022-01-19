#!/usr/bin/python

# ShaderProgramFactory.py
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
import logging
logger = logging.getLogger(__name__)

class ShaderProgramFactory(object):
    def __init__(self):
        pass

    _programs = {}

    @staticmethod
    def get_program(class_name, context=None, window=None):
        """

        Parameters
        ----------
        class_name is the real class of the ShaderProgram

        Returns
        -------
        object of the given class. If there's already an existing one. That one is returned.
        """
        existing_program = ShaderProgramFactory._programs.get((class_name, context))
        if existing_program:
            return existing_program
        else:
            new_program = class_name()
            if context and window:
                context.SetCurrent(window)
            ShaderProgramFactory._programs[(class_name, context)] = new_program
            logger.debug("New shader program created: {}".format(class_name))
            return new_program
