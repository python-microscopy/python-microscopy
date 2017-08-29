#!/usr/bin/python

###############
# __init__.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
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
################
"""
This package contains the image viewer.

The bulk of the viewer code lives in :py:mod:`PYME.DSView.dsviewer`, with the other modules providing support. A lot of
the functionality is provided through the plugin modules in the :py:mod:`PYME.DSView.modules` package.

The viewer itself is launched using the `dh5view` script.
"""
from .dsviewer import View3D, ViewIm3D
from PYME.IO.image import ImageStack
