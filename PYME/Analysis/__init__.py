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
This module contains (mostly) general purpose image analysis functions. For the most part, these operate on pixelated
image data (i.e. conventional images), with the exception of :py:mod:`PYME.Analysis.points` which contains functions
for the post-processing of point data sets, and :py:mod:`PYME.Analysis.Tracking` which performs tracking and point
linking on previously localized data.

Code for extracting single molecule positions is located in :py:mod:`PYME.localization`
"""