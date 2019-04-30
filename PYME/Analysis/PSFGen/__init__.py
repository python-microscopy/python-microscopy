#!/usr/bin/python

##################
# __init__.py
#
# Copyright David Baddeley, 2009
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
##################
"""
This module contains various routines for generating simulated PSFs.

These include:

The Gibson and Lanni model for imaging through stratified media (ps_app) and models based on Fourier propagation. For the
later, (:py:mod:`PYME.Analysis.PSFGen.fourier`) generates far field paraxial / low NA PSFs, whereas
(:py:mod:`PYME.Analysis.PSFGen.fourierHNA`) uses a modified Fourier propagation scheme which better reflects a high NA
and/or vectorial imaging scenario.

Functions are available for simulating the most common PSFs types used in localization microscopy, along with PSFs having
arbitrary pupils and / or aberrations expressed as in terms of Zernike modes.

(:py:mod:`PYME.Analysis.PSFGen.fourierHNA`) also has a basic implementation of the Gerchberg-Saxton algorithm for pupil
phase extraction.
"""

from .ps_app import *
