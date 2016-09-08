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
This package contains deconvolution code. It is currently fairly disorganized, and has a fair bit of deprecated and
and orphaned code. The interesting (and moderately up to date) modules are:

- :py:mod:`PYME.Deconv.dec` : ICTM deconvolution
- :py:mod:`PYME.Deconv.richardsonLucy` : Richardson-Lucy deconvolution
- :py:mod:`PYME.Deconv.deconvDialogs` : GUI elements for configuring deconvolution.

Also of interest are the deconvolution modules in :py:mod:`PYME.DSView.modules.deconvolution` and
:py:class:`PYME.recipes.processing.Deconvolve` which show how the code here is called and interfaced.
"""
