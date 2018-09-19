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

from .lut import *

def applyLUT(seg, gain, offset, lut, ima):
    if seg.dtype == 'uint8':
        applyLUTu8(seg, gain, offset, lut, ima)
    elif seg.dtype == 'uint16':
        #print lut.strides
        applyLUTu16(seg, gain, offset, lut, ima)
    else:
        applyLUTf(seg.astype('f'), gain, offset, lut, ima)
