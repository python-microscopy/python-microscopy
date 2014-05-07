#!/usr/bin/python
##################
# test_lut.py
#
# Copyright David Baddeley, 2011
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

from pylab import *

import lut

lut1 = (255*cm.gray(linspace(0,1,256))[:,:3].T).astype('uint8').copy()
print((lut1.shape))

def testLut():
    d = (100*rand(5,5)).astype('uint16')
    o = np.zeros((5,5,3), 'uint8')
    
    lut.applyLUTu8(d.astype('uint8'), .01, 0,lut1, o)
    lut.applyLUTu16(d, .01, 0,lut1, o)
    lut.applyLUTf(d.astype('uint16'), .01, 0,lut1, o)
    
    print(o)
    print((lut1.T[((d + 0)*.01*256).astype('i')]))

