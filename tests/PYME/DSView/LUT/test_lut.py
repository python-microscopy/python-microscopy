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
import pytest

# mark some tests as expected to fail if we are testing on a headless system
try:
    import wx
    HAVE_WX = True
except ImportError:
    HAVE_WX = False

@pytest.mark.xfail(not HAVE_WX, reason="Fails on a headless system as PYME.DSView.__init__ imports wx")
def test():
    import numpy as np
    from matplotlib import cm
    #from pylab import cm, rand
    
    from PYME.DSView.LUT import lut
    
    lut1 = (255*cm.gray(np.linspace(0,1,256))[:,:3].T).astype('uint8').copy()
    print((lut1.shape))
    
    def testLut():
        d = (100*np.random.rand(5,5)).astype('uint16')
        o = np.zeros((5,5,3), 'uint8')
        
        lut.applyLUTu8(d.astype('uint8'), .01, 0,lut1, o)
        lut.applyLUTu16(d, .01, 0,lut1, o)
        lut.applyLUTf(d.astype('uint16'), .01, 0,lut1, o)
        
        print(o)
        print((lut1.T[((d + 0)*.01*256).astype('i')]))
        
    testLut()

if __name__ == '__main__':
    test()