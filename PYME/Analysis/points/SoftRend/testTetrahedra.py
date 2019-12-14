#!/usr/bin/python

###############
# testTetrahedra.py
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

def run_t():
    #from pylab import *
    import numpy as np
    from PYME.DSView.dsviewer import View3D
    from PYME.Analysis.points import gen3DTriangs
    
    x = 5e3*np.random.rand(1000)
    y = 2.5e3*np.random.rand(1000)
    z = 5e3*np.random.rand(1000)
    
    
    im = np.zeros((250, 150, 25), order='F')
    
    gen3DTriangs.renderTetrahedra(im, x, y, z, scale=[1, 1, 1], pixelsize=[10, 10, 100])
    View3D(im)
    
if __name__ == '__main__':
    run_t()
