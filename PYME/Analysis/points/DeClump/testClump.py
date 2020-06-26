#!/usr/bin/python

###############
# testClump.py
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

if __name__ == '__main__':
    # from pylab import *
    import numpy as np
    
    
    from PYME.Analysis.points.DeClump import deClump
    
    
    t = np.arange(0, 200, .02)
    print((len(t)))
    x = np.random.randn(10000)
    y = np.random.randn(10000)
    delta_x = .05*np.ones(x.shape)
    
    asg = deClump.findClumps(t.astype('i'), x.astype('f4'), y.astype('f4'), delta_x.astype('f4'), 2)
    
    print(asg)
