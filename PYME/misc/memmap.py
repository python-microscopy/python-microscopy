#!/usr/bin/python

###############
# memmap.py
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


#from guppy import hpy
#hp = hpy()

import numpy as np
import sys

def memmap(h, Astep = 1024,Amax = 1024**3):
    mmap = np.zeros(Amax/Astep, 'uint8')
    
    for x in h.nodes:
        #address
        a = id(x)/Astep 
        s = sys.getsizeof(x)/Astep
        
        mmap[a:(a+s+1)] |= 1
        
        #special case for numpy arrays, as sizeof  doesn't work
        #get the underlying data adress and size and add that too
        if isinstance(x, np.ndarray):
            a = x.ctypes.data/Astep
            s = x.nbytes/Astep
            
            mmap[a:(a+s+1)] |= 2
            
        
    return mmap
        
    