# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:39:57 2012

@author: david
"""

from guppy import hpy
hp = hpy()

import numpy as np
import sys

def memmap(h, Astep = 1024,Amax = 1024**3):
    mmap = np.zeros(Amax/Astep, 'uint8')
    
    for x in h.nodes:
        #address
        a = id(x)/Astep 
        s = sys.getsizeof(x)/Astep
        
        mmap[a:(a+s+1)] = 1
        
        #special case for numpy arrays, as sizeof  doesn't work
        #get the underlying data adress and size and add that too
        if isinstance(x, np.ndarray):
            a = x.ctypes.data/Astep
            s = x.nbytes/Astep
            
            mmap[a:(a+s+1)] = 1
            
        
    return mmap
        
    