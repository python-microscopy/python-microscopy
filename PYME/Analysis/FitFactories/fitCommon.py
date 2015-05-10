# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 17:58:51 2014

@author: david
"""

try:
    import copy_reg
except ImportError:
    import copyreg as copy_reg

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)

def replNoneWith1(n):
        if n == None:
            return 1
        else:
            return n

def fmtSlicesUsed(slicesUsed):
    if slicesUsed == None:
        return ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
    else:
        return tuple([(sl.start, sl.stop, replNoneWith1(sl.step)) for sl in slicesUsed] )