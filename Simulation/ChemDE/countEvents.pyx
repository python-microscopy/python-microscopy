#!/usr/bin/python
##################
# countStates.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy as np
cimport numpy as np

def countEvents(np.ndarray[np.int32_t, ndim=1] trace, int threshold):
    cdef int nEvents, lastObs
    cdef Py_ssize_t i

    nEvents = 0
    lastObs = -100000

    for i in range(trace.shape[0]):
        if trace[i]:
            if (i- lastObs) >= threshold:
                nEvents += 1
            lastObs = i
    return nEvents


