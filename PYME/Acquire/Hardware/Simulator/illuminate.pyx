#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy as np
cimport numpy as np
#from libc.stdlib cimport random, RAND_MAX
cdef extern from "stdlib.h":
    long int rand()
    long int RAND_MAX

#ctypedef np.int_t INT_t
#ctypedef np.float_t FLOAT_t

cdef float IRMAX = 1.0/RAND_MAX

def illuminate(np.ndarray[float, ndim=3] transTensor,
               fl, 
               np.ndarray[int, ndim=1] state, 
               np.ndarray[float, ndim=2] abscosthetas, 
               np.ndarray[float, ndim=1] dose, 
               ilFrac, 
               int activeState):
    
    cdef np.ndarray[float, ndim=1] c0 = abscosthetas[:,0]*dose[1]*ilFrac
    cdef np.ndarray[float, ndim=1] c1 = abscosthetas[:,1]*dose[2]*ilFrac
    #grab transition matrix
    #for i in range(fl.shape[0]):
    #    fli = fl[i]
    
    cdef np.ndarray[float, ndim=2] exc = fl['exc']
    
    cdef int nStates = transTensor.shape[0]
    
    cdef np.ndarray[float, ndim=1] transVec = np.zeros([nStates], 'f')

    cdef np.ndarray[float, ndim=1] Iout = np.zeros([c0.shape[0]], 'f')
    
    cdef float tvs
    cdef float transCs
    cdef int i
    cdef int j
    cdef int st_i
    cdef float r
    cdef float dose0 = dose[0]
    cdef float c0i, c1i, tvj
    
    
    #print transTensor.shape[0], transTensor.shape[1], transTensor.shape[2]

    for i in range(fl.shape[0]):
        st_i = state[i]
        c0i = c0[i]
        c1i = c1[i]
        
        tvs = 0
        
        for j in range(nStates):
            tvj = dose0*transTensor[st_i, j,0]
            tvj += c0i*transTensor[st_i, j,1] #vstack((c0,c0, c0,c0)).T 
            tvj += c1i*transTensor[st_i, j,2]
            
            tvs += tvj
            transVec[j] = tvj
                    
        transVec[st_i] = 1 - tvs
        
        #r = float(np.random.rand())
        r = IRMAX*float(rand())
        
        transCs = 0
        j = 0
        transCs += transVec[j]
        while transCs < r and j < (nStates - 1):
            j += 1
            transCs += transVec[j]
            
            
        
        state[i] = j
        
        if (j == activeState):
            Iout[i] = (exc[i, 0]*c0i + exc[i,1]*c1i)
    
    #return (fl['state'] == activeState)*(fl['exc'][:,0]*c0 + fl['exc'][:,1]*c1)
    return Iout