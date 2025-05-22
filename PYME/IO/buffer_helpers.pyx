import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int64_t DTYPE_t

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def update_indices_add(unsigned short [:,:,::1] frameBuffer not None, unsigned short [:,:,::1] indices not None, unsigned short [:,::1] data, int slot):
    cdef int idx, x, y, sl, j
    cdef unsigned short gt
    
    cdef int sx = data.shape[0]
    cdef int sy = data.shape[1]
    cdef int n_slots = frameBuffer.shape[0]
    
    cdef unsigned short d_xy = 0
    cdef unsigned short cur_slot = 0
    
    cdef unsigned short * pData = &data[0,0]
    cdef unsigned short * pFrameBuffer = &frameBuffer[0,0,0]
    cdef unsigned short * pIDX = &indices[0,0,0]
    cdef unsigned short * pIDX_slot = &indices[slot, 0,0]
    
    cdef int npx = sx*sy
    
    for sl in range(n_slots):
        pIDX = &indices[sl,0,0]
        pFrameBuffer = &frameBuffer[sl, 0, 0]
        
        for j in range(npx):
            #following loop does addition and updating in one
            #Whilst finding where to insert can be O(log(n)), complexity is limited by insertion itself which is always O(n)
            #we can do both within the one loop, meaning that we get the position finding more or less for free
            #idx = 0 #index (order) where this pixel will be in buffer
            d_xy = pData[j]

            gt = (pFrameBuffer[j] < d_xy)
            pIDX_slot[j] += gt
            pIDX[j] += (1-gt)
            
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def update_indices_add_pc(unsigned short [:,:,::1] frameBuffer not None, unsigned short [:,:,::1] indices not None,
                          unsigned short [:,::1] data, int slot, unsigned short [:,::1] pct not None, unsigned short percentile):
    cdef int idx, x, y, sl, j
    cdef unsigned short gt, tmp
    
    cdef int sx = data.shape[0]
    cdef int sy = data.shape[1]
    cdef int n_slots = frameBuffer.shape[0]
    
    cdef unsigned short d_xy = 0
    cdef unsigned short cur_slot = 0
    
    cdef unsigned short * pData = &data[0,0]
    cdef unsigned short * pFrameBuffer = &frameBuffer[0,0,0]
    cdef unsigned short * pIDX = &indices[0,0,0]
    cdef unsigned short * pIDX_slot = &indices[slot, 0,0]
    cdef unsigned short * pPCT = &pct[0,0]
    
    cdef int npx = sx*sy
    
    for sl in range(n_slots):
        pIDX = &indices[sl,0,0]
        pFrameBuffer = &frameBuffer[sl, 0, 0]
        
        for j in range(npx):
            #following loop does addition and updating in one
            #Whilst finding where to insert can be O(log(n)), complexity is limited by insertion itself which is always O(n)
            #we can do both within the one loop, meaning that we get the position finding more or less for free
            #idx = 0 #index (order) where this pixel will be in buffer
            d_xy = pData[j]

            gt = (pFrameBuffer[j] < d_xy)
            pIDX_slot[j] += gt
            pIDX[j] += (1-gt)
            
            #tmp = (pIDX[j] == percentile)
            #pPCT[j] = tmp*sl + (1-tmp)*pPCT[j] #pFrameBuffer[j]

#constants
cdef enum:
    MAXSHORT = 65535
    MAXIDX = 10000

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def update_indices_remove(unsigned short [:,:,::1] frameBuffer not None, unsigned short [:,:,::1] indices not None, int slot):
    cdef int idx, x, y, sl, gt
    
    cdef int n_slots = frameBuffer.shape[0]
    cdef int sx = frameBuffer.shape[1]
    cdef int sy = frameBuffer.shape[2]
    
    cdef unsigned short * pFrameBuffer = &frameBuffer[slot,0,0]
    cdef unsigned short * pIDX = &indices[0,0,0]
    cdef unsigned short * pIDX_slot = &indices[slot, 0,0]
    
    cdef int npx = sx*sy
    
    for sl in range(n_slots):
        pIDX = &indices[sl,0,0]
        
        for j in range(npx):
            #following loop does addition and updating in one
            #Whilst finding where to insert can be O(log(n)), complexity is limited by insertion itself which is always O(n)
            #we can do both within the one loop, meaning that we get the position finding more or less for free
            #idx = 0 #index (order) where this pixel will be in buffer

            pIDX[j] -= (pIDX[j] > pIDX_slot[j])
            
    for j in range(npx):
        pIDX_slot[j] = MAXIDX
        pFrameBuffer[j] = MAXSHORT
    

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def update_indices_remove_pc(unsigned short [:,:,::1] frameBuffer not None, unsigned short [:,:,::1] indices not None,
                             int slot, unsigned short [:,::1] pct not None, unsigned short percentile):
    cdef int idx, x, y, sl, gt
    
    cdef int n_slots = frameBuffer.shape[0]
    cdef int sx = frameBuffer.shape[1]
    cdef int sy = frameBuffer.shape[2]
    
    cdef unsigned short * pFrameBuffer = &frameBuffer[slot,0,0]
    cdef unsigned short * pIDX = &indices[0,0,0]
    cdef unsigned short * pIDX_slot = &indices[slot, 0,0]
    cdef unsigned short * pPCT = &pct[0,0]
    cdef unsigned short * pFrameSl = &frameBuffer[0,0,0]
    
    cdef int npx = sx*sy
    
    for sl in range(n_slots):
        pIDX = &indices[sl,0,0]
        pFrameSl = &frameBuffer[sl, 0, 0]
        
        for j in range(npx):
            #following loop does addition and updating in one
            #Whilst finding where to insert can be O(log(n)), complexity is limited by insertion itself which is always O(n)
            #we can do both within the one loop, meaning that we get the position finding more or less for free
            #idx = 0 #index (order) where this pixel will be in buffer

            pIDX[j] -= (pIDX[j] > pIDX_slot[j])
            
            if (pIDX[j] == percentile):
                pPCT[j] = pFrameBuffer[j]
            
    for j in range(npx):
        pIDX_slot[j] = MAXIDX
        pFrameBuffer[j] = MAXSHORT
        

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def get_pct(unsigned short [:,:,::1] frameBuffer not None, unsigned short [:,:,::1] indices not None,
            unsigned short pct_idx, unsigned short [:,::1] pctBuffer not None,):
    cdef int idx, x, y, sl, gt
    
    cdef int n_slots = frameBuffer.shape[0]
    cdef int sx = frameBuffer.shape[1]
    cdef int sy = frameBuffer.shape[2]
    
    
    cdef unsigned short * pIDX = &indices[0,0,0]
    cdef unsigned short * pPCT = &pctBuffer[0,0]
    cdef unsigned short * pFrameSl = &frameBuffer[0,0,0]
    
    cdef int npx = sx*sy
    
    for sl in range(n_slots):
        pIDX = &indices[sl,0,0]
        pFrameSl = &frameBuffer[sl, 0, 0]
        
        for j in range(npx):
            #if pIDX[j] == pct_idx:
            pPCT[j] += pFrameSl[j]*(pIDX[j] == pct_idx)
    