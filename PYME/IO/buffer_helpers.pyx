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
ctypedef np.int_t DTYPE_t

def update_indices_add(np.ndarray[np.uint16_t, ndim=3]frameBuffer, np.ndarray[np.int_t, ndim=3]indices, np.ndarray[np.float_t, ndim=2]data, int slot):
    cdef int idx, x, y, sl, gt
    
    cdef int sx = data.shape[0]
    cdef int sy = data.shape[1]
    cdef int n_slots = frameBuffer.shape[0]
    
    for x in range(sx):
        for y in range(sy):

            #following loop does addition and updating in one
            #Whilst finding where to insert can be O(log(n)), complexity is limited by insertion itself which is always O(n)
            #we can do both within the one loop, meaning that we get the position finding more or less for free
            idx = 0 #index (order) where this pixel will be in buffer
            for sl in range(n_slots):
                if not sl == slot: #ignore data we are adding
                    #alternatively (might potentially be faster as it avoids branches - wouldn't know without trying):
                    gt = (frameBuffer[sl, x,y] < data[x,y])
                    idx += gt
                    indices[sl, x, y] += (1-gt)

            indices[slot] = idx