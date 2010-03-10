import numpy
from multiprocessing import sharedctypes
from numpy import ctypeslib


class shmarray(numpy.ndarray):
    def __new__(cls, ctypesArray, shape, dtype=float,
          strides=None, offset=0, order=None):
        
        #some magic (copied from numpy.ctypeslib) to make sure the ctypes array
        #has the array interface
        tp = type(ctypesArray)
        try: tp.__array_interface__
        except AttributeError: ctypeslib.prep_array(tp)

        obj = numpy.ndarray.__new__(cls, shape, dtype, ctypesArray, offset, strides,
                         order)
        # set the new 'info' attribute to the value passed
        obj.ctypesArray = ctypesArray
        
        return obj

    def __array_finalize__(self, obj):
        
        if obj is None: return
        
        self.ctypesArray = getattr(obj, 'ctypesArray', None)

    def __reduce_ex__(self, protocol):
        return shmarray, (self.ctypesArray, self.shape, self.dtype, self.strides)

    def __reduce__(self):
        return __reduce_ex__(self, 0)


def zeros(shape, dtype='d'):
    '''Create a shared array initialised to zeros. Avoid object arrays, as these
    will almost certainly break'''
    shape = numpy.atleast_1d(shape)

    #we're going to use a flat ctypes array
    N = numpy.prod(shape)
    #print N

    #if the dtype's relatively simple create the corresponding ctypes array
    #otherwise create a suitably sized byte array
    print dtype, N

    if type(dtype) == numpy.dtype:
        dtc = dtype.char
    else:
        dtc = dtype

    print dtc
    if dtc in sharedctypes.typecode_to_type.keys():
        dt = dtc
    else:
        dt = 'b'
        N *= numpy.dtype(dtype).itemsize

    print dt, N

    dtype = numpy.dtype(dtype)
    a = sharedctypes.RawArray(dt, N)

    return shmarray(a, shape, dtype)

def create_copy(a):
    '''create a a shared copy of an array'''
    #create an empty array
    b = zeros(a.shape, a.dtype)

    #copy contents across
    b[:] = a[:]

    return b






