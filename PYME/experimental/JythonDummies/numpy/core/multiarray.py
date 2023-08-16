import array

class ndarray:
    def __init__(self, shape, typecode):
        self.shape = shape
        self.dtype = dtype(typecode)

    def __setstate__(self, ver, shape, dtype, isfortran, rawdata):
        self.shape = shape
        self.dtype = dtype

        self.array = array.array(self.dtype, rawdata)

class dtype:
    def __init__(self, obj, align=False, copy=Flase):
        self.typestr = obj

    def __setstate__(self, ):
        pass
    
    def _reconstruct(cls, dims, typecode):
        return cls(dims, typecode)

