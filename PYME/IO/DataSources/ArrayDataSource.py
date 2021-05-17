import numpy as np
from .BaseDataSource import BaseDataSource, DefaultList
import tables

_array_types = (np.ndarray, tables.EArray)

# we might not have dask or zarr installed, support them if present
try:
    import dask.array as da
    _array_types = _array_types + (da.Array,)
except ImportError:
    pass

try:
    import zarr
    _array_types = _array_types + (zarr.Array,)
except ImportError:
    pass

def atleast_nd(a, n):
    while a.ndim < n:
        a = np.expand_dims(a, a.ndim)
    
    return a

class ArrayDataSource(BaseDataSource): #permit indexing with more dimensions larger than len(shape)
    def __init__(self, data, dimOrder='XYTC', dim_1_is_z=False):
        self.data = data
        self.type = 'Array'
        
        self.dim_1_is_z = dim_1_is_z
        
        if not isinstance(data, _array_types): # is a data source
            # TODO - duck type instead?
            raise DeprecationWarning('Expecting array data')
        
        #self.additionalDims = dimOrder[2:len(data.shape)]
        
        shape = list(self.data.shape)
        if self.dim_1_is_z:
            shape[:2] = self.data.shape[1:3]
            shape[2] = self.data.shape[0]
        
        self._shape = DefaultList(shape)
        self._ndim = self.data.ndim
        
        #if len(self.data.shape) > 3:
        #    self.sizeC = self.data.shape[3]
        
        self.oldData = None
        self.oldSlice = None #buffer last lookup
    
    @property
    def ndim(self):
        return self._ndim
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __getattr__(self, name):
        return getattr(self.data, name)
    
    def __getitem__(self, keys):
        keys = list(keys)
        #print keys
        #for i in range(len(keys)):
        #    if not isinstance(keys[i], slice):
        #        keys[i] = slice(int(keys[i]), int(keys[i]) + 1)
        #if keys == self.oldSlice:
        #    return self.oldData
        
        self.oldSlice = keys
        
        if len(keys) > len(self.data.shape):
            keys = keys[:len(self.data.shape)]
        
        if self.dim_1_is_z:
            keys = [keys[2]] + keys[:2] + keys[3:]
        
        #print keys
        
        #if self.type == 'Array':
        r = self.data.__getitem__(tuple(keys))
        
        if not isinstance(r, (np.ndarray, np.number)):
            # make slicing work for dask arrays TODO - revisit??
            r = r.compute()
        
        self.oldData = r
        
        return r
    
    def getSlice(self, ind):
        if self.ndim == 2:
            assert(ind == 0)
            return self[:,:]
        elif self.ndim == 3:
            #3D
            return atleast_nd(self[:, :, ind].squeeze(), 2)
        elif self.ndim == 4:
            #4D. getSlice should collapse last 2 dimensions
            return atleast_nd(self[:,:,ind % self.shape[2], ind // self.shape[2]].squeeze(), 2)
        elif self.ndim == 5:
            zi = ind % self.shape[2]
            ti = (ind // self.shape[2]) % self.shape[3]
            ci = ind //(self.shape[3]*self.shape[2])
            return atleast_nd(self[:,:,zi, ti, ci].squeeze(), 2)
        else:
            raise RuntimeError('unsupported ndim = %d' % ndim)
    
    def getSliceShape(self):
        return tuple(self.shape[:2])

    
    def getNumSlices(self):
        return np.prod(self.shape[2:])
    
class XYZTCArrayDataSource(ArrayDataSource):
    def __init__(self, data):
        ArrayDataSource.__init__(self, data)
        
        if self._ndim == 4:
            self._shape = DefaultList(self._shape[:3] + [1, self._shape[3]])
            
    @property
    def ndim(self):
        return 5
            
    def __getitem__(self, keys):
        keys = list(keys)
        
        if self._ndim == 4:
            t = keys.pop(3)
            #assert(t == 0)
            
        return ArrayDataSource.__getitem__(self, tuple(keys))
            
