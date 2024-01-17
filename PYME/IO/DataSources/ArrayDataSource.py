import numpy as np
from .BaseDataSource import BaseDataSource, DefaultList
import tables

_array_types = (np.ndarray, tables.EArray)

# we might not have dask or zarr installed, support them if present
try:
    import dask.array as da
    _array_types = _array_types + (da.Array,)
    _dask_array = da.Array
except ImportError:
    # used for an isinstance check - exploit the fact that it can be an empty
    # tuple and will return false
    _dask_array = ()

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
        
        if isinstance(r, _dask_array):
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


    
# class XYZTCArrayDataSource(ArrayDataSource):
#     def __init__(self, data):
#         ArrayDataSource.__init__(self, data)
        
#         if self._ndim == 4:
#             self._shape = DefaultList(self._shape[:3] + [1, self._shape[3]])
            
#     @property
#     def ndim(self):
#         return 5
            
#     def __getitem__(self, keys):
#         keys = list(keys)
        
#         if self._ndim == 4:
#             t = keys.pop(3)
#             #assert(t == 0)
            
#         return ArrayDataSource.__getitem__(self, tuple(keys))

class XYZTCArrayDataSource(BaseDataSource): #permit indexing with more dimensions larger than len(shape)
    def __init__(self, data, dimOrder='XYZTC'):
        self.data = data
        self.type = 'Array'
        
        if not isinstance(data, _array_types): # is a data source
            # TODO - duck type instead?
            raise DeprecationWarning('Expecting array data')
        
        #self.additionalDims = dimOrder[2:len(data.shape)]
        
        self._dim_order = dimOrder
        
        print(dimOrder)

        self._dim_to_idx = {k : i for i, k in enumerate(dimOrder) }

        print(self._dim_to_idx)

        
        self._shape = np.ones(5, 'i')
        self._slice_order = np.zeros(5, 'i')
        for j, ax in enumerate('XYZTC'):
            ix = self._dim_to_idx[ax]
            print(ix)
            self._slice_order[ix] = j
            if ix < self.data.ndim:
                self._shape[j] = self.data.shape[ix]

        # zarr/dask support
        if hasattr(data, 'chuncks'):
            self.chunks =  np.ones(5, 'i')
            self.chunksize = np.ones(5, 'i')
            for j, ax in enumerate('XYZTC'):
                ix = self._dim_to_idx[ax]
                if ix < self.data.ndim:
                    self.chunks[j] = self.data.chunks[ix]
                    self.chunksize[j] = self.data.chunksize[ix]

        self.chunks = tuple(self.chunks)
        self.chunksize = tuple(self.chunksize)    

        self._transpose = self._dim_to_idx['Y'] < self._dim_to_idx['X']
        
        
        self._oldData = None
        self._oldSlice = None #buffer last lookup
    
    
    
    @property
    def ndim(self):
        return 5
    
    @property
    def shape(self):
        return tuple(self._shape)
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __getattr__(self, name):
        return getattr(self.data, name)
    
    def __getitem__(self, keys):
        keys = list(keys)
        #print keys
        for i in range(len(keys)):
            if not isinstance(keys[i], slice):
                keys[i] = slice(int(keys[i]), int(keys[i]) + 1)
        
        #if keys == self.oldSlice:
        #    return self.oldData
        
        self.oldSlice = keys
        
        keys = [keys[j] for j in self._slice_order]
        keys = keys[:self.data.ndim]
        
        #print keys
        
        #if self.type == 'Array':
        r = atleast_nd(self.data.__getitem__(tuple(keys)), 5)
        if not self._dim_order == 'XYZTC':
            r = np.moveaxis(r, np.arange(5), self._slice_order)
        
        if isinstance(r, _dask_array):
            # make slicing work for dask arrays TODO - revisit??
            r = r.compute()
        
        self.oldData = r
        
        return r
    
    def getSlice(self, ind):
            zi = ind % self.shape[2]
            ti = (ind // self.shape[2]) % self.shape[3]
            ci = ind //(self.shape[3]*self.shape[2])
            d = atleast_nd(self[:,:,zi, ti, ci].squeeze(), 2)

            #if self._transpose:
            #    d = d.T

            return d
    
    def getSliceShape(self):
        return tuple(self.shape[:2])

    
    def getNumSlices(self):
        return np.prod(self.shape[2:])
            
