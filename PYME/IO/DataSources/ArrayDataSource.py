import numpy as np
from .BaseDataSource import BaseDataSource
import tables

class ArrayDataSource(BaseDataSource): #permit indexing with more dimensions larger than len(shape)
    def __init__(self, data, dimOrder='XYTC', dim_1_is_z=False):
        self.data = data
        self.type = 'Array'
        
        self.dim_1_is_z = dim_1_is_z
        
        if not isinstance(data, (np.ndarray, tables.EArray)): # is a data source
            raise DeprecationWarning('Expecting array data')
        
        self.additionalDims = dimOrder[2:len(data.shape)]
        
        if len(self.data.shape) > 3:
            self.sizeC = self.data.shape[3]
        
        self.oldData = None
        self.oldSlice = None #buffer last lookup
    
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
        
        if len(keys) > len(self.data.shape):
            keys = keys[:len(self.data.shape)]
        if self.dim_1_is_z:
            keys = [keys[2]] + keys[:2] + keys[3:]
        
        #print keys
        
        #if self.type == 'Array':
        r = self.data.__getitem__(tuple(keys))
        #else:
        #    raise DeprecationWarning('We should only be wrapping arrays')
        #r = np.concatenate([np.atleast_2d(self.data.getSlice(i)[keys[0], keys[1]])[:, :, None] for i in
        #                    range(*keys[1].indices(self.data.getNumSlices()))], 2)
        
        self.oldData = r
        
        return r
    
    def getSlice(self, ind):
        if len(self.data.shape) == 3:
            #3D
            return self[:, :, ind].squeeze()
        elif len(self.data.shape) == 4:
            #4D. getSlice should collapse last 2 dimensions
            return self[:,:,ind % self.data.shape[2], ind // self.data.shape[2]].squeeze()
    
    def getSliceShape(self):
        if self.dim_1_is_z:
            return tuple(self.data.shape[1:3])
        else:
            return tuple(self.data.shape[:2])
    
    def getNumSlices(self):
        if self.dim_1_is_z:
            return self.data.shape[0]
        else:
            if len(self.data.shape) > 2:
                return np.prod(self.data.shape[2:])
            else:
                return 1