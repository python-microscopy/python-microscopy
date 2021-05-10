#!/usr/bin/python
##################
# dataWrap.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################
"""Classes to wrap a source of data so that it looks like an array"""
import numpy as np
import tables
from PYME.IO.DataSources.BaseDataSource import DefaultList, BaseDataSource
from PYME.IO.DataSources.ArrayDataSource import ArrayDataSource

def atleast_nd(a, n):
    while a.ndim < n:
        a = np.expand_dims(a, a.ndim)
        
    return a

class ListWrapper(BaseDataSource):
    def __init__(self, dataList):
        self.dataList = dataList
        self.wrapList = [Wrap(d) for d in dataList]

        self.listDim = self.wrapList[0].ndim
        self._shape = DefaultList([self.wrapList[0].shape[i] for i in range(self.listDim)] + [len(self.wrapList),])
        
        self._ds_n_slices = self.wrapList[0].getNumSlices()

    def getSlice(self, ind):
        n = int(ind/self._ds_n_slices)
        i = int(ind % self._ds_n_slices)
        
        return self.wrapList[n].getSlice(i)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self.listDim + 1

    @property
    def dtype(self):
        return self.wrapList[0].dtype

    def getSliceShape(self):
        return self.wrapList[0].getSliceShape()

    def getNumSlices(self):
        return self._ds_n_slices*len(self.wrapList)

    def getEvents(self):
        return self.wrapList[0].getEvents()

    @property
    def is_complete(self):
        return self.wrapList[0].is_complete()
    
    def __getattr__(self, name):
        return getattr(self.wrapList[0], name)

    def __getitem__(self, keys):
        keys = list(keys)
        #print keys

        if len(keys) > self.listDim:
            kL = keys[self.listDim]
        else:
            kL = 0 #default to taking the first channel
            
        #if kL.__class__ == slice:
        #    return ListWrap([self.wrapList[i].__getitem__(keys[:self.listDim]) for i in range(*kL.indices(len(self.wrapList)))])

        if isinstance(kL, slice):
            return np.concatenate([atleast_nd(self.wrapList[i].__getitem__(keys[:self.listDim]), self.ndim) for i in range(*kL.indices(len(self.wrapList)))], self.listDim)
        else:
            return self.wrapList[kL].__getitem__(keys[:self.listDim])


def Wrap(datasource):
    """Wrap a data source such that it is indexable like a numpy array."""
    
    if isinstance(datasource, list):
        datasource = ListWrapper(datasource)
    elif not isinstance(datasource, (BaseDataSource,)): #only if not already wrapped
        if isinstance(datasource, tables.EArray):
            datasource = ArrayDataSource(datasource, dim_1_is_z=True)
        else:
            datasource = ArrayDataSource(datasource)

    return datasource
