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
from PYME.IO.DataSources.BaseDataSource import BaseDataSource, DefaultList, XYZTCDataSource
from PYME.IO.DataSources.ArrayDataSource import ArrayDataSource

class ListWrap:
    def __init__(self, dataList, listDim = None):
        self.dataList = dataList
        self.wrapList = [Wrap(d) for d in dataList]

        if not listDim is None:
            self.listDim = listDim
        else:
            self.listDim = self.wrapList[0].nTrueDims

        self.shape = DefaultList([self.wrapList[0].shape[i] for i in range(self.listDim)] + [len(self.wrapList),])

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
            return np.concatenate([self.wrapList[i].__getitem__(keys[:self.listDim]) for i in range(*kL.indices(len(self.wrapList)))], self.listDim)
        else:
            return self.wrapList[kL].__getitem__(keys[:self.listDim])



        
        
# class CropDataWrap: #permit indexing with more dimensions larger than len(shape)
#     def __init__(self, data, xslice=None, yslice=None, zslice=None):
#         self.data = data
#         self.type = 'Array'
#
#
#
#         self.dim_1_is_z = False
#
#         if not isinstance(data, (np.ndarray, tables.EArray)): # is a data source
#             self.type = 'DataSource'
#             #self.shape = data.getSliceShape() + (data.getNumSlices(),)
#             #print self.shape
#             #self.data.shape = self.shape
#             #self.nTrueDims = 3
#             #self.dim_1_is_z = True
#         #else:
#         self.nTrueDims = len(data.shape)
#         #self.shape = data.shape# + (1, 1, 1, 1, 1)
#         self.oldData = None
#         self.oldSlice = None #buffer last lookup
#
#
#         if data.__class__ == tables.EArray:
#              self.dim_1_is_z = True
#              #self.shape = self.shape[1:3] + (self.shape[0],) + self.shape[3:]
#
#         self._datashape = data.shape
#
#         if xslice is None:
#             self.xslice = slice(0, self._datashape[0])
#         else:
#             self.xslice = xslice
#
#         if zslice is None:
#             self.zslice = slice(0, self._datashape[2])
#         else:
#             self.zslice = zslice
#
#         if yslice is None:
#             self.yslice = slice(0, self._datashape[1])
#         else:
#             self.yslice = yslice
#
#         #self.shape = DefaultList(data.shape)
#
#     @property
#     def shape(self):
#         #if self.type == 'DataSource':
#         #    shape = self.data.getSliceShape() + (self.data.getNumSlices(),)
#         #else: #self.type == 'Array'
#         shape = self._datashape
#
#         if self.dim_1_is_z:
#             shape = shape[1:3] + (shape[0],) + shape[3:]
#
#         #if not self.xslice == None:
#         shape[0] = len(range(*self.xslice.indices(shape[0])))
#
#         #if not self.yslice == None:
#         shape[1] = len(range(*self.yslice.indices(shape[1])))
#
#         #if not self.yslice == None:
#         shape[2] = len(range(*self.zslice.indices(shape[2])))
#
#         return DefaultList(shape)
#
#
#
#     def __getattr__(self, name):
#         return getattr(self.data, name)
#
#     def __getitem__(self, keys):
#         keys = list(keys)
#         #print keys
#         for i in range(len(keys)):
#             if not isinstance(keys[i],slice):
#                 keys[i] = slice(keys[i],keys[i] + 1)
#         #if keys == self.oldSlice:
#         #    return self.oldData
#         self.oldSlice = keys
#         if len(keys) > len(self.data.shape):
#             keys = keys[:len(self.data.shape)]
#         if self.dim_1_is_z:
#             keys = [keys[2]] + keys[:2] + keys[3:]
#
#         #print keys
#
#         if self.type == 'Array':
#             r = self.data.__getitem__(keys)
#         else:
#             r = np.concatenate([np.atleast_3d(self.data[self.xslice, self.yslice, self.zslice, j][keys[0], keys[1], keys[2]])[:,:,:,None] for i in (range(self.shape[3]))], 3)
#
#         self.oldData = r
#
#         return r
#
#     def getSlice(self, ind):
#         return self[:,:,ind].squeeze()
#
#     def getSliceShape(self):
#         return tuple(self.shape[:2])
#
#     def getNumSlices(self):
#         return self.shape[2]


def Wrap(datasource):
    """Wrap a data source such that it is indexable like a numpy array."""
    
    if isinstance(datasource, list):
        datasource = ListWrap(datasource)
    elif not isinstance(datasource, (ListWrap, BaseDataSource, XYZTCDataSource)): #only if not already wrapped
        if isinstance(datasource, tables.EArray):
            datasource = ArrayDataSource(datasource, dim_1_is_z=True)
        else:
            datasource = ArrayDataSource(datasource)

    return datasource
