#!/usr/bin/python
##################
# dataWrap.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
'''Classes to wrap a source of data so that it looks like an array'''
import numpy as np
import tables

class DefaultList(list):
    '''List which returns a default value for items not in the list'''
    def __init__(self, *args):
        list.__init__(self, *args)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return 1



class ListWrap:
    def __init__(self, dataList, listDim = None):
        self.dataList = dataList
        self.wrapList = [Wrap(d) for d in dataList]

        if not listDim == None:
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

        return self.wrapList[kL].__getitem__(keys[:self.listDim])


class DataWrap: #permit indexing with more dimensions larger than len(shape)
    def __init__(self, data):
        self.data = data
        self.type = 'Array'

        self.dim_1_is_z = False

        if not isinstance(data, np.ndarray) and not isinstance(data, tables.EArray): # is a data source
            self.type = 'DataSource'
            self.shape = data.getSliceShape() + (data.getNumSlices(),)
            #print self.shape
            self.data.shape = self.shape
            self.dim_1_is_z = True

        self.nTrueDims = len(data.shape)
        #self.shape = data.shape# + (1, 1, 1, 1, 1)
        self.oldData = None
        self.oldSlice = None #buffer last lookup


        if data.__class__ == tables.EArray:
             self.dim_1_is_z = True
             self.shape = self.shape[1:3] + (self.shape[0],) + self.shape[3:]

        self.shape = DefaultList(data.shape)

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __getitem__(self, keys):
        keys = list(keys)
        #print keys
        for i in range(len(keys)):
            if not keys[i].__class__ == slice:
                keys[i] = slice(keys[i],keys[i] + 1)
        if keys == self.oldSlice:
            return self.oldData
        self.oldSlice = keys
        if len(keys) > len(self.data.shape):
            keys = keys[:len(self.data.shape)]
        if self.dim_1_is_z:
            keys = [keys[2]] + keys[:2] + keys[3:]

        #print keys

        if self.type == 'Array':
            r = self.data.__getitem__(keys)
        else:
            r = np.concatenate([np.atleast_2d(self.data.getSlice(i)[keys[1], keys[2]])[:,:,None] for i in range(*keys[0].indices(self.data.getNumSlices()))], 2)

        self.oldData = r

        return r


def Wrap(datasource):
    '''Wrap a data source such that it is indexable like a numpy array.'''
    
    if datasource.__class__ ==list:
        datasource = ListWrap(datasource)
    elif not datasource.__class__ in [DataWrap, ListWrap]: #only if not already wrapped
        datasource = DataWrap(datasource)

    return datasource