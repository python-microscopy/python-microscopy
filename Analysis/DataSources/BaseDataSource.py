# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:22:33 2012

@author: david
"""
import numpy as np
class DefaultList(list):
    '''List which returns a default value for items not in the list'''
    def __init__(self, *args):
        list.__init__(self, *args)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return 1


class BaseDataSource(object):
    oldData = None
    oldSlice = None
    nTrueDims =3
    
    @property
    def shape(self):
        #if self.type == 'DataSource':
        return DefaultList(self.getSliceShape() + (self.getNumSlices(),) )
        
    def getSlice(self, ind):
        raise NotImplementedError

    def getSliceShape(self):
        raise NotImplementedError

    def getNumSlices(self):
        raise NotImplementedError

    def getEvents(self):
        raise NotImplementedError
        
    def __getitem__(self, keys):
        keys = list(keys)
        #print keys
        for i in range(len(keys)):
            if not keys[i].__class__ == slice:
                keys[i] = slice(keys[i],keys[i] + 1)
        if keys == self.oldSlice:
            return self.oldData
        self.oldSlice = keys
        #if len(keys) > len(self.data.shape):
        #    keys = keys[:len(self.data.shape)]
        #if self.dim_1_is_z:
        #    keys = [keys[2]] + keys[:2] + keys[3:]

        #print keys

        r = np.concatenate([np.atleast_2d(self.getSlice(i)[keys[0], keys[1]])[:,:,None] for i in range(*keys[2].indices(self.getNumSlices()))], 2)

        self.oldData = r

        return r