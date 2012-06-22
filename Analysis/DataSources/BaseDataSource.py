# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:22:33 2012

@author: david
"""

class BaseDataSource(object):
    @property
    def shape(self):
        #if self.type == 'DataSource':
        return (self.getNumSlices(),) + self.getSliceShape()
        
    def getSlice(self, ind):
        raise NotImplementedError

    def getSliceShape(self):
        raise NotImplementedError

    def getNumSlices(self):
        raise NotImplementedError

    def getEvents(self):
        raise NotImplementedError