#!/usr/bin/python

"""
@author: zacsimile
"""
import numpy as np
from numpy.core.fromnumeric import searchsorted
from .BaseDataSource import XYZTCDataSource

class DataSource(XYZTCDataSource):
    """
    This lets us concatenate datasources and access them via a single data source.
    """
    def __init__(self, datasources, *args, **kwargs):


        self.n_datasources = len(datasources)
        self.datasources = datasources
        self._slice_shape = self.datasources[0].getSliceShape()

        for ds in self.datasources[1:]:
            assert(ds.getSliceShape() == self._slice_shape)

        self.datasource_index = np.hstack([0, np.cumsum([ds.getNumSlices() for ds in self.datasources])])
        self._n_slices = self.datasource_index[-1]

        if not (kwargs.get('size_z', None) and kwargs.get('size_t', None) and kwargs.get('size_c', None)):
            # should also conveniently work for 0,0,0
            kwargs['size_z'] = self._n_slices
     
        XYZTCDataSource.__init__(self, *args, **kwargs)

    def getSlice(self, ind):
        i = np.searchsorted(self.datasource_index, ind, side='right')-1
        rem = ind-self.datasource_index[i]
        return self.datasources[i].getSlice(rem)
       
    def getNumSlices(self):
        return self._n_slices

    def getSliceShape(self):
        return self._slice_shape

    def getEvents(self):
        events = []
        for ds in self.datasources:
            events.append(ds.getEvents())
        return np.hstack(events)

    def release(self):
        for ds in self.datasources:
            ds.release()

    def reloadData(self):
        for ds in self.datasources:
            ds.reloadData()
