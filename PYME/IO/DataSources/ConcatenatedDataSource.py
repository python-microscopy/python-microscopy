#!/usr/bin/python

"""
@author: zacsimile
"""

from typing import AsyncIterable
from numpy.core.fromnumeric import size
from .BaseDataSource import XYZTCDataSource, XYZTCWrapper

class DataSource(XYZTCDataSource):
    """
    This lets us concatenate datasources and access them via a single data source.
    """
    def __init__(self, datasources, dimension='Z'):
        # we will concatenate and index datasources along this dimension
        self._dimension = dimension

        # assert dimension is valid: real dimension, exists in this data
        assert(self._dimension in 'XYZTC')

        self.n_datasources = len(datasources)
        self.datasources = []

        def auto_promote(ds):
            if (not isinstance(ds, XYZTCDataSource)) and (not ds.ndim == 5):
                    return XYZTCWrapper.auto_promote(ds)
            return ds

        self.datasources.append(auto_promote(datasources[0]))
        self._slice_shape = self.datasources[0].getSliceShape()
        self._input_order = self.datasources[0]._input_order
        self._sizes = self.datasources[0]._sizes

        # Populate with XYZTC datasources
        for ds in datasources[1:]:
            pds = auto_promote(ds)

            # Input order must match
            assert(pds._input_order == self._input_order)

            # For now, only let us concatenate datasources of the same size
            assert(pds.getSliceShape() == self._slice_shape)
            assert(pds._sizes == self._sizes)

            self.datasources.append(pds)

        size_z, size_t, size_c = self._sizes
        self._size_d = size_z*size_t*size_c

        if self._dimension in 'ZTC':
            if self._dimension == 'Z':
                self._n_slices = size_z*self.n_datasources
                size_z = self._n_slices
            elif self._dimension == 'T':
                self._n_slices = size_t*self.n_datasources
                size_t = self._n_slices
            elif self._dimension == 'C':
                self._n_slices = size_z  # TODO: should depend on dimension ordering?
                size_c = size_c*self.n_datasources
        else:
            raise NotImplementedError("Haven't set up X and Y stitching yet.")

        XYZTCDataSource.__init__(self, self._input_order, size_z, size_t, size_c)

    def getSlice(self, ind):
        # If we've concatenated along ZTC, this is straightforward ...
        if self._dimension in 'ZTC':
            i, rem = ind // self._size_d, ind % self._size_d
            return self.datasources[i].getSlice(rem)
        else:
            raise NotImplementedError("Haven't set up X and Y stitching yet.")

    def getSliceShape(self):
        return self._slice_shape

    def getNumSlices(self):
        return self._n_slices

    def getEvents(self):
        events = []
        for ds in self.datasources:
            events.extend(ds.getEvents())
        return events

    def release(self):
        for ds in self.datasources:
            ds.release()

    def reloadData(self):
        for ds in self.datasources:
            ds.reloadData()
