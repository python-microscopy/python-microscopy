#!/usr/bin/python
##################
# BufferedDataSource.py
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
import numpy
from .BaseDataSource import XYZTCDataSource, XYZTCWrapper
import threading

class DataSource(XYZTCDataSource): #buffer our io to avoid decompressing multiple times
    moduleName = 'BufferedDataSource'
    def __init__(self,datasource, bLen = 12):
        
        if (not isinstance(datasource, XYZTCDataSource)) and (not datasource.ndim == 5) :
            datasource = XYZTCWrapper.auto_promote(datasource)
        
        self.bLen = bLen
        self.buffer = None #delay creation until we know the dtype
        #self.buffer = numpy.zeros((bLen,) + dataSource.getSliceShape(), 'uint16')
        self.insertAt = 0
        self.bufferedSlices = -1*numpy.ones((bLen,), 'i')
        self.dataSource = datasource

        self.lock = threading.Lock()

        size_z, size_t, size_c = datasource.shape[2:]
        XYZTCDataSource.__init__(self, input_order=datasource._input_order, size_z=size_z, size_t=size_t, size_c=size_c)

    def getSlice(self,ind):
        #global bufferMisses
        #print self.bufferedSlices, self.insertAt, ind
        #return self.dataSource.getSlice(ind)
        with self.lock:
            #print ind
            if ind in self.bufferedSlices: #return from buffer
                #print int(numpy.where(self.bufferedSlices == ind)[0])
                #print self.bufferedSlices
                ret = self.buffer[int(numpy.where(self.bufferedSlices == ind)[0]), :, :].copy()
                #print 'buf'
            else: #get from our data source and store in buffer
                sl = self.dataSource.getSlice(ind)
                self.bufferedSlices[self.insertAt] = ind
                #print sl.shape
                #print self.insertAt
                #print self.buffer

                if self.buffer is None: #buffer doesn't exist yet
                    self.buffer = numpy.zeros((self.bLen, ) + self.dataSource.getSliceShape(), sl.dtype)

                #print self.buffer.shape

                self.buffer[self.insertAt,:, :] = sl
                self.insertAt += 1
                self.insertAt %= self.bLen

                #bufferMisses += 1

                ret = sl
            return ret
    
    def getSliceShape(self):
        #return (self.im.size[1], self.im.size[0])
        return self.dataSource.getSliceShape()
        #return self.data.shape[:2]

    def getNumSlices(self):
        return self.dataSource.getNumSlices()

    def getEvents(self):
        return self.dataSource.getEvents()

    def release(self):
        return self.dataSource.release()

    def reloadData(self):
        return self.dataSource.reloadData()

    @property
    def is_complete(self):
        return self.dataSource.is_complete()
        


