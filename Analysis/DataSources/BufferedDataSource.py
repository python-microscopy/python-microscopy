#!/usr/bin/python
##################
# BufferedDataSource.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy

class DataSource: #buffer our io to avoid decompressing multiple times
    moduleName = 'BufferedDataSource'
    def __init__(self,dataSource, bLen = 12):
        self.bLen = bLen
        self.buffer = None #delay creation until we know the dtype
        #self.buffer = numpy.zeros((bLen,) + dataSource.getSliceShape(), 'uint16')
        self.insertAt = 0
        self.bufferedSlices = -1*numpy.ones((bLen,), 'i')
        self.dataSource = dataSource

    def getSlice(self,ind):
        #global bufferMisses
        #print self.bufferedSlices, self.insertAt, ind
        #return self.dataSource.getSlice(ind)
        if ind in self.bufferedSlices: #return from buffer
            #print int(numpy.where(self.bufferedSlices == ind)[0])
            return self.buffer[int(numpy.where(self.bufferedSlices == ind)[0]),:,:]
        else: #get from our data source and store in buffer
            sl = self.dataSource.getSlice(ind)
            self.bufferedSlices[self.insertAt] = ind
            #print sl.shape
            #print self.insertAt
            #print self.buffer

            if self.buffer == None: #buffer doesn't exist yet
                self.buffer = numpy.zeros((self.bLen,) + self.dataSource.getSliceShape(), sl.dtype)
                
            #print self.buffer.shape

            self.buffer[self.insertAt, :,:] = sl
            self.insertAt += 1
            self.insertAt %=self.bLen

            #bufferMisses += 1

            return sl

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
        


