#!/usr/bin/python

##################
# HDFDataSource.py
#
# Copyright David Baddeley, 2009
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

from BaseDataSource import BaseDataSource
import numpy as np

class dataBuffer: #buffer our io to avoid decompressing multiple times
    def __init__(self,dataSource, bLen = 12):
        self.bLen = bLen
        self.buffer = None #delay creation until we know the dtype
        #self.buffer = numpy.zeros((bLen,) + dataSource.getSliceShape(), 'uint16')
        self.insertAt = 0
        self.bufferedSlices = -1*np.ones((bLen,), 'i')
        self.dataSource = dataSource
        
    def getSlice(self,ind):
        global bufferMisses
        #print self.bufferedSlices, self.insertAt, ind
        #return self.dataSource.getSlice(ind)
        if ind in self.bufferedSlices: #return from buffer
            #print int(numpy.where(self.bufferedSlices == ind)[0])
            return self.buffer[int(np.where(self.bufferedSlices == ind)[0]),:,:]
        else: #get from our data source and store in buffer
            sl = self.dataSource.getSlice(ind)
            self.bufferedSlices[self.insertAt] = ind

            if self.buffer == None: #buffer doesn't exist yet
                self.buffer = np.zeros((self.bLen,) + self.dataSource.getSliceShape(), sl.dtype)
                
            self.buffer[self.insertAt, :,:] = sl
            self.insertAt += 1
            self.insertAt %=self.bLen

            #bufferMisses += 1
            
            #if bufferMisses % 10 == 0:
            #    print nTasksProcessed, bufferMisses

            return sl

class backgroundBuffer:
    def __init__(self, dataBuffer):
        self.dataBuffer = dataBuffer
        self.curFrames = set()
        self.curBG = np.zeros(dataBuffer.dataSource.getSliceShape(), 'f4')

    def getBackground(self, bgindices):
        bgi = set(bgindices)

        #subtract frames we're currently holding but don't need
        for fi in self.curFrames.difference(bgi):
            self.curBG[:] = (self.curBG - self.dataBuffer.getSlice(fi))[:]

        #add frames we don't already have
        nSlices = self.dataBuffer.dataSource.getNumSlices()
        for fi in bgi.difference(self.curFrames):
            if fi >= nSlices:
                #drop frames which run over the end of our data
                bgi.remove(fi)
            else:
                self.curBG[:] = (self.curBG + self.dataBuffer.getSlice(fi))[:]

        self.curFrames = bgi

        return self.curBG/len(bgi)

class DataSource(BaseDataSource):
    moduleName = 'BGSDataSource'
    def __init__(self, datasource, bgRange=None):
        self.datasource = datasource
        
        self.dBuffer = dataBuffer(self.datasource, 50)
        self.bBuffer = backgroundBuffer(self.dBuffer)
        
        self.bgRange = bgRange
        self.dataStart = 0

    def getSlice(self, ind):
        sl = self.dBuffer.getSlice(ind)
        
        if self.bgRange:
            bgi = range(max(ind + self.bgRange[0],self.dataStart), max(ind + self.bgRange[1],self.dataStart))
            #print len(bgi)
            if len(bgi) > 0:
                return sl - self.bBuffer.getBackground(bgi)
            else:
                return sl
        else:
            return sl


    def getSliceShape(self):
        return self.datasource.getSliceShape()

    def getNumSlices(self):
        return self.datasource.getNumSlices()

    def getEvents(self):
        return self.datasource.getEvents()

    def release(self):
        self.datasource.release()

    def reloadData(self):
        self.datasource.reloadData()
        
        self.dBuffer = dataBuffer(self.datasource, 50)
        self.bBuffer = backgroundBuffer(self.dBuffer)
