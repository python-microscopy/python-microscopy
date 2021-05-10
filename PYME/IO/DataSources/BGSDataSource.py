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

from .BaseDataSource import XYTCDataSource, XYZTCDataSource, XYZTCWrapper
import numpy as np

from six.moves import xrange

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

            if self.buffer is None: #buffer doesn't exist yet
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
        
class bgFrameBuffer:
    MAXSHORT = 65535
    MAXIDX = 10000
    def __init__(self, initialSize = 30, percentile=.25):
        self.frameBuffer = None
        self.indices = None
        self.initSize = initialSize
        self.frameNos = {}
        self.availableSlots = []
        self.validData = None
        
        self.pctile = percentile
        
        self.curBG = None
        
    def _indexAdd(self,data, slot):
        self.indices[slot,:,:] = 0
        for i in xrange(self.frameBuffer.shape[0]):
            if not i == slot:
                dg = (self.frameBuffer[i,:,:] > data).astype('uint16')
                self.indices[slot, :,:] += (1 - dg)
                self.indices[i, :,:] += dg
                #print i, self.indices[slot,:,:]
            
        
    def addFrame(self, frameNo, data):
        if len(self.availableSlots) == 0:
            self._growBuffer(data)
            
        slot = self.availableSlots.pop()
        self.frameNos[frameNo] = slot
        self.frameBuffer[slot, :,:] = data
        self.validData[slot] = 1
        
        #dg = self.frameBuffer <= data
        
        #self.indices[slot, :, :] = dg.sum(0)
        #self.indices += (dg < 1)
        self._indexAdd(data, slot)
        
    def removeFrame(self, frameNo):
        slot = self.frameNos.pop(frameNo)
        
        
        self.frameBuffer[slot, :,:] = self.MAXSHORT
        self.indices -= (self.indices > self.indices[slot, :,:])
        self.indices[slot, :,:] = self.MAXIDX
        
        self.validData[slot] = 0        
        self.availableSlots.append(slot)
        
    def _createBuffers(self, size, shape, dtype):
        bufShape = (size,) + shape #[:2]
        self.frameBuffer = self.MAXSHORT*np.ones(bufShape, dtype)
        self.indices = self.MAXIDX*np.ones(bufShape, np.uint16)
        self.validData = np.zeros(size, np.bool)
        
    def _growBuffer(self, data=None):
        if self.frameBuffer is None:
            #starting from scratch
            self._createBuffers(self.initSize, data.shape, data.dtype)
            
            self.availableSlots += list(range(self.initSize))
            
        else:
            #keep a copy of the existing data
            ofb = self.frameBuffer
            oi = self.indices
            ov = self.validData
            
            #make new buffers half as large again
            oldsize = ofb.shape[0]
            newsize = int(oldsize*1.5)
            self._createBuffers(newsize, ofb.shape[1:], ofb.dtype)
            
            self.frameBuffer[:oldsize, :,:] = ofb
            self.indices[:oldsize, :,:] = oi
            self.validData[:oldsize] = ov
            
            #add new frames to list of availiable frames
            self.availableSlots += list(range(oldsize, newsize))
            
    def getPercentile(self, pctile):
        pcIDX = int(self.validData.sum()*pctile)
        print(pcIDX)
        
        return (self.frameBuffer*(self.indices==pcIDX)).max(0).squeeze()
            
        
        
class backgroundBufferM:
    def __init__(self, dataBuffer, percentile=0.5, offset=0):
        self.dataBuffer = dataBuffer
        self.curFrames = set()
        self.curBG = np.zeros(dataBuffer.dataSource.getSliceShape(), 'f4')
        self.offset = offset #using a percentile not equal to 0.5 will lead to bias
        
        self.bfb = bgFrameBuffer(percentile=percentile)
        self.pctile = percentile
        
        self.bgSegs = None

    def getBackground(self, bgindices):
        bgi = set(bgindices)
        
        if bgi == self.curFrames:
            return self.curBG

        #subtract frames we're currently holding but don't need
        for fi in self.curFrames.difference(bgi):
            self.bfb.removeFrame(fi)

        #add frames we don't already have
        nSlices = self.dataBuffer.dataSource.getNumSlices()
        for fi in bgi.difference(self.curFrames):
            if fi >= nSlices:
                #drop frames which run over the end of our data
                bgi.remove(fi)
            else:
                self.bfb.addFrame(fi, self.dataBuffer.getSlice(fi).squeeze())

        self.curFrames = bgi
        self.curBG = self.bfb.getPercentile(self.pctile).astype('f')
        
        med = self.bfb.getPercentile(0.5).astype('f')
        off = np.median(med.flat[::20]) - np.median(self.curBG.flat[::20]) #correct for the offset introduced in median calculation
        self.curBG += off

        return self.curBG - self.offset

class XTZCBackgroundSource(XYTCDataSource):
    moduleName = 'BGSDataSource'
    def __init__(self, datasource, bgRange=None):
        self.datasource = datasource
        
        self.dBuffer = dataBuffer(self.datasource, 50)
        self.bBufferMn = backgroundBuffer(self.dBuffer)
        self.bBufferP = backgroundBufferM(self.dBuffer)
        self.bBuffer = self.bBufferMn
        
        self.bgRange = bgRange
        self.dataStart = 0
        
    @property
    def additionalDims(self):
        return self.datasource.additionalDims
    
    @property
    def sizeC(self):
        return self.datasource.sizeC
        
    def setBackgroundBufferPCT(self, pctile=0):
        if not pctile == 0:
            self.bBufferP.pctile = pctile
            self.bBuffer = self.bBufferP
        else:
            self.bBuffer = self.bBufferMn

    def getSlice(self, ind):
        sl = self.dBuffer.getSlice(ind)
        
        if self.bgRange:
            if (len(self.bgRange) == 3):
                step = self.bgRange[2]
            else:
                step = 1
            bgi = list(range(max(ind + self.bgRange[0],self.dataStart), max(ind + self.bgRange[1],self.dataStart), step))
            #print len(bgi)
            if len(bgi) > 0:
                return sl - self.bBuffer.getBackground(bgi)
            else:
                return sl.astype('f4')
        else:
            return sl.astype('f4')


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


class XYZTCBgsSource(XYZTCDataSource):
    moduleName = 'BGSDataSource'
    
    def __init__(self, datasource, bgRange=None):
        if (not isinstance(datasource, XYZTCDataSource)) and (not datasource.ndim == 5) :
            datasource = XYZTCWrapper.auto_promote(datasource)
        
        self._datasource = datasource
        
        self.dBuffer = dataBuffer(self._datasource, 50)
        self.bBufferMn = backgroundBuffer(self.dBuffer)
        self.bBufferP = backgroundBufferM(self.dBuffer)
        self.bBuffer = self.bBufferMn
        
        self.bgRange = bgRange
        self.dataStart = 0
        
        size_z, size_t, size_c = datasource.shape[2:]
        
        XYZTCDataSource.__init__(self, input_order=datasource._input_order, size_z=size_z, size_t=size_t, size_c=size_c)
    
    def setBackgroundBufferPCT(self, pctile=0):
        if not pctile == 0:
            self.bBufferP.pctile = pctile
            self.bBuffer = self.bBufferP
        else:
            self.bBuffer = self.bBufferMn
    
    def getSlice(self, ind):
        sl = self.dBuffer.getSlice(ind)
        
        if self.bgRange:
            if (len(self.bgRange) == 3):
                step = self.bgRange[2]
            else:
                step = 1
            bgi = list(
                range(max(ind + self.bgRange[0], self.dataStart), max(ind + self.bgRange[1], self.dataStart), step))
            #print len(bgi)
            if len(bgi) > 0:
                return sl - self.bBuffer.getBackground(bgi)
            else:
                return sl.astype('f4')
        else:
            return sl.astype('f4')
    
    def __getattr__(self, item):
        return getattr(self._datasource, item)
    
    def reloadData(self):
        self._datasource.reloadData()
        
        self.dBuffer = dataBuffer(self._datasource, 50)
        self.bBuffer = backgroundBuffer(self.dBuffer)


    def getSliceShape(self):
        return self._datasource.getSliceShape()

    def getNumSlices(self):
        return self._datasource.getNumSlices()

    def getEvents(self):
        return self._datasource.getEvents()

    @property
    def is_complete(self):
        return self._datasource.is_complete()

DataSource = XYZTCBgsSource