# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:45:24 2015

@author: david
"""
import numpy as np

#bufferMisses = 0

class dataBuffer: #buffer our io to avoid decompressing multiple times
    def __init__(self,dataSource, bLen = 12):
        self.bLen = bLen
        self.buffer = None #delay creation until we know the dtype
        #self.buffer = np.zeros((bLen,) + dataSource.getSliceShape(), 'uint16')
        self.insertAt = 0
        self.bufferedSlices = -1*np.ones((bLen,), 'i')
        self.dataSource = dataSource
        
    def getSlice(self,ind):
        #global bufferMisses
        #print self.bufferedSlices, self.insertAt, ind
        #return self.dataSource.getSlice(ind)
        if ind in self.bufferedSlices: #return from buffer
            #print int(np.where(self.bufferedSlices == ind)[0])
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
        
    def addFrame(self, frameNo, data):
        if len(self.availableSlots) == 0:
            self._growBuffer(data)
            
        slot = self.availableSlots.pop()
        self.frameNos[frameNo] = slot
        self.frameBuffer[slot, :,:] = data
        self.validData[slot] = 1
        
        dg = self.frameBuffer <= data
        
        self.indices[slot, :, :] = dg.sum(0)
        self.indices += (dg < 1)
        
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
        if self.frameBuffer == None:
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
    def __init__(self, dataBuffer, percentile=.5):
        self.dataBuffer = dataBuffer
        self.curFrames = set()
        self.curBG = np.zeros(dataBuffer.dataSource.getSliceShape(), 'f4')
        
        self.bfb = bgFrameBuffer(percentile=percentile)
        
        self.bgSegs = None
        self.pctile = percentile

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
        off = np.median(med) - np.median(self.curBG)
        self.curBG += off

        return self.curBG