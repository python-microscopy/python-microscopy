#!/usr/bin/python
##################
# UnsplitDataSource.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
#import numpy

class DataSource: 
    moduleName = 'UnsplitDataSource'
    def __init__(self,dataSource, unmixer, mixmatrix, offset, ROI, chan=0):
        self.unmixer = unmixer
        self.dataSource = dataSource
        self.sliceShape = list(self.dataSource.getSliceShape())
        self.sliceShape[1]/=2
        
        self.mixmatrix = mixmatrix
        self.offset = offset
        self.ROI = ROI
        self.chan = chan

    def getSlice(self,ind):
        sl = self.dataSource.getSlice(ind)
        
        sl = self.unmixer.Unmix(sl, self.mixMatrix, self.offset, self.ROI)

        return sl[:,:,self.chan]

    def getSliceShape(self):
        #return (self.im.size[1], self.im.size[0])
        return self.sliceShape
        #return self.data.shape[:2]

    def getNumSlices(self):
        return self.dataSource.getNumSlices()

    def getEvents(self):
        return self.dataSource.getEvents()

    def release(self):
        return self.dataSource.release()
        

    def reloadData(self):
        return self.dataSource.reloadData()
        


