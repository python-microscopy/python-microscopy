#!/usr/bin/python

##################
# HDFDataSource.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.ParallelTasks.relativeFiles import getFullFilename
from BaseDataSource import BaseDataSource
#import tables

class DataSource(BaseDataSource):
    moduleName = 'FlatFieldDataSource'
    def __init__(self, parentSource, mdh, flatfield):
        #self.h5Filename = getFullFilename(h5Filename)#convert relative path to full path
        #self.h5File = tables.openFile(self.h5Filename)
        self.source = parentSource
        self.mdh = mdh
        #self.flat = flatfield
        
        x0 = mdh.getEntry('Camera.ROIPosX') - 1
        x1 = x0 + mdh.getEntry('Camera.ROIWidth') + 1

        y0 = mdh.getEntry('Camera.ROIPosY') - 1
        y1 = y0 + mdh.getEntry('Camera.ROIHeight') + 1

        print x0, x1, y0, y1

        #self.offset = mdh.getEntry()

        self.flat = flatfield[x0:x1, y0:y1]


    def getSlice(self, ind):
        #if ind >= self.h5File.root.ImageData.shape[0]:
        #        self.reloadData() #try reloading the data in case it's grown
        print self.getSliceShape(), self.flat.shape
        
        return (self.source.getSlice(ind) - self.mdh.getEntry('Camera.ADOffset'))*self.flat


    def getSliceShape(self):
        return self.source.getSliceShape()

    def getNumSlices(self):
        return self.source.getNumSlices()

    def getEvents(self):
        return self.source.getEvents()

    def release(self):
        self.source.release()
