#!/usr/bin/python

##################
# TiffDataSource.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.ParallelTasks.relativeFiles import getFullFilename
from PYME.FileUtils import readTiff

class DataSource:
    moduleName = 'TiffDataSource'
    def __init__(self, filename, taskQueue):
        self.filename = getFullFilename(filename)#convert relative path to full path
        self.data = readTiff.read3DTiff(self.filename)

    def getSlice(self, ind):
        return self.data[:,:,ind]


    def getSliceShape(self):
        return self.data.shape[:2]

    def getNumSlices(self):
        return self.data.shape[2]

    def getEvents(self):
        return []

    def release(self):
        pass

    def reloadData(self):
        pass
