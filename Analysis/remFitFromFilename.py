#!/usr/bin/python

##################
# remFitFromFilename.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import taskDef
import ofind
import matplotlib
import read_kdf
import numpy


#matplotlib.interactive(False)
from pylab import *

class fitResult:
    def __init__(self, task, results):
        self.taskID = task.taskID
        self.filename = task.filename
        self.seriesName = task.seriesName
        self.results = results

class fitTask(taskDef.Task):
    def __init__(self, filename, seriesName, threshold, metadata, fitModule, bgfiles = [], SNThreshold = False):
        '''Create a new fitting task, which opens data from a supplied filename.
        -------------
        Parameters:
        filename - name of file containing the frame to be fitted
        seriesName - name of the series to which the file belongs (to be used in future for sorting processed data)
        threshold - threshold to be used to detect points n.b. this applies to the filtered, potentially bg subtracted data
        taskDef.Task.__init__(self)
        metadata - image metadata (see MetaData.py)
        fitModule - name of module defining fit factory to use
        bgffiles - (optional) list of files to be averaged and subtracted from image prior to point detection - n.B. fitting is still performed on raw data'''
        taskDef.Task.__init__(self)

        self.threshold = threshold
        self.filename = filename
        self.seriesName = seriesName

        self.bgfiles = bgfiles

	self.md = metadata
        
        self.fitModule = fitModule
        self.SNThreshold = SNThreshold


    def __call__(self, gui=False):
        fitMod = __import__(self.fitModule) #import our fitting module

        #read the data
        self.data = read_kdf.ReadKdfData(self.filename)

        #when camera buffer overflows, empty pictures are produced - deal with these here
        if self.data.max() == 0:
            return fitResult(self, [])
        
        #squash 4th dimension
        self.data = self.data.reshape((self.data.shape[0], self.data.shape[1],self.data.shape[2])) 
        
        #calculate background
        self.bg = 0
        if not len(self.bgfiles) == 0:
            self.bg = numpy.zeros(self.data.shape, 'f')
            for bgfn in self.bgfiles:
                self.bg += read_kdf.ReadKdfData(bgfn).reshape(self.data.shape)

            self.bg /= len(self.bgfiles)

        #Find objects
        self.ofd = ofind.ObjectIdentifier(self.data - self.bg)
        self.ofd.FindObjects(self.calcThreshold(),0)
        
        #If we're running under a gui - display found objects
        if gui:
            clf()
            imshow(self.ofd.filteredData, cmap=cm.hot, hold=False)
            plot([p.y for p in self.ofd], [p.x for p in self.ofd], 'o', mew=1, mec='g', mfc=None)
            #axis('image')
            #gca().set_ylim([255,0])
            colorbar()
            show()

        #Create a fit 'factory'
        fitFac = fitMod.FitFactory(self.data, self.md)
        
        #perform fit for each point that we detected
        self.res  = [fitFac.FromPoint(round(p.x), round(p.y)) for p in self.ofd]

        return fitResult(self, self.res )

    def calcThreshold(self):
        if self.SNThreshold:
            return numpy.sqrt(numpy.maximum(self.data.mean(2) - self.md.CCD.ADOffset, 1))*self.threshold
        else:
            return self.threshold
