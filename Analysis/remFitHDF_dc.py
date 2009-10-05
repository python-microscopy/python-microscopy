#!/usr/bin/python

##################
# remFitHDF_dc.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.ParallelTasks import taskDef
from PYME.ParallelTasks.relativeFiles import getFullFilename
import ofind
import matplotlib
#import read_kdf
import numpy

#import tables

tables= None

h5File = None
h5Filename = None
h5Buffer = None

bufferMisses = 0

lastIndex = -10
pointsLast = None

#matplotlib.interactive(False)
from pylab import *

import copy

def tqPopFcn(workerN, NWorkers, NTasks):
    return workerN * NTasks/NWorkers #let each task work on its own chunk of data ->
    

def reloadH5File():
    global h5File
    h5File.close()
    h5File = tables.openFile(h5Filename)
    return h5File

class fitResult(taskDef.TaskResult):
    def __init__(self, task, results):
        taskDef.TaskResult.__init__(self, task)
        self.filename = task.filename
        #self.seriesName = task.seriesName
        self.index = task.index
        self.results = results

class hdfBuffer: #buffer our io to avoid decompressing multiple times
    def __init__(self,h5File, bLen = 12):
        self.bLen = bLen
        self.buffer = numpy.zeros((bLen,) + h5File.root.ImageData.shape[1:], 'uint16')
        self.insertAt = 0
        self.bufferedSlices = list(-1*numpy.ones((bLen,), 'i'))
        self.h5File = h5File
        
    def getSlice(self,ind):
        global bufferMisses
        if ind in self.bufferedSlices:
            return self.buffer[self.bufferedSlices.index(ind),:,:]
        else:
            if ind >= self.h5File.root.ImageData.shape[0]:
                self.h5File = reloadH5File() #try reloading the data in case it's grown
            sl = self.h5File.root.ImageData[ind, :,:]
            self.bufferedSlices[self.insertAt] = ind
            self.buffer[self.insertAt, :,:] = sl
            self.insertAt += 1
            self.insertAt %=self.bLen

            bufferMisses += 1
            
            return sl
             
        

class fitTask(taskDef.Task):
    def __init__(self, filename, index, threshold, metadata, fitModule, bgindices = [], SNThreshold = False):
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
        self.index = index

        self.bgindices = bgindices

	self.md = metadata
        
        self.fitModule = fitModule
        self.SNThreshold = SNThreshold


    def __call__(self, gui=False, taskQueue=None):
        global h5File, h5Filename, h5Buffer, tables
        
        self.filename = getFullFilename(self.filename) #generate ourselves a full file name from the relative filename
        #print self.filename

        tables = __import__('tables') #dodgy workaround for bug in pytables when using Pyro under windows

        
        fitMod = __import__('PYME.Analysis.FitFactories.' + self.fitModule, fromlist=['PYME', 'Analysis','FitFactories']) #import our fitting module

        #read the data
        #self.data = read_kdf.ReadKdfData(self.filename)
        if not h5Filename == self.filename: #avoid unnecessary opening and closing of file
            if not h5File == None:
                h5File.close()

            h5File = tables.openFile(self.filename)
            h5Filename = self.filename
            h5Buffer = hdfBuffer(h5File)

            lastIndex = -10
            pointsLast = None

        
        #self.data = h5File.root.ImageData[self.index, :,:]
        self.data = h5Buffer.getSlice(self.index)

        #when camera buffer overflows, empty pictures are produced - deal with these here
        if self.data.max() == 0:
            return fitResult(self, [])
        
        #squash 4th dimension
        #self.data = self.data.reshape((self.data.shape[0], self.data.shape[1],self.data.shape[2]))
        self.data = self.data.reshape((self.data.shape[0], self.data.shape[1],1))
        
        g = self.data[:, :(self.data.shape[1]/2)]
        r = self.data[:, (self.data.shape[1]/2):]
        r = np.fliplr(r)
        
        #calculate background
        self.bg = 0
        if not len(self.bgindices) == 0:
            self.bg = numpy.zeros(self.data.shape, 'f')
            for bgi in self.bgindices:
                #self.bg += h5File.root.ImageData[bgi, :,:].reshape(self.data.shape)
                #self.bg += h5Buffer.getSlice(bgi).reshape(self.data.shape)
                bs = h5Buffer.getSlice(bgi)
                bs = bs.reshape(self.data.shape)
                self.bg = self.bg + bs.astype('f')

            self.bg *= 1.0/len(self.bgindices)

        #Find objects
        #self.ofd = ofind.ObjectIdentifier(self.data.astype('f') - self.bg)
        #self.ofd.FindObjects(self.calcThreshold(),0)

        dat_bg = self.data.astype('f') - self.bg
        
        g_ = dat_bg[:, :(self.data.shape[1]/2)]
        r_ = dat_bg[:, (self.data.shape[1]/2):]
        r_ = np.fliplr(r_)

        self.ofd = self.findObjects(g_ + r_, self.calcThreshold(g + r))

        #if lastIndex == (self.index - 1):
        #    prevPoints = pointsLast
        #else:
        #    d_l = h5Buffer.getSlice(self.index - 1).reshape(self.data.shape).astype('f')
        #    prevPoints = self.findObjects(d_l, self.calcThreshold(d_l))
        
        #If we're running under a gui - display found objects
        if gui:
            clf()
            imshow(self.ofd.filteredData.T, cmap=cm.hot, hold=False)
            plot([p.x for p in self.ofd], [p.y for p in self.ofd], 'o', mew=1, mec='g')
            #axis('image')
            #gca().set_ylim([255,0])
            colorbar()
            show()

        #Create a fit 'factory'
        md = copy.copy(self.md)
        md.tIndex = self.index

        fitFac = fitMod.FitFactory(np.concatenate((g.reshape(g.shape[0], -1, 1), r.reshape(g.shape[0], -1, 1)),2), md)

        #print 'Have Fit Factory'
        
        #perform fit for each point that we detected
        if 'FitResultsDType' in dir(fitMod):
            self.res = numpy.empty(len(self.ofd), fitMod.FitResultsDType)
            for i in range(len(self.ofd)):
                p = self.ofd[i]
                self.res[i] = fitFac.FromPoint(round(p.x), round(p.y))
        else:
            self.res  = [fitFac.FromPoint(round(p.x), round(p.y)) for p in self.ofd]

        lastIndex = self.index
        pointsLast = self.ofd

        return fitResult(self, self.res )

    def findObjects(self,data,threshold):
        #Find objects
        ofd = ofind.ObjectIdentifier(data)
        ofd.FindObjects(threshold,0)

        return ofd

    def calcThreshold(self, data):
        if self.SNThreshold:
            fudgeFactor = 0.25 #to account for the fact that the blurring etc... in ofind doesn't preserve intensities - at the moment completely arbitrary so a threshold setting of 1 results in reasonable detection.
            return numpy.sqrt(numpy.maximum(self.md.CCD.electronsPerCount*(data.astype('f').mean(2) - 2*self.md.CCD.ADOffset)/self.md.CCD.EMGain, 1))*self.md.CCD.electronsPerCount*fudgeFactor*self.threshold
        else:
            return self.threshold
