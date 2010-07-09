#!/usr/bin/python

##################
# remFitBuf.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.ParallelTasks import taskDef
import ofind
#import ofind_nofilt #use for drift estimation - faster
import ofind_xcorr
import numpy
import numpy as np

dBuffer = None
bBuffer = None
dataSourceID = None

bufferMisses = 0

splitterFitModules = ['SplitterFitFR','SplitterFitQR','SplitterFitCOIR', 'BiplaneFitR', 'SplitterShiftEstFR', 'SplitterObjFindR', 'SplitterFitPsfIR']

from pylab import *

import copy

def tqPopFcn(workerN, NWorkers, NTasks):
    return workerN * NTasks/NWorkers #let each task work on its own chunk of data ->
    
class fitResult(taskDef.TaskResult):
    def __init__(self, task, results, driftResults=[]):
        taskDef.TaskResult.__init__(self, task)
        self.index = task.index
        self.results = results
        self.driftResults = driftResults

class dataBuffer: #buffer our io to avoid decompressing multiple times
    def __init__(self,dataSource, bLen = 12):
        self.bLen = bLen
        self.buffer = numpy.zeros((bLen,) + dataSource.getSliceShape(), 'uint16')
        self.insertAt = 0
        self.bufferedSlices = -1*numpy.ones((bLen,), 'i')
        self.dataSource = dataSource
        
    def getSlice(self,ind):
        global bufferMisses
        #print self.bufferedSlices, self.insertAt, ind
        #return self.dataSource.getSlice(ind)
        if ind in self.bufferedSlices: #return from buffer
            #print int(numpy.where(self.bufferedSlices == ind)[0])
            return self.buffer[int(numpy.where(self.bufferedSlices == ind)[0]),:,:]
        else: #get from our data source and store in buffer
            sl = self.dataSource.getSlice(ind)
            self.bufferedSlices[self.insertAt] = ind
            self.buffer[self.insertAt, :,:] = sl
            self.insertAt += 1
            self.insertAt %=self.bLen

            bufferMisses += 1

            return sl
        
class backgroundBuffer:
    def __init__(self, dataBuffer):
        self.dataBuffer = dataBuffer
        self.curFrames = set()
        self.curBG = numpy.zeros(dataBuffer.dataSource.getSliceShape(), 'f4')

    def getBackground(self, bgindices):
        bgi = set(bgindices)

        #subtract frames we're currently holding but don't need
        for fi in self.curFrames.difference(bgi):
            self.curBG[:] = (self.curBG - self.dataBuffer.getSlice(fi))[:]

        #add frames we don't already have
        for fi in bgi.difference(self.curFrames):
            if fi >= self.dataBuffer.dataSource.getNumSlices():
                #drop frames which run over the end of our data
                bgi.remove(fi)
            else:
                self.curBG[:] = (self.curBG + self.dataBuffer.getSlice(fi))[:]

        self.curFrames = bgi

        return self.curBG/len(bgi)

        

class fitTask(taskDef.Task):
    def __init__(self, dataSourceID, index, threshold, metadata, fitModule, dataSourceModule='HDFDataSource', bgindices = [], SNThreshold = False, driftEstInd=[], calObjThresh=200):
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
        self.dataSourceID = dataSourceID
        self.index = index

        self.bgindices = bgindices

        self.md = metadata
        
        self.fitModule = fitModule
        self.dataSourceModule = dataSourceModule
        self.SNThreshold = SNThreshold
        self.driftEstInd = driftEstInd
        self.driftEst = not len(self.driftEstInd) == 0
        self.calObjThresh = calObjThresh
                 
        self.bufferLen = 50 #12
        if self.driftEst: 
            #increase the buffer length as we're going to look forward as well
            self.bufferLen = 50 #17


    def __call__(self, gui=False, taskQueue=None):
        global dBuffer, bBuffer, dataSourceID
                        
        fitMod = __import__('PYME.Analysis.FitFactories.' + self.fitModule, fromlist=['PYME', 'Analysis','FitFactories']) #import our fitting module

        DataSource = __import__('PYME.Analysis.DataSources.' + self.dataSourceModule, fromlist=['DataSource']).DataSource #import our data source

        #read the data
        if not dataSourceID == self.dataSourceID: #avoid unnecessary opening and closing of 
            dBuffer = dataBuffer(DataSource(self.dataSourceID, taskQueue), self.bufferLen)
            bBuffer = backgroundBuffer(dBuffer)
            dataSourceID = self.dataSourceID
        
        self.data = dBuffer.getSlice(self.index)

        #when camera buffer overflows, empty pictures are produced - deal with these here
        if self.data.max() == 0:
            return fitResult(self, [])
        
        #squash 4th dimension
        self.data = self.data.reshape((self.data.shape[0], self.data.shape[1],1))

        if self.fitModule in splitterFitModules:
#            if (self.md.getEntry('Camera.ROIHeight') + 1 + 2*(self.md.getEntry('Camera.ROIPosY')-1)) == 512:
            #was setup correctly for the splitter
            g = self.data[:, :(self.data.shape[1]/2)]
            r = self.data[:, (self.data.shape[1]/2):]
            r = np.fliplr(r)
#            else:
#                #someone bodged something
#                print 'Warning - splitter incorrectly set - '

        #calculate background
        self.bg = 0
        if not len(self.bgindices) == 0:
#            self.bg = numpy.zeros(self.data.shape, 'f')
#            for bgi in self.bgindices:
#                bs = dBuffer.getSlice(bgi).astype('f')
#                bs = bs.reshape(self.data.shape)
#                self.bg = self.bg + bs
#
#            self.bg *= 1.0/len(self.bgindices)
            self.bg = bBuffer.getBackground(self.bgindices).reshape(self.data.shape)

        if self.fitModule == 'ConfocCOIR': #special case - no object finding
            md = copy.copy(self.md)
            md.tIndex = self.index
            self.res = fitMod.ConfocCOI(self.data, md, background = self.bg)
            return fitResult(self, self.res, [])

        #Find objects
        bgd = self.data.astype('f') - self.bg

#        if self.fitModule in splitterFitModules:
##            g_ = bgd[:, :(self.data.shape[1]/2)]
##            r_ = bgd[:, (self.data.shape[1]/2):]
##            r_ = np.fliplr(r_)
##
##            bgd = g_ + r_
#
#            self.data = numpy.concatenate((g.reshape(g.shape[0], -1, 1), r.reshape(g.shape[0], -1, 1)),2)

        if not 'PSFFile' in self.md.getEntryNames():
            self.ofd = ofind.ObjectIdentifier(bgd * (bgd > 0))
        else: #if we've got a PSF then use cross-correlation object identificatio      
            self.ofd = ofind_xcorr.ObjectIdentifier(bgd * (bgd > 0), self.md.getEntry('PSFFile'), 7, 5e-2)
            

        if 'Analysis.DebounceRadius' in self.md.getEntryNames():
            debounce = self.md.getEntry('Analysis.DebounceRadius')
        else:
            debounce = 5
        self.ofd.FindObjects(self.calcThreshold(),0, splitter=(self.fitModule in splitterFitModules), debounceRadius=debounce)

        if self.fitModule in splitterFitModules:
            self.data = numpy.concatenate((g.reshape(g.shape[0], -1, 1), r.reshape(g.shape[0], -1, 1)),2)

            if not len(self.bgindices) == 0:
                g_ = self.bg[:, :(self.bg.shape[1]/2)]
                r_ = self.bg[:, (self.bg.shape[1]/2):]
                r_ = np.fliplr(r_)

                #print g.shape, r.shape, g_.shape, r_.shape

                self.bg = numpy.concatenate((g_.reshape(g.shape[0], -1, 1), r_.reshape(g.shape[0], -1, 1)),2)

        if self.driftEst: #do the same for objects which are on the whole time
             self.mIm = numpy.ones(self.data.shape, 'f')
             for dri in self.driftEstInd:
                 bs = dBuffer.getSlice(dri)
                 bs = bs.reshape(self.data.shape)
                 #multiply images together, thus favouring images which are on over multiple frames
                 self.mIm = self.mIm*numpy.maximum(bs.astype('f') - numpy.median(bs.ravel()), 1)

             #self.mIm = numpy.absolute(self.mIm)
             if not 'PSFFile' in self.md.getEntryNames():
                 self.ofdDr = ofind.ObjectIdentifier(self.mIm)
             else:
                 self.ofdDr = ofind_xcorr.ObjectIdentifier(self.mIm, self.md.getEntry('PSFFile'), 7, 3e-2)
             thres = self.calObjThresh**10
             self.ofdDr.FindObjects(thres,0)
             while len(self.ofdDr) >= 10: #just go for the brightest ones
                 thres = thres * max(2, len(self.ofdDr)/5)
                 self.ofdDr.FindObjects(thres,0)
                 
        
        #If we're running under a gui - display found objects
        if gui:
            import pylab
            cm = pylab.cm
            pylab.clf()
            pylab.imshow(self.ofd.filteredData.T, cmap=pylab.cm.hot, hold=False)
            pylab.plot([p.x for p in self.ofd], [p.y for p in self.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)

            if self.fitModule in splitterFitModules:
                pylab.plot([p.x for p in self.ofd], [bgd.shape[1] - p.y for p in self.ofd], 'o', mew=2, mec='r', mfc='none', ms=9)


            if self.driftEst:
                 pylab.plot([p.x for p in self.ofdDr], [p.y for p in self.ofdDr], 'o', mew=2, mec='b', mfc='none', ms=9)
            #axis('image')
            #gca().set_ylim([255,0])
            pylab.colorbar()
            pylab.show()

        #Create a fit 'factory'
        md = copy.copy(self.md)
        md.tIndex = self.index

        #if self.fitModule == 'LatGaussFitFRTC'  or self.fitModule == 'BiplaneFitR':
        #    fitFac = fitMod.FitFactory(numpy.concatenate((g.reshape(g.shape[0], -1, 1), r.reshape(g.shape[0], -1, 1)),2), md)
        #else:
        fitFac = fitMod.FitFactory(self.data, md, background = self.bg)

        #print 'Have Fit Factory'
        
        #perform fit for each point that we detected
        if 'FitResultsDType' in dir(fitMod):
            self.res = numpy.empty(len(self.ofd), fitMod.FitResultsDType)
            if 'Analysis.ROISize' in md.getEntryNames():
                rs = md.getEntry('Analysis.ROISize')
                for i in range(len(self.ofd)):
                    p = self.ofd[i]
                    self.res[i] = fitFac.FromPoint(p.x, p.y, roiHalfSize=rs)
            else:
                for i in range(len(self.ofd)):
                    p = self.ofd[i]
                    self.res[i] = fitFac.FromPoint(p.x, p.y)
        else:
            self.res  = [fitFac.FromPoint(p.x, p.y) for p in self.ofd]

        self.drRes = []
        if self.driftEst:
            nToFit = min(10,len(self.ofdDr)) #don't bother fitting lots of calibration objects 
            if 'FitResultsDType' in dir(fitMod):
                self.drRes = numpy.empty(nToFit, fitMod.FitResultsDType)
                for i in range(nToFit):
                    p = self.ofdDr[i]
                    self.drRes[i] = fitFac.FromPoint(p.x, p.y)
            else:
                self.drRes  = [fitFac.FromPoint(p.x, p.y) for p in self.ofd[:nToFit]]    

        #print fitResult(self, self.res, self.drRes)
        return fitResult(self, self.res, self.drRes)

    def calcThreshold(self):
        if self.SNThreshold:
            fudgeFactor = 1 #to account for the fact that the blurring etc... in ofind doesn't preserve intensities - at the moment completely arbitrary so a threshold setting of 1 results in reasonable detection.
            return (numpy.sqrt(self.md.Camera.ReadNoise**2 + numpy.maximum(self.md.Camera.ElectronsPerCount*(self.md.Camera.NoiseFactor**2)*(self.data.astype('f').sum(2) - self.md.Camera.ADOffset)*self.md.Camera.TrueEMGain, 1))/self.md.Camera.ElectronsPerCount)*fudgeFactor*self.threshold
        else:
            return self.threshold
