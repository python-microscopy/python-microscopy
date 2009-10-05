#!/usr/bin/python

##################
# remFitPSFFromFilename.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import taskDef
import ofind
import LatPSFFit
import Pyro.core
import matplotlib
import read_kdf
import numpy
import copy_reg

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)

#matplotlib.interactive(False)
from pylab import *

class fitResult:
    def __init__(self, task, results):
        self.taskID = task.taskID
        self.filename = task.filename
        self.seriesName = task.seriesName
        self.results = results

class fitTask(taskDef.Task):
    def __init__(self, filename, seriesName, threshold, metadata, bgfiles = []):
        taskDef.Task.__init__(self)
        self.threshold = threshold
        self.filename = filename
        self.seriesName = seriesName

        self.bgfiles = bgfiles

	self.md = metadata

        #if len(data.shape) == 2:
        #    self.data = data.reshape((data.shape[0], data.shape[1], 1))
        #else:
        #    self.data = data
        

    #def setReturnAdress(self, retAddress):
    #    self.retAddress = retAdress

    def __call__(self, gui=False):
        self.data = read_kdf.ReadKdfData(self.filename)
        self.data = self.data.reshape((self.data.shape[0], self.data.shape[1], 1))
        #print self.data.shape
        self.bg = 0
        if not len(self.bgfiles) == 0:
            self.bg = numpy.zeros(self.data.shape, 'f')
            for bgfn in self.bgfiles:
                self.bg += read_kdf.ReadKdfData(bgfn).reshape(self.data.shape)

            self.bg /= len(self.bgfiles)

        self.ofd = ofind.ObjectIdentifier(self.data - self.bg)
        self.ofd.FindObjects(self.threshold,0)
        
        #print globals()
        if gui:
            #print 'hello'
            clf()
            imshow(self.ofd.filteredData, cmap=cm.hot, hold=False)
            plot([p.y for p in self.ofd], [p.x for p in self.ofd], 'o', mew=1, mec='g', mfc=None)
            axis('image')
            gca().set_ylim([255,0])
            colorbar()
            show()

        fitFac = LatPSFFit.PSFFitFactory(self.data, self.md)
        self.res  = [fitFac.FromPoint(round(p.x), round(p.y)) for p in self.ofd]
        #print self.res
        #def getRes(self):
        #return fitResult(self, [(r.fitResults, r.fitErr) for r in self.res] )
        return fitResult(self, self.res )
