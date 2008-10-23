from PYME.ParallelTasks import taskDef
import ofind
import matplotlib
import numpy
import scipy


#matplotlib.interactive(False)
from pylab import *

def tqPopFcn(workerN, NWorkers, NTasks):
    return workerN * NTasks/NWorkers #let each task work on its own chunk of data ->
    


class fitResult(taskDef.TaskResult):
    def __init__(self, task, results):
        taskDef.TaskResult.__init__(self, task)
        #self.filename = task.filename
        #self.seriesName = task.seriesName
        #self.index = task.index
        self.results = results

        

class fitTask(taskDef.Task):
    def __init__(self, data, threshold, metadata, fitModule,  SNThreshold = False, findOn = 'top', horizOffset = 0):
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
        self.data = data

	self.md = metadata
        
        self.fitModule = fitModule
        self.SNThreshold = SNThreshold

        self.findOn = findOn

        self.horizOffset = horizOffset


    def __call__(self, gui=False):
        fitMod = __import__('PYME.Analysis.FitFactories.' + self.fitModule, fromlist=['PYME', 'Analysis','FitFactories']) #import our fitting module
        

        #when camera buffer overflows, empty pictures are produced - deal with these here
        if self.data.max() == 0:
            return fitResult(self, [])
        
        #squash 4th dimension
        #self.data = self.data.reshape((self.data.shape[0], self.data.shape[1],self.data.shape[2]))
        self.data = self.data.reshape((self.data.shape[0], self.data.shape[1],1))

        top = self.data[:,:self.data.shape[1]/2, :]
        bottom = scipy.fliplr(self.data[:,self.data.shape[1]/2:, :])

        #Find objects
        if self.findOn == 'top':
            self.ofd = ofind.ObjectIdentifier(top.astype('f'))
        else:
            self.ofd = ofind.ObjectIdentifier(bottom.astype('f'))

        self.ofd.FindObjects(self.calcThreshold(),0)
        
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
        fitFacTop = fitMod.FitFactory(top, self.md)
        fitFacBottom = fitMod.FitFactory(bottom, self.md)

        #print self.horizOffset
        
        #perform fit for each point that we detected
        if False: #'FitResultsDType' in dir(fitMod):
            self.res = numpy.empty(len(self.ofd), fitMod.FitResultsDType)
            for i in range(len(self.ofd)):
                p = self.ofd[i]
                self.res[i] = fitFac.FromPoint(round(p.x), round(p.y))
        else:
            self.res  = [[fitFacTop.FromPoint(round(p.x), round(p.y), roiHalfSize=7) for p in self.ofd],[fitFacBottom.FromPoint(min(round(p.x + self.horizOffset), (top.shape[0] - 10)), round(p.y),roiHalfSize=7) for p in self.ofd] ]

        return fitResult(self, self.res )

    def calcThreshold(self):
        if self.SNThreshold:
            return numpy.sqrt(numpy.maximum(self.data.mean(2) - self.md.CCD.ADOffset, 1))*self.threshold
        else:
            return self.threshold
