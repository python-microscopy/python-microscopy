from PYME.ParallelTasks import taskDef
import ofind
import numpy

dBuffer = None
queueID = None

bufferMisses = 0

from pylab import *

def tqPopFcn(workerN, NWorkers, NTasks):
    return workerN * NTasks/NWorkers #let each task work on its own chunk of data ->
    
class fitResult(taskDef.TaskResult):
    def __init__(self, task, results):
        taskDef.TaskResult.__init__(self, task)
        self.index = task.index
        self.results = results

class remoteDataSource:
    def __init__(self, taskQueue, queueName):
        self.taskQueue = taskQueue
        self.queueName = queueName

    def getSlice(self, ind):
        return self.taskQueue.getQueueData(self.QueueName, 'ImageData',ind)

    def getSliceShape(self):
        return self.taskQueue.getQueueData(self.QueueName, 'ImageShape')

class dataBuffer: #buffer our io to avoid decompressing multiple times
    def __init__(self,dataSource, bLen = 12):
        self.bLen = bLen
        self.buffer = numpy.zeros((bLen,) + dataSource.getSliceShape(), 'uint16')
        self.insertAt = 0
        self.bufferedSlices = list(-1*numpy.ones((bLen,), 'i'))
        self.dataSource = dataSource
        
    def getSlice(self,ind):
        global bufferMisses
        if ind in self.bufferedSlices: #return from buffer
            return self.buffer[self.bufferedSlices.index(ind),:,:]
        else: #get from our data source and store in buffer
            sl = self.dataSource.getSlice(ind)
            self.bufferedSlices[self.insertAt] = ind
            self.buffer[self.insertAt, :,:] = sl
            self.insertAt += 1
            self.insertAt %=self.bLen

            bufferMisses += 1
            
            return sl
        

    
        

class fitTask(taskDef.Task):
    def __init__(self, index, threshold, metadata, fitModule, bgindices = [], SNThreshold = False):
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
        #self.filename = filename
        self.index = index

        self.bgindices = bgindices

	self.md = metadata
        
        self.fitModule = fitModule
        self.SNThreshold = SNThreshold


    def __call__(self, gui=False, taskQueue=None):
        global dBuffer, queueID
                        
        fitMod = __import__('PYME.Analysis.FitFactories.' + self.fitModule, fromlist=['PYME', 'Analysis','FitFactories']) #import our fitting module

        #read the data
        if not queueID == self.queueID: #avoid unnecessary opening and closing of 
            dBuffer = dataBuffer(remoteDataSource(taskQueue, self.queueID))
            queueID = self.queueID
        
        self.data = dBuffer.getSlice(self.index)

        #when camera buffer overflows, empty pictures are produced - deal with these here
        if self.data.max() == 0:
            return fitResult(self, [])
        
        #squash 4th dimension
        self.data = self.data.reshape((self.data.shape[0], self.data.shape[1],1))
        
        #calculate background
        self.bg = 0
        if not len(self.bgindices) == 0:
            self.bg = numpy.zeros(self.data.shape, 'f')
            for bgi in self.bgindices:
                bs = dBuffer.getSlice(bgi)
                bs = bs.reshape(self.data.shape)
                self.bg = self.bg + bs.astype('f')

            self.bg *= 1.0/len(self.bgindices)

        #Find objects
        self.ofd = ofind.ObjectIdentifier(self.data.astype('f') - self.bg)
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
        fitFac = fitMod.FitFactory(self.data, self.md)
        
        #perform fit for each point that we detected
        if 'FitResultsDType' in dir(fitMod):
            self.res = numpy.empty(len(self.ofd), fitMod.FitResultsDType)
            for i in range(len(self.ofd)):
                p = self.ofd[i]
                self.res[i] = fitFac.FromPoint(round(p.x), round(p.y))
        else:
            self.res  = [fitFac.FromPoint(round(p.x), round(p.y)) for p in self.ofd]

        return fitResult(self, self.res )

    def calcThreshold(self):
        if self.SNThreshold:
            return numpy.sqrt(numpy.maximum(self.data.mean(2) - self.md.CCD.ADOffset, 1))*self.threshold
        else:
            return self.threshold
