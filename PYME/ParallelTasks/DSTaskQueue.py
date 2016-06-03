# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 23:59:45 2015

@author: david
"""
import time
import numpy as np
import threading
from .HDFTaskQueue import HDFResultsTaskQueue, doNix, popZero, CHUNKSIZE, MAXCHUNKSIZE, fitTask

class DSTaskQueue(HDFResultsTaskQueue):
    ''' task queue which, when initialised with an hdf image filename, automatically generates tasks - should also (eventually) include support for dynamically adding to data file for on the fly analysis'''
    def __init__(self, name, mdh, dataSourceModule, dataSourceID, resultsFilename=None, onEmpty = doNix, fTaskToPop = popZero, startAt = 10):

        self.acceptNewTasks = False
        self.releaseNewTasks = False

        initialTasks = []

        HDFResultsTaskQueue.__init__(self, name, resultsFilename, initialTasks, onEmpty, fTaskToPop)
        
        self.resultsMDH.copyEntriesFrom(mdh)
        self.metaData.copyEntriesFrom(self.resultsMDH)
        
        self.queueID = name
        
        self.dataSourceID = dataSourceID
        
        #load data source
        self.dataSourceModule = dataSourceModule
        DataSource = __import__('PYME.IO.DataSources.' + dataSourceModule, fromlist=['PYME', 'io', 'DataSources']).DataSource #import our data source
        self.ds = DataSource(self.dataSourceID)
        
        if dataSourceModule == 'ClusterPZFDataSource':
            #fit modules will be able to directly access the data
            self.serveData = False
        else:
            self.serveData = True
            
            #the data source we pass to our workers should be this queue
            self.dataSourceID = self.queueID
            self.dataSourceModule = 'TQDataSource'

        self.openTasks = []
        self.frameNum = startAt
       
        self.lastTaskTime = 0
        
        self.doPoll = True
        
        self.pollT = threading.Thread(target=self._updatePoll)
        self.pollT.start()
                

    def postTask(self,task):
        print("can't post new tasks")
			

    def postTasks(self,tasks):
        print("can't post new tasks")
        #print 'posting tasks not implemented yet'

    def getNumberOpenTasks(self, exact=True):
        #when doing real time analysis we might not want to tasks out straight
        #away, in order that our caches still work
        nOpen = len(self.openTasks)

        #Answer truthfully if we are being asked for the exact number of tasks, 
        #we have enough tasks to give a full chunk, or if it was a while since 
        #we last gave out any tasks (necessary to make sure all tasks get 
        #processed at the end of the run.
        if exact or nOpen > CHUNKSIZE or (time.time() - self.lastTaskTime) > 2:
            return nOpen
        else: #otherwise lie and make the workers wait
            return 0

    def getTask(self, workerN = 0, NWorkers = 1):
        """get task from front of list, blocks"""

        while len(self.openTasks) < 1:
            time.sleep(0.01)
            
            #patch up old data which doesn't have BGRange in metadata
        if not 'Analysis.BGRange' in self.metaData.getEntryNames():
            nBGFrames = self.metaData.getOrDefault('Analysis.NumBGFrames', 10)
            self.metaData.setEntry('Analysis.BGRange', (-nBGFrames, 0))
        
        
        taskNum = self.openTasks.pop(self.fTaskToPop(workerN, NWorkers, len(self.openTasks)))

        bgi = range(max(taskNum + self.metaData.Analysis.BGRange[0],self.metaData.EstimatedLaserOnFrameNo), max(taskNum + self.metaData.Analysis.BGRange[1],self.metaData.EstimatedLaserOnFrameNo))
        

        print self.dataSourceID, self.dataSourceModule        
        task = fitTask(self.dataSourceID, taskNum, self.metaData['Analysis.DetectionThreshold'], self.metaData, self.metaData['Analysis.FitModule'], bgindices =bgi, SNThreshold = True, dataSourceModule = self.dataSourceModule)
        
        task.queueID = self.queueID
        task.initializeWorkerTimeout(time.clock())
        with self.inProgressLock:
            self.tasksInProgress.append(task)

        self.lastTaskTime = time.time()

        return task

    def getTasks(self, workerN = 0, NWorkers = 1):
        """get task from front of list, blocks"""
        while len(self.openTasks) < 1:
            time.sleep(0.01)

        if not 'Analysis.BGRange' in self.metaData.getEntryNames():
            nBGFrames = self.metaData.getOrDefault('Analysis.NumBGFrames', 10)
            self.metaData.setEntry('Analysis.BGRange', (-nBGFrames, 0))

        tasks = []
        
        if not 'Analysis.ChunkSize' in self.metaData.getEntryNames():
            cs = min(max(CHUNKSIZE, min(MAXCHUNKSIZE, len(self.openTasks))),len(self.openTasks))
        else:
            cs = min(self.metaData['Analysis.ChunkSize'], len(self.openTasks))

        for i in range(cs):
            taskNum = self.openTasks.pop(self.fTaskToPop(workerN, NWorkers, len(self.openTasks)))

            bgi = range(max(taskNum + self.metaData.Analysis.BGRange[0],self.metaData.EstimatedLaserOnFrameNo), max(taskNum + self.metaData.Analysis.BGRange[1],self.metaData.EstimatedLaserOnFrameNo))
 
            task = fitTask(self.dataSourceID, taskNum, self.metaData['Analysis.DetectionThreshold'], self.metaData, self.metaData['Analysis.FitModule'], bgindices =bgi, SNThreshold = True, dataSourceModule = self.dataSourceModule)
            
            task.queueID = self.queueID
            task.initializeWorkerTimeout(time.clock())
            with self.inProgressLock:
                self.tasksInProgress.append(task)            

            tasks.append(task)

        self.lastTaskTime = time.time()

        return tasks

        
    def getQueueData(self, fieldName, *args):
        '''Get data, defined by fieldName and potntially additional arguments,  ascociated with queue'''
        if fieldName == 'ImageShape':
            with self.dataFileLock.rlock:
                res = self.h5DataFile.root.ImageData.shape[1:]           
            return res
        elif fieldName == 'ImageData':
            sliceNum, = args
            res = self.ds.getSlice(sliceNum)            
            return res
        elif fieldName == 'NumSlices':
            return self.getNumSlices()
        elif fieldName == 'Events':            
            return self.ds.getEvents()
        else:
            return HDFResultsTaskQueue.getQueueData(self, fieldName, *args)
    
    def _updatePoll(self):
        while (self.doPoll == True):
            self._updateTasks()
            time.sleep(1)
    
    def _updateTasks(self):
        nfn = self.ds.getNumSlices()
        if nfn > self.frameNum:
            self.openTasks += range(self.frameNum, nfn+1)
            self.frameNum = nfn+1
        
        ev = self.ds.getEvents()
        if len(ev) > self.h5ResultsFile.root.Events.shape[0]:        
            self.addQueueEvents(ev[self.h5ResultsFile.root.Events.shape[0]:])

    def releaseTasks(self, startingAt = 0):
        self.openTasks += range(startingAt, self.imNum)
        self.releaseNewTasks = True
        
    def cleanup(self):
        self.doPoll = False
        HDFResultsTaskQueue.cleanup(self)
