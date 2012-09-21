#!/usr/bin/python

##################
# HDFTaskQueue.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import tables
from taskQueue import *
from PYME.Analysis.remFitBuf import fitTask

from PYME.Analysis import MetaData
from PYME.Acquire import MetaDataHandler

import os
import sys
import numpy as np

from PYME.FileUtils.nameUtils import genResultFileName
from PYME.ParallelTasks.relativeFiles import getFullFilename

CHUNKSIZE = 50
MAXCHUNKSIZE = 100 #allow chunk size to be improved to allow better caching

#def genDataFilename(name):
#	fn = os.g

#global lock for all calls into HDF library - on linux you seem to be able to
#get away with locking separately for each file (or maybe not locking at all -
#is linux hdf5 threadsafe?)

tablesLock = threading.Lock()

#class TaskWatcher(threading.Thread):
#	def __init__(self, tQueue):
#		threading.Thread.__init__(self)
#		self.tQueue = tQueue
#		self.alive = True
#
#	def run(self):
#		while self.alive:
#			self.tQueue.checkTimeouts()
#			#print '%d tasks in queue' % self.tQueue.getNumberOpenTasks()
#			#try:
#                        #        mProfile.report()
#                        #finally:
#                        #        pass
#                        print mProfile.files
#			time.sleep(10)
#
#tw = TaskWatcher(tq)
#    #tw.start()

bufferMisses = 0

class dataBuffer: #buffer our io to avoid decompressing multiple times
    def __init__(self,dataSource, bLen = 1000):
        self.bLen = bLen
        self.buffer = {} #delay creation until we know the dtype
        #self.buffer = numpy.zeros((bLen,) + dataSource.getSliceShape(), 'uint16')
        #self.insertAt = 0
        #self.bufferedSlices = -1*numpy.ones((bLen,), 'i')
        self.dataSource = dataSource
        
    def getSlice(self,ind):
        global bufferMisses
        #print self.bufferedSlices, self.insertAt, ind
        #return self.dataSource.getSlice(ind)
        if ind in self.bufferedSlices.keys(): #return from buffer
            #print int(numpy.where(self.bufferedSlices == ind)[0])
            return self.buffer[ind]
        else: #get from our data source and store in buffer
            sl = self.dataSource[ind,:,:]
            
            self.buffer[sl] = sl

            bufferMisses += 1
            
            #if bufferMisses % 10 == 0:
            #    print nTasksProcessed, bufferMisses

            return sl

class myLock:
    def __init__(self):
        self.lock = threading.Lock()

    def acquire(self):
        self.lock.acquire()
        fr = sys._getframe()
        print 'Acquired Lock - ' + fr.f_back.f_code.co_name + ' %d' % fr.f_back.f_lineno

    def release(self):
        print 'Released Lock'
        self.lock.release()

#tablesLock = myLock()

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)


class HDFResultsTaskQueue(TaskQueue):
    '''Task queue which saves it's results to a HDF file'''
    def __init__(self, name, resultsFilename, initialTasks=[], onEmpty = doNix, fTaskToPop = popZero):
        if resultsFilename == None:
            resultsFilename = genResultFileName(name)

        if os.path.exists(resultsFilename): #bail if output file already exists
            raise RuntimeError('Output file already exists: ' + resultsFilename)

        TaskQueue.__init__(self, name, initialTasks, onEmpty, fTaskToPop)
        self.resultsFilename = resultsFilename

        self.numClosedTasks = 0

        self.h5ResultsFile = tables.openFile(self.resultsFilename, 'w')

        self.prepResultsFile()

        #self.fileResultsLock = threading.Lock()
        self.fileResultsLock = tablesLock

        self.resultsMDH = MetaDataHandler.HDFMDHandler(self.h5ResultsFile)

        self.resultsEvents = self.h5ResultsFile.createTable(self.h5ResultsFile.root, 'Events', SpoolEvent,filters=tables.Filters(complevel=5, shuffle=True))
        
        self.haveResultsTable = False

    def prepResultsFile(self):
        pass
	

    def getCompletedTask(self):
        return None

    def setQueueMetaData(self, fieldName, value):
        self.fileResultsLock.acquire()
        self.resultsMDH.setEntry(fieldName, value)
        self.fileResultsLock.release()

    def getQueueMetaData(self, fieldName):
        res  = None
        self.fileResultsLock.acquire()
        try:
            res = self.resultsMDH.getEntry(fieldName)
        finally:
            self.fileResultsLock.release()
        return res

    def addQueueEvents(self, events):
        self.fileResultsLock.acquire()
        try:
            self.resultsEvents.append(events)
        finally:
            self.fileResultsLock.release()


    def getQueueMetaDataKeys(self):
        self.fileResultsLock.acquire()
        res = self.resultsMDH.getEntryNames()
        self.fileResultsLock.release()
        return res

    def getNumberTasksCompleted(self):
		return self.numClosedTasks

    def purge(self):
		self.openTasks = []
		self.numClosedTasks = 0
		self.tasksInProgress = []

    def cleanup(self):
        #self.h5DataFile.close()
        self.h5ResultsFile.close()

    def fileResult(self, res):
        #print res, res.results, res.driftResults, self.h5ResultsFile
        if res == None:
            print 'res == None'
            
        if res.results == [] and res.driftResults == []: #if we had a dud frame
            return

        self.fileResultsLock.acquire() #get a lock
            
        if not len(res.results) == 0:
            #print res.results, res.results == []
            if not self.haveResultsTable: # self.h5ResultsFile.__contains__('/FitResults'):
                self.h5ResultsFile.createTable(self.h5ResultsFile.root, 'FitResults', res.results, filters=tables.Filters(complevel=5, shuffle=True), expectedrows=500000)
                self.haveResultsTable = True
            else:
                self.h5ResultsFile.root.FitResults.append(res.results)

        if not len(res.driftResults) == 0:
            if not self.h5ResultsFile.__contains__('/DriftResults'):
                self.h5ResultsFile.createTable(self.h5ResultsFile.root, 'DriftResults', res.driftResults, filters=tables.Filters(complevel=5, shuffle=True), expectedrows=500000)
            else:
                self.h5ResultsFile.root.DriftResults.append(res.driftResults)

        #self.h5ResultsFile.flush()

        self.fileResultsLock.release() #release lock

        self.numClosedTasks += 1
        
    def checkTimeouts(self):
        self.inProgressLock.acquire()
        curTime = time.clock()
        for it in self.tasksInProgress:
            if 'workerTimeout' in dir(it):
                if curTime > it.workerTimeout:
                    self.openTasks.append(it.index)
                    self.tasksInProgress.remove(it)

        self.inProgressLock.release()
        
        self.fileResultsLock.acquire() #get a lock
        
        self.h5ResultsFile.flush()

        self.fileResultsLock.release() #release lock
        
        
    def getQueueData(self, fieldName, *args):
        '''Get data, defined by fieldName and potntially additional arguments,  ascociated with queue'''
        if fieldName == 'FitResults':
            startingAt, = args
            self.fileResultsLock.acquire()
            if self.h5ResultsFile.__contains__('/FitResults'):
                res = self.h5ResultsFile.root.FitResults[startingAt:]
            else:
                res = []
            self.fileResultsLock.release()
            return res
        else:
            return None



class HDFTaskQueue(HDFResultsTaskQueue):
    ''' task queue which, when initialised with an hdf image filename, automatically generates tasks - should also (eventually) include support for dynamically adding to data file for on the fly analysis'''
    def __init__(self, name, dataFilename = None, resultsFilename=None, onEmpty = doNix, fTaskToPop = popZero, startAt = 'guestimate', frameSize=(-1,-1), complevel=6, complib='zlib'):
        if dataFilename == None:
           self.dataFilename = genDataFilename(name)
        else:
            self.dataFilename = dataFilename

        if resultsFilename == None:
            resultsFilename = genResultFileName(self.dataFilename)
        else:
            resultsFilename = resultsFilename
		
        ffn = getFullFilename(self.dataFilename)

        self.acceptNewTasks = False
        self.releaseNewTasks = False

        self.postTaskBuffer = []

        initialTasks = []


        if os.path.exists(ffn): #file already exists - read from it
            self.h5DataFile = tables.openFile(ffn, 'r')
            #self.metaData = MetaData.genMetaDataFromHDF(self.h5DataFile)
            self.dataMDH = MetaDataHandler.NestedClassMDHandler(MetaDataHandler.HDFMDHandler(self.h5DataFile))
            #self.dataMDH.mergeEntriesFrom(MetaData.TIRFDefault)
            self.imageData = self.h5DataFile.root.ImageData


            if startAt == 'guestimate': #calculate a suitable starting value
                tLon = self.dataMDH.EstimatedLaserOnFrameNo
                if tLon == 0:
                    startAt = 0
                else:
                    startAt = tLon + 10

            if startAt == 'notYet':
                initialTasks = []
            else:
                initialTasks = list(range(startAt, self.h5DataFile.root.ImageData.shape[0]))

            self.imNum = len(self.imageData)

        else: #make ourselves a new file
            self.h5DataFile = tables.openFile(ffn, 'w')
            filt = tables.Filters(complevel, complib, shuffle=True)

            self.imageData = self.h5DataFile.createEArray(self.h5DataFile.root, 'ImageData', tables.UInt16Atom(), (0,)+tuple(frameSize), filters=filt)
            self.events = self.h5DataFile.createTable(self.h5DataFile.root, 'Events', SpoolEvent,filters=filt)
            self.imNum=0
            self.acceptNewTasks = True

            self.dataMDH = MetaDataHandler.HDFMDHandler(self.h5DataFile)
            self.dataMDH.mergeEntriesFrom(MetaData.TIRFDefault)


        HDFResultsTaskQueue.__init__(self, name, resultsFilename, initialTasks, onEmpty, fTaskToPop)

        
        self.resultsMDH.copyEntriesFrom(self.dataMDH)

        #copy events to results file
        if len (self.h5DataFile.root.Events) > 0:
            self.resultsEvents.append(self.h5DataFile.root.Events[:])

        self.metaData = None #MetaDataHandler.NestedClassMDHandler(self.resultsMDH)
        self.metaDataStale = True
        self.queueID = name

        self.numSlices = self.imageData.shape[0]

        #self.dataFileLock = threading.Lock()
        self.dataFileLock = tablesLock
        #self.getTaskLock = threading.Lock()
        self.lastTaskTime = 0
                
    def prepResultsFile(self):
        pass

    def postTask(self,task):
        #self.postTaskBuffer = []

        #self.openTasks.append(task)
        #print 'posting tasks not implemented yet'
        if self.acceptNewTasks:
            self.dataFileLock.acquire()
            self.imageData.append(task)
            self.h5DataFile.flush()
            self.numSlices = self.imageData.shape[0]
            self.dataFileLock.release()

            if self.releaseNewTasks:
                self.openTasks.append(self.imNum)
            self.imNum += 1
        else:
            print "can't post new tasks"
			

    def postTasks(self,tasks):
        #self.openTasks += tasks
        if self.acceptNewTasks:
            self.dataFileLock.acquire()
            for task in tasks:
                self.imageData.append(task)
                #self.h5DataFile.flush()
                #self.dataFileLock.release()

                if self.releaseNewTasks:
                    self.openTasks.append(self.imNum)
                self.imNum += 1

            self.h5DataFile.flush()
            self.numSlices = self.imageData.shape[0]
            self.dataFileLock.release()
        else:
            print "can't post new tasks"
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
        #print 'Task requested'
        #self.getTaskLock.acquire()
        while len(self.openTasks) < 1:
            time.sleep(0.01)

        if self.metaDataStale:
            self.dataFileLock.acquire()
            self.metaData = MetaDataHandler.NestedClassMDHandler(self.resultsMDH)
            self.metaDataStale = False
            self.dataFileLock.release()

            #patch up old data which doesn't have BGRange in metadata
            if not 'Analysis.BGRange' in self.metaData.getEntryNames():
                if 'Analysis.NumBGFrames' in self.metaData.getEntryNames():
                    nBGFrames = self.metaData.Analysis.NumBGFrames
                else:
                    nBGFrames = 10

                self.metaData.setEntry('Analysis.BGRange', (-nBGFrames, 0))
        
        
        taskNum = self.openTasks.pop(self.fTaskToPop(workerN, NWorkers, len(self.openTasks)))

        #if 'Analysis.BGRange' in self.metaData.getEntryNames():
        bgi = range(max(taskNum + self.metaData.Analysis.BGRange[0],self.metaData.EstimatedLaserOnFrameNo), max(taskNum + self.metaData.Analysis.BGRange[1],self.metaData.EstimatedLaserOnFrameNo))
        #elif 'Analysis.NumBGFrames' in self.metaData.getEntryNames():
        #    bgi = range(max(taskNum - self.metaData.Analysis.NumBGFrames,self.metaData.EstimatedLaserOnFrameNo), taskNum)
        #else:
        #    bgi = range(max(taskNum - 10,self.metaData.EstimatedLaserOnFrameNo), taskNum)
        
        task = fitTask(self.queueID, taskNum, self.metaData.Analysis.DetectionThreshold, self.metaData, self.metaData.Analysis.FitModule, 'TQDataSource', bgindices = bgi, SNThreshold = True)
        
        task.queueID = self.queueID
        task.initializeWorkerTimeout(time.clock())
        self.inProgressLock.acquire()
        self.tasksInProgress.append(task)
        self.inProgressLock.release()
        #self.getTaskLock.release()

        self.lastTaskTime = time.time()

        return task

    def getTasks(self, workerN = 0, NWorkers = 1):
        """get task from front of list, blocks"""
        #print 'Task requested'
        #self.getTaskLock.acquire()
        while len(self.openTasks) < 1:
            time.sleep(0.01)

        if self.metaDataStale:
            self.dataFileLock.acquire()
            self.metaData = MetaDataHandler.NestedClassMDHandler(self.resultsMDH)
            self.metaDataStale = False
            self.dataFileLock.release()

            if not 'Analysis.BGRange' in self.metaData.getEntryNames():
                if 'Analysis.NumBGFrames' in self.metaData.getEntryNames():
                    nBGFrames = self.metaData.Analysis.NumBGFrames
                else:
                    nBGFrames = 10

                self.metaData.setEntry('Analysis.BGRange', (-nBGFrames, 0))


        tasks = []

        for i in range(min(max(CHUNKSIZE, min(MAXCHUNKSIZE, len(self.openTasks))),len(self.openTasks))):

            taskNum = self.openTasks.pop(self.fTaskToPop(workerN, NWorkers, len(self.openTasks)))

            #if 'Analysis.BGRange' in self.metaData.getEntryNames():
            bgi = range(max(taskNum + self.metaData.Analysis.BGRange[0],self.metaData.EstimatedLaserOnFrameNo), max(taskNum + self.metaData.Analysis.BGRange[1],self.metaData.EstimatedLaserOnFrameNo))
            #elif 'Analysis.NumBGFrames' in self.metaData.getEntryNames():
            #    bgi = range(max(taskNum - self.metaData.Analysis.NumBGFrames,self.metaData.EstimatedLaserOnFrameNo), taskNum)
            #else:
            #    bgi = range(max(taskNum - 10,self.metaData.EstimatedLaserOnFrameNo), taskNum)

            task = fitTask(self.queueID, taskNum, self.metaData.Analysis.DetectionThreshold, self.metaData, self.metaData.Analysis.FitModule, 'TQDataSource', bgindices =bgi, SNThreshold = True)

            task.queueID = self.queueID
            task.initializeWorkerTimeout(time.clock())
            self.inProgressLock.acquire()
            self.tasksInProgress.append(task)
            self.inProgressLock.release()
            #self.getTaskLock.release()

            tasks.append(task)

        self.lastTaskTime = time.time()

        return tasks

	
    


    def cleanup(self):
        self.h5DataFile.close()
        self.h5ResultsFile.close()

    def setQueueMetaData(self, fieldName, value):
        self.dataFileLock.acquire()
        self.dataMDH.setEntry(fieldName, value)
        self.dataFileLock.release()
        HDFResultsTaskQueue.setQueueMetaData(self, fieldName, value)
        self.metaDataStale = True
        
    def getQueueData(self, fieldName, *args):
        '''Get data, defined by fieldName and potntially additional arguments,  ascociated with queue'''
        if fieldName == 'ImageShape':
            self.dataFileLock.acquire()
            res = self.h5DataFile.root.ImageData.shape[1:]
            self.dataFileLock.release()
            return res
        elif fieldName == 'ImageData':
            sliceNum, = args
            self.dataFileLock.acquire()
            res = self.h5DataFile.root.ImageData[sliceNum, :,:]
            self.dataFileLock.release()
            return res
        elif fieldName == 'NumSlices':
            #self.dataFileLock.acquire()
            #res = self.h5DataFile.root.ImageData.shape[0]
            #self.dataFileLock.release()
            #print res, self.numSlices
            #return res
            return self.numSlices
        elif fieldName == 'Events':
            self.dataFileLock.acquire()
            res = self.h5DataFile.root.Events[:]
            self.dataFileLock.release()
            return res
        elif fieldName == 'PSF':
            from PYME.ParallelTasks.relativeFiles import getFullExistingFilename
            res = None
            #self.dataFileLock.acquire()
            #try:
                #res = self.h5DataFile.root.PSFData[:]
            #finally:
            #    self.dataFileLock.release()
            #try:
            modName = self.resultsMDH.getEntry('PSFFile')
            mf = open(getFullExistingFilename(modName), 'rb')
            res = np.load(mf)
            mf.close()
            #except:
                #pass

            return res
        else:
            return HDFResultsTaskQueue.getQueueData(self, fieldName, *args)

    def logQueueEvent(self, event):
        eventName, eventDescr, evtTime = event
        ev = self.events.row

        ev['EventName'] = eventName
        ev['EventDescr'] = eventDescr
        ev['Time'] = evtTime

        self.dataFileLock.acquire()
        ev.append()
        self.events.flush()
        self.dataFileLock.release()

        ev = self.resultsEvents.row

        ev['EventName'] = eventName
        ev['EventDescr'] = eventDescr
        ev['Time'] = evtTime

        self.fileResultsLock.acquire()
        #print len(self.events)
        ev.append()
        self.resultsEvents.flush()
        self.fileResultsLock.release()

        #self.dataFileLock.release()


    def releaseTasks(self, startingAt = 0):
        self.openTasks += range(startingAt, self.imNum)
        self.releaseNewTasks = True
