import tables
from taskQueue import *
from PYME.Analysis.remFitBuf import fitTask

from PYME.Analysis import MetaData
from PYME.Acquire import MetaDataHandler

import os

from PYME.FileUtils.nameUtils import genResultFileName
from PYME.ParallelTasks.relativeFiles import getFullFilename

#def genDataFilename(name):
#	fn = os.g

#global lock for all calls into HDF library - on linux you seem to be able to
#get away with locking separately for each file (or maybe not locking at all -
#is linux hdf5 threadsafe?)
tablesLock = threading.Lock()


class HDFResultsTaskQueue(TaskQueue):
    '''Task queue which saves it's results to a HDF file'''
    def __init__(self, name, resultsFilename, initialTasks=[], onEmpty = doNix, fTaskToPop = popZero):
        if resultsFilename == None:
            resultsFilename = genResultFileName(name)

        if os.path.exists(resultsFilename): #bail if output file already exists
            raise 'Output file already exists'

        TaskQueue.__init__(self, name, initialTasks, onEmpty, fTaskToPop)
        self.resultsFilename = resultsFilename

        self.numClosedTasks = 0

        self.h5ResultsFile = tables.openFile(self.resultsFilename, 'w')

        self.prepResultsFile()

        #self.fileResultsLock = threading.Lock()
        self.fileResultsLock = tablesLock

        self.resultsMDH = MetaDataHandler.HDFMDHandler(self.h5ResultsFile)

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
        if res.results == [] and res.driftResults == []: #if we had a dud frame
            return

        self.fileResultsLock.acquire() #get a lock
            
        if not res.results == []:
            if not self.h5ResultsFile.__contains__('/FitResults'):
                self.h5ResultsFile.createTable(self.h5ResultsFile.root, 'FitResults', res.results, filters=tables.Filters(complevel=5, shuffle=True))
            else:
                self.h5ResultsFile.root.FitResults.append(res.results)

        if not res.driftResults == []:
            if not self.h5ResultsFile.__contains__('/DriftResults'):
                self.h5ResultsFile.createTable(self.h5ResultsFile.root, 'DriftResults', res.driftResults, filters=tables.Filters(complevel=5, shuffle=True))
            else:
                self.h5ResultsFile.root.DriftResults.append(res.driftResults)

        self.h5ResultsFile.flush()

        self.fileResultsLock.release() #release lock

        self.numClosedTasks += 1
        
    def getQueueData(self, fieldName, *args):
        '''Get data, defined by fieldName and potntially additional arguments,  ascociated with queue'''
        if fieldName == 'FitResults':
            startingAt, = args
            self.fileResultsLock.acquire()
            res = self.h5ResultsFile.root.FitResults[startingAt:]
            self.fileResultsLock.release()
            return res
        else:
            return None

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)

class HDFTaskQueue(HDFResultsTaskQueue):
    ''' task queue which, when initialised with an hdf image filename, automatically generated tasks - should also (eventually) include support for dynamically adding to data file for on the fly analysis'''
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

        initialTasks = []


        if os.path.exists(ffn): #file already exists - read from it
            self.h5DataFile = tables.openFile(ffn, 'r')
            self.metaData = MetaData.genMetaDataFromHDF(self.h5DataFile)


            if startAt == 'guestimate': #calculate a suitable starting value
                tLon = self.metaData.EstimatedLaserOnFrameNo
                if tLon == 0:
                    startAt = 0
                else:
                    startAt = tLon + 10

            if startAt == 'notYet':
                initialTasks = []
            else:
                initialTasks = list(range(startAt, self.h5DataFile.root.ImageData.shape[0]))
        else: #make ourselves a new file
            self.h5DataFile = tables.openFile(ffn, 'w')
            filt = tables.Filters(complevel, complib, shuffle=True)

            self.imageData = self.h5DataFile.createEArray(self.h5DataFile.root, 'ImageData', tables.UInt16Atom(), (0,)+tuple(frameSize), filters=filt)
            self.events = self.h5DataFile.createTable(self.h5DataFile.root, 'Events', SpoolEvent,filters=filt)
            self.imNum=0
            self.acceptNewTasks = True




        HDFResultsTaskQueue.__init__(self, name, resultsFilename, initialTasks, onEmpty, fTaskToPop)

        self.dataMDH = MetaDataHandler.HDFMDHandler(self.h5DataFile)
        self.dataMDH.mergeEntriesFrom(MetaData.TIRFDefault)
        self.resultsMDH.copyEntriesFrom(self.dataMDH)
        

        self.metaData = None #MetaDataHandler.NestedClassMDHandler(self.resultsMDH)
        self.metaDataStale = True
        self.queueID = name

        #self.dataFileLock = threading.Lock()
        self.dataFileLock = tablesLock
                
    def prepResultsFile(self):
        pass

    def postTask(self,task):
		#self.openTasks.append(task)
		#print 'posting tasks not implemented yet'
		if self.acceptNewTasks:
            self.dataFileLock.acquire()
            self.imageData.append(task)
            self.h5DataFile.flush()
            self.dataFileLock.release()

			if self.releaseNewTasks:
				self.openTasks.append(self.imNum)
			self.imNum += 1
		else:
			print "can't post new tasks"
			

    def postTasks(self,tasks):
        #self.openTasks += tasks
        print 'posting tasks not implemented yet'

    def getTask(self, workerN = 0, NWorkers = 1):
        """get task from front of list, blocks"""
        #print 'Task requested'
        while len(self.openTasks) < 1:
            time.sleep(0.01)

        if self.metaDataStale:
            self.metaData = MetaDataHandler.NestedClassMDHandler(self.resultsMDH)
            self.metaDataStale = False
        
        
        taskNum = self.openTasks.pop(self.fTaskToPop(workerN, NWorkers, len(self.openTasks)))
        
        task = fitTask(self.queueID, taskNum, self.metaData.Analysis.DetectionThreshold, self.metaData, self.metaData.Analysis.FitModule, 'TQDataSource', bgindices =range(max(taskNum - 10,self.metaData.EstimatedLaserOnFrameNo), taskNum), SNThreshold = True)
        
        task.queueID = self.queueID
        task.initializeWorkerTimeout(time.clock())
        self.tasksInProgress.append(task)

        return task

	
	def checkTimeouts(self):
		curTime = time.clock()
		for it in self.tasksInProgress:
			if 'workerTimeout' in dir(it):
				if curTime > it.workerTimeout:
					self.openTasks.insert(0, it.taskNum)
					self.tasksInProgress.remove(it)


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
            self.dataFileLock.acquire()
            res = self.h5DataFile.root.ImageData.shape[0]
            self.dataFileLock.release()
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


    def releaseTasks(self, startingAt = 0):
        self.openTasks += range(startingAt, self.imNum)
        self.releaseNewTasks = True
