import tables
from taskQueue import *
from PYME.Analysis.remFitBuf import fitTask

from PYME.Analysis import MetaData

import os

from PYME.FileUtils.nameUtils import genResultFileName
from PYME.ParallelTasks.relativeFiles import getFullFilename

#def genDataFilename(name):
#	fn = os.g


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

                self.fileResultsLock = threading.Lock()
                
                
	def prepResultsFile(self):
            pass
	

        def getCompletedTask(self):
            return None

	

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
            if res.results == []: #if we had a dud frame
                return 

            self.fileResultsLock.acquire() #get a lock
            
            if not self.h5ResultsFile.__contains__('/FitResults'):
                self.h5ResultsFile.createTable(self.h5ResultsFile.root, 'FitResults', res.results, filters=tables.Filters(complevel=5, shuffle=True))
            else:
                self.h5ResultsFile.root.FitResults.append(res.results)

	    self.h5ResultsFile.flush()

	    self.fileResultsLock.release() #release lock

	    self.numClosedTasks += 1

                                      

class HDFTaskQueue(HDFResultsTaskQueue):
	''' task queue which, when initialised with an hdf image filename, automatically generated tasks - should also (eventually) include support for dynamically adding to data file for on the fly analysis'''
	def __init__(self, name, fitParams, dataFilename = None, resultsFilename=None, onEmpty = doNix, fTaskToPop = popZero, startAt = -1):		
                if dataFilename == None:
                   self.dataFilename = genDataFilename(name)
                else: 
                    self.dataFilename = dataFilename

                if resultsFilename == None:
                    resultsFilename = genResultFileName(self.dataFilename)
                else:
                    resultsFilename = resultsFilename  
		
                
		self.h5DataFile = tables.openFile(getFullFilename(self.dataFilename), 'r')
		self.metaData = MetaData.genMetaDataFromHDF(self.h5DataFile)
		print self.metaData.CCD.ADOffset

		if startAt == -1: #calculate a suitable starting value
			tLon = self.metaData.EstimatedLaserOnFrameNo
			if tLon == 0:
				startAt = 0
			else:
				startAt = tLon + 10

                initialTasks = list(range(startAt, self.h5DataFile.root.ImageData.shape[0]))

		HDFResultsTaskQueue.__init__(self, name, resultsFilename, initialTasks, onEmpty, fTaskToPop)

		self.queueID = name

		self.fitParams = fitParams
		self.dataFileLock = threading.Lock()
                
	def prepResultsFile(self):
            pass

        def postTask(self,task):
		#self.openTasks.append(task)
		print 'posting tasks not implemented yet'

	def postTasks(self,tasks):
		#self.openTasks += tasks
		print 'posting tasks not implemented yet'

	def getTask(self, workerN = 0, NWorkers = 1):
		"""get task from front of list, blocks"""
		#print 'Task requested'
		while len(self.openTasks) < 1:
			time.sleep(0.01)

		taskNum = self.openTasks.pop(self.fTaskToPop(workerN, NWorkers, len(self.openTasks)))

		task = fitTask(self.queueID, taskNum, self.fitParams['threshold'], self.metaData, self.fitParams['fitModule'], 'TQDataSource', bgindices =range(max(taskNum,self.metaData.EstimatedLaserOnFrameNo), taskNum), SNThreshold = True)

                task.queueID = self.queueID
		task.initializeWorkerTimeout(time.clock())
		self.tasksInProgress.append(task)
		
		return task

	
	def checkTimeouts(self):
		curTime = time.clock()
		for it in self.tasksInProgress:
			if 'workerTimeout' in dir(it):
				if curTime > workerTimeout:
					self.openTasks.insert(0, it.taskNum)
					self.tasksInProgress.remove(it)


        def cleanup(self):
            self.h5DataFile.close()
            self.h5ResultsFile.close()


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
		else:
			return None
