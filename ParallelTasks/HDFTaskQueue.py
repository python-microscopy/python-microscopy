import tables

class HDFTaskQueue(TaskQueue):
	def __init__(self, name, dataFilename = None, resultsFilename=None, onEmpty = doNix, fTaskToPop = popZero):
		#Pyro.core.ObjBase.__init__(self)
		#self.name = name
		self.queueID = name
		
                if dataFilename == None:
                   self.dataFilename = genDataFilename(name)
                else: 
                    self.dataFilename = dataFilename

                if resultsFilename == None:
                    self.resultsFilename = genResultsFilename(self.dataFilename)
                else:
                    self.resultsFilename = resultsFilename

                #self.openTasks = list(initialTasks)
		self.numClosedTasks = 0
		self.tasksInProgress = []
		self.onEmpty = onEmpty #function to call when queue is empty
		self.fTaskToPop = fTaskToPop #function to call to decide which task to give a worker (useful if the workers need to have share information with, e.g., previous tasks as this can improve eficiency of per worker buffering of said info).
                
		self.h5DataFile = tables.openFile(self.dataFilename, 'r')
                self.h5ResultsFile = tables.openFile(self.dataFilename, 'w')
                
                self.prepResultsFile()
                
                self.openTasks = list(range(self.h5DataFile.root.ImageData.shape[0]))
                
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

		task = HDFQueueTask(taskNum)

                task.queueID = self.queueID
		task.initializeWorkerTimeout(time.clock())
		self.tasksInProgress.append(task)
		
		return task

	def returnCompletedTask(self, taskResult):
		for it in self.tasksInProgress:
			if (it.taskID == taskResult.taskID):
				self.tasksInProgress.remove(it)
		
                self.fileResult(taskResult)
                self.numClosedTasks += 1

		if (len(self.openTasks) + len(self.tasksInProgress)) == 0: #no more tasks
			self.onEmpty(self)

	def checkTimeouts(self):
		curTime = time.clock()
		for it in self.tasksInProgress:
			if 'workerTimeout' in dir(it):
				if curTime > workerTimeout:
					self.openTasks.insert(0, it.taskNum)
					self.tasksInProgress.remove(it)

        def getCompletedTask(self):
            return None

	

	def getNumberTasksCompleted(self):
		return self.numClosedTasks

	def purge(self):
		self.openTasks = []
		self.numClosedTasks = 0
		self.tasksInProgress = []

        def getData(self, sliceNum):
            return self.h5DataFile.root.ImageData[sliceNum, :,:]

        def cleanup(self):
            self.h5DataFile.close()
            self.h5ResultsFile.close()

        def fileResult(self, res):            
            if not self.h5ResultsFile._contains_('/FitResults'):
                self.h5ResultsFile.createTable(h5ResultsFile.root, 'FitResults', res.results, filters=tables.Filters(complevel=5, shuffle=True))
            else:
                self.h5ResultsFile.root.FitResults.append(res.results)

                                      


