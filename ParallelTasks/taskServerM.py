#!/usr/bin/python
import Pyro.core
import Pyro.naming
import time
import random
import threading
import numpy

def doNix(taskQueue): #do nothing
	pass

def popZero(workerN, NWorkers, NTasks): #give worker oldest task irrespective of which worker called
	return 0
	

class TaskWatcher(threading.Thread):
	def __init__(self, tQueue):
		threading.Thread.__init__(self)
		self.tQueue = tQueue

	def run(self):
		while True:
			self.tQueue.checkTimeouts()
			#print '%d tasks in queue' % self.tQueue.getNumberOpenTasks()
			time.sleep(10)

class TaskQueue:
	def __init__(self, name, initialTasks=[], onEmpty = doNix, fTaskToPop = popZero):
		#Pyro.core.ObjBase.__init__(self)
		#self.name = name
		self.queueID = name
		self.openTasks = list(initialTasks)
		self.closedTasks = []
		self.tasksInProgress = []
		self.onEmpty = onEmpty #function to call when queue is empty
		self.fTaskToPop = fTaskToPop #function to call to decide which task to give a worker (useful if the workers need to have share information with, e.g., previous tasks as this can improve eficiency of per worker buffering of said info).
                
	def postTask(self,task):
		self.openTasks.append(task)
		#print '[%s] - Recieved new task' % self.queueID

	def postTasks(self,tasks):
		self.openTasks += tasks
		#print '[%s] - Recieved %d new tasks' % (self.queueID, len(tasks))

	def getTask(self, workerN = 0, NWorkers = 1):
		"""get task from front of list, blocks"""
		#print 'Task requested'
		while len(self.openTasks) < 1:
			time.sleep(0.01)

		task = self.openTasks.pop(self.fTaskToPop(workerN, NWorkers, len(self.openTasks)))

		task.queueID = self.queueID
		task.initializeWorkerTimeout(time.clock())
		self.tasksInProgress.append(task)
		#print '[%s] - Task given to worker' % self.queueID
		return task

	def returnCompletedTask(self, taskResult):
		for it in self.tasksInProgress:
			if (it.taskID == taskResult.taskID):
				self.tasksInProgress.remove(it)
		self.closedTasks.append(taskResult)

		if (len(self.openTasks) + len(self.tasksInProgress)) == 0: #no more tasks
			self.onEmpty(self)

	def getCompletedTask(self):
		if len(self.closedTasks) < 1:
			return None
		else:
			return self.closedTasks.pop(0)

	def checkTimeouts(self):
		curTime = time.clock()
		for it in self.tasksInProgress:
			if 'workerTimeout' in dir(it):
				if curTime > workerTimeout:
					self.openTasks.insert(0, it)
					self.tasksInProgress.remove(it)

	def getNumberOpenTasks(self):
		return len(self.openTasks)

	def getNumberTasksInProgress(self):
		return len(self.tasksInProgress)

	def getNumberTasksCompleted(self):
		return len(self.closedTasks)

	def purge(self):
		self.openTasks = []
		self.closedTasks = []
		self.tasksInProgress = []

	def setPopFcn(self, fcn):
		''' sets the function which determines which task to give a worker'''
		self.fTaskToPop = fcn





class TaskQueueSet(Pyro.core.ObjBase):
	def __init__(self):
		Pyro.core.ObjBase.__init__(self)
		self.taskQueues = {}
		self.numTasksProcessed = 0
		self.numTasksProcByWorker = {}
		self.lastTaskByWorker = {}
		self.activeWorkers = []
		self.activeTimeout = 10
		

	def postTask(self, task, queueName='Default'):
		#print queueName
		if not queueName in self.taskQueues.keys():
			self.taskQueues[queueName] = TaskQueue(queueName)

		self.taskQueues[queueName].postTask(task)

	def postTasks(self, tasks, queueName='Default'):
		if not queueName in self.taskQueues.keys():
			self.taskQueues[queueName] = TaskQueue(queueName)

		self.taskQueues[queueName].postTasks(tasks)

	def getTask(self, workerName='Unspecified'):
		"""get task from front of list, blocks"""
		#print 'Task requested'
		while self.getNumberOpenTasks() < 1:
			time.sleep(0.01)
		
		if not workerName in self.activeWorkers:
			self.activeWorkers.append(workerName)
		queuesWithOpenTasks = [q for q in self.taskQueues.values() if q.getNumberOpenTasks() > 0]	

		return queuesWithOpenTasks[int(numpy.round(len(queuesWithOpenTasks)*numpy.random.rand() - 0.5))].getTask(self.activeWorkers.index(workerName), len(self.activeWorkers))


	def returnCompletedTask(self, taskResult, workerName='Unspecified'):
		self.taskQueues[taskResult.queueID].returnCompletedTask(taskResult)
		self.numTasksProcessed += 1
		if not workerName in self.numTasksProcByWorker.keys():
			self.numTasksProcByWorker[workerName] = 0

		self.numTasksProcByWorker[workerName] += 1
		self.lastTaskByWorker[workerName] = time.time()

	def getCompletedTask(self, queueName = 'Default'):
		if not queueName in self.taskQueues.keys():
			return None
		else:
			return self.taskQueues[queueName].getCompletedTask()

	def checkTimeouts(self):
		for q in self.taskQueues.values():
			q.checkTimeouts()

		t = time.time()
		for w in self.activeWorkers:
			if self.lastTaskByWorker[w] < (t - self.activeTimeout):
				self.activeWorkers.remove(w)

	def getNumberOpenTasks(self, queueName = None):
		#print queueName
		if queueName == None:
			nO = 0
			for q in self.taskQueues.values():
				nO += q.getNumberOpenTasks()
			return nO
		else:
			return self.taskQueues[queueName].getNumberOpenTasks()

	def getNumberTasksInProgress(self, queueName = None):
		if queueName == None:
			nP = 0
			for q in self.taskQueues.values():
				nP += q.getNumberTasksInProgress()
			return nP
		else:
			return self.taskQueues[queueName].getNumberTasksInProgress()

	def getNumberTasksCompleted(self, queueName = None):
		if queueName == None:
			nC = 0
			for q in self.taskQueues.values():
				nC += q.getNumberTasksCompleted()
			return nC
		else:
			return self.taskQueues[queueName].getNumberTasksCompleted()

	def purge(self, queueName = 'Default'):
		if queueName in self.taskQueues.keys():
			self.taskQueues[queueName].purge()

	def removeQueue(self, queueName):
		self.taskQueues.pop(queueName)
	
	def getNumTasksProcessed(self, workerName = None):
		if workerName == None:
			return self.numTasksProcessed
		else:
			return self.numTasksProcByWorker[workerName]

	def getWorkerNames(self):
		return self.numTasksProcByWorker.keys()

	def getQueueNames(self):
		return self.taskQueues.keys()

	def setPopFcn(self, queueName, fcn):
		self.taskQueues[queueName].setPopFcn(fcn)
			

if __name__ == '__main__':

	Pyro.config.PYRO_MOBILE_CODE = 1
	Pyro.core.initServer()
	ns=Pyro.naming.NameServerLocator().getNS()
	daemon=Pyro.core.Daemon()
	daemon.useNameServer(ns)

	tq = TaskQueueSet()
	uri=daemon.connect(tq,"taskQueue")

	tw = TaskWatcher(tq)
	tw.start()
	try:
		daemon.requestLoop()
	finally:
		daemon.shutdown(True)
