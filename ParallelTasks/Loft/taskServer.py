#!/usr/bin/python
import Pyro.core
import Pyro.naming
import time
import random
import threading

class TaskWatcher(threading.Thread):
	def __init__(self, tQueue):
		threading.Thread.__init__(self)
		self.tQueue = tQueue

	def run(self):
		while True:
			self.tQueue.checkTimeouts()
			print '%d tasks in queue' % self.tQueue.getNumberOpenTasks()
			time.sleep(10)

class TaskQueue(Pyro.core.ObjBase):
	def __init__(self):
		Pyro.core.ObjBase.__init__(self)
		self.openTasks = []
		self.closedTasks = []
		self.tasksInProgress = []
                
	def postTask(self,task):
		self.openTasks.append(task)
		print 'Recieved new task'

	def getTask(self):
		"""get task from front of list, blocks"""
		print 'Task requested'
		while len(self.openTasks) < 1:
			time.sleep(0.01)

		task = self.openTasks.pop(0)

		task.initializeWorkerTimeout(time.clock())
		self.tasksInProgress.append(task)
		print 'Task given to worker'
		return task

	def returnCompletedTask(self, taskResult):
		for it in self.tasksInProgress:
			if (it.taskID == taskResult.taskID):
				self.tasksInProgress.remove(it)
		self.closedTasks.append(taskResult)

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

			   

Pyro.config.PYRO_MOBILE_CODE = 1
Pyro.core.initServer()
ns=Pyro.naming.NameServerLocator().getNS()
daemon=Pyro.core.Daemon()
daemon.useNameServer(ns)

tq = TaskQueue()
uri=daemon.connect(tq,"taskQueue")

tw = TaskWatcher(tq)
tw.start()
try:
        daemon.requestLoop()
finally:
        daemon.shutdown(True)
