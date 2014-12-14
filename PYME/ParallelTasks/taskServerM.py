#!/usr/bin/python

##################
# taskServerM.py
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

#!/usr/bin/python
import Pyro.core
import Pyro.naming
import time
import random
import threading
import numpy
from taskQueue import *
from HDFTaskQueue import *

import os

if 'PYRO_NS_HOSTNAME' in os.environ.keys():
    Pyro.config.PYRO_NS_HOSTNAME=os.environ['PYRO_NS_HOSTNAME']
    print((Pyro.config.PYRO_NS_HOSTNAME))

#from PYME.mProfile import mProfile
#mProfile.profileOn(['taskServerM.py', 'HDFTaskQueue.py'])

if 'PYME_TASKQUEUENAME' in os.environ.keys():
    taskQueueName = os.environ['PYME_TASKQUEUENAME']
else:
    taskQueueName = 'taskQueue'
	
class TaskWatcher(threading.Thread):
	def __init__(self, tQueue):
		threading.Thread.__init__(self)
		self.tQueue = tQueue
		self.alive = True

	def run(self):
		while self.alive:
			self.tQueue.checkTimeouts()
			#print '%d tasks in queue' % self.tQueue.getNumberOpenTasks()
			#try:
                        #        mProfile.report()
                        #finally:
                        #        pass
                        #print mProfile.files
			time.sleep(10)







class TaskQueueSet(Pyro.core.ObjBase):
    def __init__(self):
        Pyro.core.ObjBase.__init__(self)
        self.taskQueues = {}
        self.numTasksProcessed = 0
        self.numTasksProcByWorker = {}
        self.lastTaskByWorker = {}
        self.activeWorkers = []
        self.activeTimeout = 10

        self.getTaskLock = threading.Lock()


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
        self.getTaskLock.acquire()
        while self.getNumberOpenTasks() < 1:
            time.sleep(0.01)

        if not workerName in self.activeWorkers:
            self.activeWorkers.append(workerName)
            
        queuesWithOpenTasks = [q for q in self.taskQueues.values() if q.getNumberOpenTasks() > 0]

        res = queuesWithOpenTasks[int(numpy.round(len(queuesWithOpenTasks)*numpy.random.rand() - 0.5))].getTask(self.activeWorkers.index(workerName), len(self.activeWorkers))
        self.getTaskLock.release()
        return res

    def getTasks(self, workerName='Unspecified'):
        """get task from front of list, blocks"""
        #print 'Task requested'
        self.getTaskLock.acquire()
        while self.getNumberOpenTasks() < 1:
            time.sleep(0.01)

        if not workerName in self.activeWorkers:
            self.activeWorkers.append(workerName)

        queuesWithOpenTasks = [q for q in self.taskQueues.values() if q.getNumberOpenTasks() > 0]

        res = queuesWithOpenTasks[int(numpy.round(len(queuesWithOpenTasks)*numpy.random.rand() - 0.5))].getTasks(self.activeWorkers.index(workerName), len(self.activeWorkers))
        self.getTaskLock.release()
        return res

    def returnCompletedTask(self, taskResult, workerName='Unspecified'):
        self.taskQueues[taskResult.queueID].returnCompletedTask(taskResult)
        self.numTasksProcessed += 1
        if not workerName in self.numTasksProcByWorker.keys():
            self.numTasksProcByWorker[workerName] = 0

        self.numTasksProcByWorker[workerName] += 1
        self.lastTaskByWorker[workerName] = time.time()

    def returnCompletedTasks(self, taskResult, workerName='Unspecified'):
        self.taskQueues[taskResult[0].queueID].returnCompletedTasks(taskResult)
        self.numTasksProcessed += len(taskResult)
        if not workerName in self.numTasksProcByWorker.keys():
            self.numTasksProcByWorker[workerName] = 0

        self.numTasksProcByWorker[workerName] += len(taskResult)
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
            if self.lastTaskByWorker.has_key(w) and self.lastTaskByWorker[w] < (t - self.activeTimeout):
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
        self.taskQueues[queueName].cleanup()
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

    def getQueueData(self, queueName, *args):
        '''Get data ascociated with queue - for cases when you might not want to send data with task every time e.g. to allow client side buffering of image data'''
        return self.taskQueues[queueName].getQueueData(*args)

    def setQueueData(self, queueName, *args):
        '''Set data ascociated with queue'''
        self.taskQueues[queueName].setQueueData(*args)

    def addQueueEvents(self, queueName, *args):
        '''Set data ascociated with queue'''
        self.taskQueues[queueName].addQueueEvents(*args)

    def getQueueMetaData(self, queueName, *args):
        '''Get meta-data ascociated with queue'''
        return self.taskQueues[queueName].getQueueMetaData(*args)

    def setQueueMetaData(self, queueName, *args):
        '''Set meta-data ascociated with queue'''
        self.taskQueues[queueName].setQueueMetaData(*args)
        
    def setQueueMetaDataEntries(self, queueName, *args):
        '''Set meta-data ascociated with queue'''
        self.taskQueues[queueName].setQueueMetaDataEntries(*args)

    def getQueueMetaDataKeys(self, queueName, *args):
        '''Get meta-data keys ascociated with queue'''
        return self.taskQueues[queueName].getQueueMetaDataKeys(*args)

    def logQueueEvent(self, queueName, *args):
        '''Report an event ot a queue'''
        return self.taskQueues[queueName].logQueueEvent(*args)

    def releaseTasks(self, queueName, *args):
        '''Release held tasks'''
        return self.taskQueues[queueName].releaseTasks(*args)

    def createQueue(self, queueType, queueName, *args, **kwargs):
        if queueName in self.taskQueues.keys():
            raise RuntimeError('queue with same name already present')

        self.taskQueues[queueName] = eval(queueType)(queueName, *args, **kwargs)
		
			

if __name__ == '__main__':
    #if True:
    #from PYME.mProfile import mProfile
    #mProfile.profileOn(['taskServerM.py', 'HDFTaskQueue.py'])

    Pyro.config.PYRO_MOBILE_CODE = 0
    Pyro.core.initServer()
    ns=Pyro.naming.NameServerLocator().getNS()
    daemon=Pyro.core.Daemon()
    daemon.useNameServer(ns)

    #get rid of any previous queue
    try:
        ns.unregister(taskQueueName)
    except Pyro.errors.NamingError:
        pass

    tq = TaskQueueSet()
    uri=daemon.connect(tq,taskQueueName)

    tw = TaskWatcher(tq)
    #tw.start()
    try:
        daemon.requestLoop()
    finally:
        daemon.shutdown(True)
        tw.alive = False
        #mProfile.report()
