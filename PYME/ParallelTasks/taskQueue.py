#!/usr/bin/python

##################
# taskQueue.py
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

import time
import threading

CHUNKSIZE = 50

def doNix(taskQueue): #do nothing
    pass

def popZero(workerN, NWorkers, NTasks): #give worker oldest task irrespective of which worker called
    return 0

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
        self.inProgressLock = threading.Lock()

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
        task.initializeWorkerTimeout(time.time())
        with self.inProgressLock:
            self.tasksInProgress.append(task)
        
        #print '[%s] - Task given to worker' % self.queueID
        return task

    def getTasks(self, workerN = 0, NWorkers = 1):
        return [self.getTask(workerN, NWorkers) for i in range(min(CHUNKSIZE,len(self.openTasks)))]

    def returnCompletedTask(self, taskResult):
            with self.inProgressLock:
                for it in self.tasksInProgress[:]:
                        if (it.taskID == taskResult.taskID):
                                self.tasksInProgress.remove(it)
            
            
            self.fileResult(taskResult)

            if (len(self.openTasks) + len(self.tasksInProgress)) == 0: #no more tasks
                    self.onEmpty(self)
            

    def returnCompletedTasks(self, taskResults):
        with self.inProgressLock:
            for taskResult in taskResults:
                for it in self.tasksInProgress[:]:
                    if (it.taskID == taskResult.taskID):
                        self.tasksInProgress.remove(it)
        
        
        #for taskResult in taskResults:
        #allow this to be over-ridden 
        self.fileResults(taskResults)

        if (len(self.openTasks) + len(self.tasksInProgress)) == 0: #no more tasks
            self.onEmpty(self)
        

    def fileResults(self, taskResults):
        #allow this to be over-ridden in derived classes to file multiple results at once
        for taskResult in taskResults:
            self.fileResult(taskResult)
    
    def fileResult(self,taskResult):
        self.closedTasks.append(taskResult)

    def getCompletedTask(self):
        if len(self.closedTasks) < 1:
            return None
        else:
            return self.closedTasks.pop(0)

    def checkTimeouts(self):
        with self.inProgressLock:
            curTime = time.time()
            for it in self.tasksInProgress:
                if 'workerTimeout' in dir(it):
                    if curTime > it.workerTimeout:
                        self.openTasks.insert(0, it)
                        self.tasksInProgress.remove(it)
        

    def getNumberOpenTasks(self, exact=True):
        return len(self.openTasks)

    def getNumberTasksInProgress(self):
        return len(self.tasksInProgress)

    def getNumberTasksCompleted(self):
        return len(self.closedTasks)

    def cleanup(self):
        pass

    def purge(self):
        self.openTasks = []
        self.closedTasks = []
        self.tasksInProgress = []

    def setPopFcn(self, fcn):
        """ sets the function which determines which task to give a worker"""
        self.fTaskToPop = fcn


class TaskQueueWithData(TaskQueue):
    def __init__(self, name, initialTasks=[], onEmpty = doNix, fTaskToPop = popZero):
        TaskQueue.__init__(self, name, initialTasks, onEmpty, fTaskToPop)

        self.data = {}

    def getTasks(self, workerN = 0, NWorkers = 1):
        return [self.getTask(workerN, NWorkers)]

    def getQueueData(self, fieldName, *args):
        """Get data, defined by fieldName and potntially additional arguments,  ascociated with queue"""

        return self.data[fieldName]

    def setQueueData(self, fieldName, value):
        """Get data, defined by fieldName and potntially additional arguments,  ascociated with queue"""

        self.data[fieldName] = value
