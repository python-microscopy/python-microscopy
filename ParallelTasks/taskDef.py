#!/usr/bin/python

##################
# taskDef.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import random

class Task:
    def __init__(self):
        self.taskID = repr(random.random())
        self.timeout = 300 #seconds
    def initializeWorkerTimeout(self, curtime):
        self.workerTimeout = curtime + self.timeout

class TaskResult:
    def __init__(self, task):
        self.taskID = task.taskID
        if 'queueID' in dir(task):
            self.queueID = task.queueID

class myTask(Task):
 	def __init__(self):
 		Task.__init__(self)
 	def __call__(self):
 		print "Hello"
