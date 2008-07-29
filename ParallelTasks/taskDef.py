import random

class Task:
    def __init__(self):
        self.taskID = repr(random.random())
    def initializeWorkerTimeout(self, curtime):
        pass

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
