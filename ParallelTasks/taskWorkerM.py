#!/usr/bin/python

##################
# taskWorkerM.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import Pyro.core
import os
import sys

if 'PYRO_NS_HOSTNAME' in os.environ.keys():
    Pyro.config.PYRO_NS_HOSTNAME=os.environ['PYRO_NS_HOSTNAME']

Pyro.config.PYRO_MOBILE_CODE=1

if 'PYME_TASKQUEUENAME' in os.environ.keys():
    taskQueueName = os.environ['PYME_TASKQUEUENAME']
else:
    taskQueueName = 'taskQueue'


tq = Pyro.core.getProxyForURI("PYRONAME://" + taskQueueName)

if sys.platform == 'win32':
    name = os.environ['COMPUTERNAME'] + ' - PID:%d' % os.getpid()
else:
    name = os.uname()[1] + ' - PID:%d' % os.getpid()

#loop forever asking for tasks
while 1:
    #ask the queue for tasks
    tasks = tq.getTasks()

    #results = []

    #loop over tasks - we pop each task and then delete it after processing
    #to keep memory usage down
    while len(tasks) > 0:
        #get the next task (a task is a function, or more generally, a class with
        #a __call__ method
        task = tasks.pop()
        try:
            #execute the task, 
            res = task(taskQueue=tq)
            tq.returnCompletedTask(res, name)
        except:
            import traceback
            traceback.print_exc()
        
        del task
        
    #tq.returnCompletedTasks(results, name)
    del tasks
    #del results
    
