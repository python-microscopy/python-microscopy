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

#!/usr/bin/python
import Pyro.core
import Pyro.naming
import random
import time

import os

from PYME.misc.computerName import GetComputerName
compName = GetComputerName()

if 'PYRO_NS_HOSTNAME' in os.environ.keys():
    Pyro.config.PYRO_NS_HOSTNAME=os.environ['PYRO_NS_HOSTNAME']

Pyro.config.PYRO_MOBILE_CODE=1

#if 'PYME_TASKQUEUENAME' in os.environ.keys():
#    taskQueueName = os.environ['PYME_TASKQUEUENAME']
#else:
#    taskQueueName = 'taskQueue'
    
ns=Pyro.naming.NameServerLocator().getNS()

#tq = Pyro.core.getProxyForURI("PYRONAME://" + taskQueueName)

procName = compName + ' - PID:%d' % os.getpid()


#loop forever asking for tasks
while 1:
    queueNames = [n[0] for n in ns.list('TaskQueues')]

    tasks = []

    #loop over all queues, looking for tasks to process
    while len(tasks) == 0 and len(queueNames) > 0:
        #try queue on current machine first
        if compName in queueNames:
            qName = compName
            queueNames.remove(qName)
        else: #pick a queue at random
            qName = queueNames.pop(random.randint(0, len(queueNames)-1))

        try:
            tq = Pyro.core.getProxyForURI(ns.resolve('TaskQueues.%s' % qName))

            #ask the queue for tasks
            tasks = tq.getTasks()
        except:
            pass

    if len(tasks) == 0: #no queues had tasks
        time.sleep(1) #put ourselves to sleep to avoid constant polling

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
            tq.returnCompletedTask(res, procName)
        except:
            import traceback
            traceback.print_exc()
        
        del task
        
    #tq.returnCompletedTasks(results, name)
    del tasks
    #del results
