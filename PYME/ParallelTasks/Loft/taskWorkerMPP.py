#!/usr/bin/python

##################
# taskWorkerM.py
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
import random
import time

import os

#from PYME.mProfile import mProfile

#mProfile.profileOn(['taskWorkerMPP.py'])

from PYME.misc.computerName import GetComputerName
compName = GetComputerName()

if 'PYRO_NS_HOSTNAME' in os.environ.keys():
    Pyro.config.PYRO_NS_HOSTNAME=os.environ['PYRO_NS_HOSTNAME']

Pyro.config.PYRO_MOBILE_CODE=0

#if 'PYME_TASKQUEUENAME' in os.environ.keys():
#    taskQueueName = os.environ['PYME_TASKQUEUENAME']
#else:
#    taskQueueName = 'taskQueue'
    
ns=Pyro.naming.NameServerLocator().getNS()

#tq = Pyro.core.getProxyForURI("PYRONAME://" + taskQueueName)

procName = compName + ' - PID:%d' % os.getpid()


#loop forever asking for tasks
for i in range(3):
    queueNames = [n[0] for n in ns.list('TaskQueues')]

    tasks = []

    #loop over all queues, looking for tasks to process
    while len(tasks) == 0 and len(queueNames) > 0:
        #try queue on current machine first
        #print queueNames
        if compName in queueNames:
            qName = compName
            queueNames.remove(qName)
        else: #pick a queue at random
            qName = queueNames.pop(random.randint(0, len(queueNames)-1))

        try:
            tq = Pyro.core.getProxyForURI(ns.resolve('TaskQueues.%s' % qName))
            tq._setOneway(['returnCompletedTask'])
            #print qName

            #ask the queue for tasks
            tasks = tq.getTasks(procName)
            
        except:
            pass
            #import traceback
            #traceback.print_exc()
        
            #pass
        
    
    
    if len(tasks) == 0: #no queues had tasks
        time.sleep(1) #put ourselves to sleep to avoid constant polling
    #else:
    #    print qName, len(tasks)

    #results = []

    #loop over tasks - we pop each task and then delete it after processing
    #to keep memory usage down
    while len(tasks) > 0:
        #get the next task (a task is a function, or more generally, a class with
        #a __call__ method
        task = tasks.pop()
        try:
            #execute the task,
            t1 = time.time()
            res = task(taskQueue=tq)
            t2 = time.time()
            tq.returnCompletedTask(res, procName, t2-t1)
        except:
            import traceback
            traceback.print_exc()
        
        del task
        
    #tq.returnCompletedTasks(results, name)
    del tasks
    #del results

#mProfile.report()