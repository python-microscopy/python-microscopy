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
import os
import sys

from PYME.mProfile import mProfile
mProfile.profileOn(['remFitBuf.py', 'taskWorkerP.py'])

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

for i in range(100):
    #print 'Geting Task ...'
    #tq.returnCompletedTask(tq.getTask()(taskQueue=tq), name)
    tasks = tq.getTasks()
    results = [task(taskQueue=tq) for task in tasks]
    tq.returnCompletedTasks(results, name)
    #print 'Completed Task'

mProfile.report()