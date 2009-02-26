#!/usr/bin/python
import Pyro.core
import os
import sys

from PYME.mProfile import mProfile

mProfile.profileOn(['remFitBuf.py', 'taskWorkerP.py'])

Pyro.config.PYRO_MOBILE_CODE=0

tq = Pyro.core.getProxyForURI("PYRONAME://taskQueue")

if sys.platform == 'win32':
    name = os.environ['COMPUTERNAME'] + ' - PID:%d' % os.getpid()
else:
    name = os.uname()[1] + ' - PID:%d' % os.getpid()

for i in range(1000):
    #print 'Geting Task ...'
    task = tq.getTask()
    ret = task(taskQueue=tq)
    tq.returnCompletedTask(ret, name)
    #tq.returnCompletedTask(tq.getTask()(taskQueue=tq), name)
    #print 'Completed Task'

mProfile.report()
