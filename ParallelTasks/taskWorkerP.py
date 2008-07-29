#!/usr/bin/python
import Pyro.core
import os

Pyro.config.PYRO_MOBILE_CODE=1

tq = Pyro.core.getProxyForURI("PYRONAME://taskQueue")

name = os.uname()[1] + ' - PID:%d' % os.getpid()

for i in range(1000):
    #print 'Geting Task ...'
    tq.returnCompletedTask(tq.getTask()(), name)
    #print 'Completed Task'
