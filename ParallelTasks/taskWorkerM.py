#!/usr/bin/python
import Pyro.core
import os
import sys

Pyro.config.PYRO_MOBILE_CODE=1

tq = Pyro.core.getProxyForURI("PYRONAME://taskQueue")

if sys.platform == 'win32':
    name = os.environ['COMPUTERNAME'] + ' - PID:%d' % os.getpid()
else:
    name = os.uname()[1] + ' - PID:%d' % os.getpid()

while 1:
    #print 'Geting Task ...'
    tq.returnCompletedTask(tq.getTask()(), name)
    #print 'Completed Task'
