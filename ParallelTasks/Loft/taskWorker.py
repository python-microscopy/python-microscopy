#!/usr/bin/python
import Pyro.core

Pyro.config.PYRO_MOBILE_CODE=1

tq = Pyro.core.getProxyForURI("PYRONAME://taskQueue")

while 1:
    print 'Geting Task ...'
    tq.returnCompletedTask(tq.getTask()())
    print 'Completed Task'
