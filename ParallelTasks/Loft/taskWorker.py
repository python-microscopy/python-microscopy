#!/usr/bin/python

##################
# taskWorker.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/python
import Pyro.core

Pyro.config.PYRO_MOBILE_CODE=1

tq = Pyro.core.getProxyForURI("PYRONAME://taskQueue")

while 1:
    print 'Geting Task ...'
    tq.returnCompletedTask(tq.getTask()())
    print 'Completed Task'
