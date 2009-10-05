#!/usr/bin/python

##################
# taskWorkerME.py
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
from PYME.Analysis import remFitBuf
from PYME.Analysis.FitFactories import LatGaussFitFR
from PYME.Analysis.DataSources import *

Pyro.config.PYRO_MOBILE_CODE=1

tq = Pyro.core.getProxyForURI("PYRONAME://taskQueue")

if sys.platform == 'win32':
    name = os.environ['COMPUTERNAME'] + ' - PID:%d' % os.getpid()
else:
    name = os.uname()[1] + ' - PID:%d' % os.getpid()

while 1:
    #print 'Geting Task ...'
    tq.returnCompletedTask(tq.getTask()(taskQueue=tq), name)
    #print 'Completed Task'
