#!/usr/bin/python

##################
# taskWorkerP.py
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
import time
import random
import threading
import numpy
from taskQueue import *
from HDFTaskQueue import *

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
