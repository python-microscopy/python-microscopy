#!/usr/bin/python

##################
# taskWorkerME.py
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

#make sure we have our dependencies
from PYME.Analysis import remFitBuf, MetaData
from PYME.localization.FitFactories import *
from PYME.localization.FitFactories.Interpolators import *
from PYME.localization.FitFactories.zEstimators import *
from PYME.Analysis.DataSources import HDFDataSource, TQDataSource
import matplotlib.backends.backend_wxagg


from PYME.misc.computerName import GetComputerName
compName = GetComputerName()

if 'PYRO_NS_HOSTNAME' in os.environ.keys():
    Pyro.config.PYRO_NS_HOSTNAME=os.environ['PYRO_NS_HOSTNAME']

Pyro.config.PYRO_MOBILE_CODE=0

ns=Pyro.naming.NameServerLocator().getNS()

procName = compName + ' - PID:%d' % os.getpid()

#loop forever asking for tasks
while 1:
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
            
        except Pyro.core.ProtocolError as e:
            if e.message == 'connection failed':
                #server is dead in the water - put it out of it's misery
                print(('Killing:', qName))
                try:
                    ns.unregister('TaskQueues.%s' % qName)
                except Pyro.errors.NamingError:
                    pass
        except Exception:
            import traceback
            traceback.print_exc()
        
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
        task = tasks.pop(0)
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
