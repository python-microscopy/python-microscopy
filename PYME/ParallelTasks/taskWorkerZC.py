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

import PYME.version
from PYME.misc import hybrid_ns

import os
import sys

from PYME.misc.computerName import GetComputerName
compName = GetComputerName()

LOCAL = False
if 'PYME_LOCAL_ONLY' in os.environ.keys():
    LOCAL = os.environ['PYME_LOCAL_ONLY'] == '1'

if 'PYRO_NS_HOSTNAME' in os.environ.keys():
    Pyro.config.PYRO_NS_HOSTNAME=os.environ['PYRO_NS_HOSTNAME']

Pyro.config.PYRO_MOBILE_CODE=0

#if 'PYME_TASKQUEUENAME' in os.environ.keys():
#    taskQueueName = os.environ['PYME_TASKQUEUENAME']
#else:
#    taskQueueName = 'taskQueue'

def main():    
    #ns=Pyro.naming.NameServerLocator().getNS()
    ns = hybrid_ns.getNS()

    #tq = Pyro.core.getProxyForURI("PYRONAME://" + taskQueueName)

    procName = compName + ' - PID:%d' % os.getpid()
    import logging
    logging.basicConfig(filename='taskWorkerZC_%d.log' % os.getpid(), level=logging.INFO)
    logger = logging.getLogger(__file__)

    serverFails = {}

    #loop forever asking for tasks
    while 1:
        queueNames = ns.list('TaskQueues')
        
        #print queueNames

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
                #print qName
                tq = Pyro.core.getProxyForURI(ns.resolve(qName))
                tq._setTimeout(10)
                tq._setOneway(['returnCompletedTask'])
                #print qName

                #ask the queue for tasks
                logging.debug('Getting tasks from server')
                tasks = tq.getTasks(procName, PYME.version.version)
                logging.debug('Got %d tasks' % len(tasks))

                #we succesfully contacted the server, so reset it's fail count
                serverFails[qName] = 0
            except Pyro.core.ProtocolError as e:
                logging.exception('Pyro error: %s' %e.message)
                if e.message == 'connection failed':
                    #remember that the server failed - and put it 'on notice'
                    nFails = 1
                    if qName in serverFails.keys():
                        nFails += serverFails[qName]

                    serverFails[qName] = nFails

                    if False:#nFails >= 4:
                        #server is dead in the water - put it out of it's misery
                        print(('Killing:', qName))
                        try:
                            ns.unregister('TaskQueues.%s' % qName)
                        except Pyro.errors.NamingError:
                            pass
            except Exception:
                import traceback
                logger.exception(traceback.format_exc())
            
                #pass
            
        
        
        if len(tasks) == 0: #no queues had tasks
            logger.debug('No tasks avaialable, waiting')
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
                logger.debug('running task')
                res = task(taskQueue=tq)
                t2 = time.time()

                if not task.resultsURI is None:
                    # new style way of returning results to reduce load on server
                    from PYME.IO import clusterResults
                    clusterResults.fileResults(task.resultsURI, res)

                logging.debug('Returning task for frame %d' % res.index)
                tq.returnCompletedTask(res, procName, t2-t1)
            except:
                import traceback
                logger.exception('Error returning results')
                traceback.print_exc()
            
            del task
            
        #tq.returnCompletedTasks(results, name)
        del tasks
        #del results

if __name__ == '__main__':
    profile = False
    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        print('profiling')
        profile = True
        from PYME.util.mProfile import mProfile
        
        mProfile.profileOn(['taskWorkerZC.py', ])
    
    if len(sys.argv) > 1 and sys.argv[1] == '-fp':
        print('profiling')
        #profile = True
        from PYME.util.fProfile import fProfile
        
        tp = fProfile.thread_profiler()
        tp.profileOn('.*taskWorkerZC.*|.*PYME.*', 'taskWorker_%d_prof.txt' % os.getpid())
    
    try:
        main()
    finally:
        if profile:
            print('Profile report')
            mProfile.profileOff()
            mProfile.report()
