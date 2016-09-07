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
#import Pyro.core
#import Pyro.naming
import random
import time

import PYME.version
#import PYME.misc.pyme_zeroconf as pzc

import os
import requests
#import socket

from PYME.localization import remFitBuf
from PYME.ParallelTasks import distribution

from PYME.misc.computerName import GetComputerName
compName = GetComputerName()

LOCAL = False
if 'PYME_LOCAL_ONLY' in os.environ.keys():
    LOCAL = os.environ['PYME_LOCAL_ONLY'] == '1'

def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('')

    #ns = pzc.getNS('_pyme-taskdist')

    procName = compName + ' - PID:%d' % os.getpid()

    #loop forever asking for tasks
    while 1:
        #queueNames = ns.list('HTTPTaskQueues')
        # queueURLs = {}
        #
        # for name, info in ns.advertised_services.items() :
        #     if name.startswith('PYMENodeServer'):
        #         queueURLs[name] = 'http://%s:%d/' % (socket.inet_ntoa(info.address), info.port)

        queueURLs = distribution.getNodeInfo()

        tasks = []

        #loop over all queues, looking for tasks to process
        while len(tasks) == 0 and len(queueURLs) > 0:
            #try queue on current machine first
            #TODO - only try local machine?
            #print queueNames
            localQueueName = 'PYMENodeServer: ' + compName
            if localQueueName in queueURLs.keys():
                qName = localQueueName
                queueURL = queueURLs.pop(qName)
            else: #pick a queue at random
                queueURL = queueURLs.pop(queueURLs.keys()[random.randint(0, len(queueURLs)-1)])

            try:
                #ask the queue for tasks
                #TODO - make the server actually return a list of tasks, not just one (or implement pipelining in another way)
                #try:
                r = requests.get(queueURL + 'node/tasks?workerID=%s' % procName, timeout=100)
                if r.status_code == 200:
                    resp = r.json()
                    if resp['ok']:
                        tasks.append((queueURL, resp['result']))
            except requests.Timeout:
                logging.error('Read timout requesting tasks from %s' % queueURL)


            except Exception:
                import traceback
                logger.exception(traceback.format_exc())
            
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
            queueURL, taskDescr = tasks.pop(0)
            if taskDescr['type'] == 'localization':
                try:
                    #execute the task,
                    #t1 = time.time()
                    #print taskDescr

                    task = remFitBuf.createFitTaskFromTaskDef(taskDescr)
                    res = task()
                    #t2 = time.time()

                    # new style way of returning results to reduce load on server
                    from PYME.IO import clusterResults
                    clusterResults.fileResults(taskDescr['outputs']['results'], res)

                    r = requests.post(queueURL + 'node/handin?taskID=%s&status=success' % taskDescr['id'])
                    if not r.status_code == 200:
                        logging.error('Returning task failed with error: %s' % r.status_code)

                except:
                    import traceback
                    traceback.print_exc()

                    r = requests.post(queueURL + 'node/handin?taskID=%s&status=failure' % taskDescr['id'])
                    if not r.status_code == 200:
                        logging.error('Returning task failed with error: %s' % r.status_code)
                finally:
                    del task
            
        #tq.returnCompletedTasks(results, name)
        del tasks
        #del results

if __name__ == '__main__':
    main()
