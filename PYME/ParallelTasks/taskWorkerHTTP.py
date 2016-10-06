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
import os

import Queue
import threading

import PYME.version
#import PYME.misc.pyme_zeroconf as pzc
from PYME import config
from PYME.misc.computerName import GetComputerName
compName = GetComputerName()

import logging
import logging.handlers
dataserver_root = config.get('dataserver-root')
if dataserver_root:
    log_dir = '%s/LOGS/%s/taskWorkerHTTP' % (dataserver_root, compName)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except:
            #as we are launching multiple processes at once, there is a race condition here and we might
            #have already created the directory between our test and the makedirs call
            pass
        
    #fh = logging.FileHandler('%s/%d.log' % (log_dir, os.getpid()), 'w')
    #fh.setLevel(logging.DEBUG)
    #logger.addHandler(fh)
    #logging.basicConfig(filename ='%s/%d.log' % (log_dir, os.getpid()), level=logging.DEBUG)
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    fh = logging.handlers.RotatingFileHandler(filename ='%s/%d.log' % (log_dir, os.getpid()), mode='w', maxBytes=1e6)
    logger.addHandler(fh)
else:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('')


import requests
import sys
import signal
#import socket

from PYME.localization import remFitBuf
from PYME.ParallelTasks import distribution

#import here to pre-populate the zeroconf nameserver
from PYME.IO import clusterIO
#time.sleep(3)


LOCAL = False
if 'PYME_LOCAL_ONLY' in os.environ.keys():
    LOCAL = os.environ['PYME_LOCAL_ONLY'] == '1'

def main(): 
    #ns = pzc.getNS('_pyme-taskdist')

    procName =  '%s_%d' % (compName, os.getpid())

    #loop forever asking for tasks
    while 1:
        #queueNames = ns.list('HTTPTaskQueues')
        # queueURLs = {}
        #
        # for name, info in ns.advertised_services.items() :
        #     if name.startswith('PYMENodeServer'):
        #         queueURLs[name] = 'http://%s:%d/' % (socket.inet_ntoa(info.address), info.port)

        queueURLs = distribution.getNodeInfo()
        localQueueName = 'PYMENodeServer: ' + compName

        queueURLs = {k: v for k, v in queueURLs.items() if k == localQueueName }

        tasks = []

        #loop over all queues, looking for tasks to process
        while len(tasks) == 0 and len(queueURLs) > 0:
            #try queue on current machine first
            #TODO - only try local machine?
            #print queueNames

            if localQueueName in queueURLs.keys():
                qName = localQueueName
                queueURL = queueURLs.pop(qName)
            else:
                logger.error('Could not find local node server')
            #else: #pick a queue at random
            #    queueURL = queueURLs.pop(queueURLs.keys()[random.randint(0, len(queueURLs)-1)])

            try:
                #ask the queue for tasks
                #TODO - make the server actually return a list of tasks, not just one (or implement pipelining in another way)
                #try:
                s = clusterIO._getSession(queueURL)
                r = s.get(queueURL + 'node/tasks?workerID=%s&numWant=50' % procName)#, timeout=0)
                if r.status_code == 200:
                    resp = r.json()
                    if resp['ok']:
                        res = resp['result']
                        if isinstance(res, list):
                            tasks += [(queueURL, t) for t in res]
                        else:
                            tasks.append((queueURL, res))
            except requests.Timeout:
                logger.info('Read timout requesting tasks from %s' % queueURL)


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
                    outputs = taskDescr['outputs']

                    if 'results' in outputs.keys():
                        #old style pickled results
                        clusterResults.fileResults(outputs['results'], res)
                    else:
                        if len(res.results) > 0:
                            clusterResults.fileResults(outputs['fitResults'], res.results)

                        if len(res.driftResults) > 0:
                            clusterResults.fileResults(outputs['driftResults'], res.driftResults)

                    s = clusterIO._getSession(queueURL)
                    r = s.post(queueURL + 'node/handin?taskID=%s&status=success' % taskDescr['id'])
                    if not r.status_code == 200:
                        logger.error('Returning task failed with error: %s' % r.status_code)

                except:
                    import traceback
                    traceback.print_exc()
                    logger.exception(traceback.format_exc())

                    s = clusterIO._getSession(queueURL)
                    r = s.post(queueURL + 'node/handin?taskID=%s&status=failure' % taskDescr['id'])
                    if not r.status_code == 200:
                        logger.error('Returning task failed with error: %s' % r.status_code)
                #finally:
                #    del task
            
        #tq.returnCompletedTasks(results, name)
        del tasks
        #del results


class taskWorker(object):
    def __init__(self):
        self.inputQueue = Queue.Queue()
        self.resultsQueue = Queue.Queue()

        self.procName = '%s_%d' % (compName, os.getpid())

        self._loop_alive = True

    def loop_forever(self):
        self.tCompute = threading.Thread(target=self.computeLoop)
        self.tCompute.daemon = True
        self.tCompute.start()

        self.tIO = threading.Thread(target=self.ioLoop)
        self.tIO.daemon = True
        self.tIO.start()

        try:
            while True:
                time.sleep(1)
        finally:
            self._loop_alive = False

    def ioLoop(self):
        #loop forever asking for tasks
        while True:
            queueURLs = distribution.getNodeInfo()
            localQueueName = 'PYMENodeServer: ' + compName

            queueURLs = {k: v for k, v in queueURLs.items() if k == localQueueName}

            tasks = []

            # new style way of returning results to reduce load on server
            from PYME.IO import clusterResults

            try:
                while True:
                    queueURL, taskDescr, res = self.resultsQueue.get_nowait()
                    outputs = taskDescr['outputs']

                    if res is None:
                        #failure
                        s = clusterIO._getSession(queueURL)
                        r = s.post(queueURL + 'node/handin?taskID=%s&status=failure' % taskDescr['id'])
                        if not r.status_code == 200:
                            logger.error('Returning task failed with error: %s' % r.status_code)
                    else:
                        #success
                        if 'results' in outputs.keys():
                            #old style pickled results
                            clusterResults.fileResults(outputs['results'], res)
                        else:
                            if len(res.results) > 0:
                                clusterResults.fileResults(outputs['fitResults'], res.results)

                            if len(res.driftResults) > 0:
                                clusterResults.fileResults(outputs['driftResults'], res.driftResults)

                        s = clusterIO._getSession(queueURL)
                        r = s.post(queueURL + 'node/handin?taskID=%s&status=success' % taskDescr['id'])
                        if not r.status_code == 200:
                            logger.error('Returning task failed with error: %s' % r.status_code)

            except Queue.Empty:
                pass

            if not self._loop_alive:
                break

            if not self.inputQueue.empty():

                #loop over all queues, looking for tasks to process
                while len(tasks) == 0 and len(queueURLs) > 0:
                    #try queue on current machine first
                    #TODO - only try local machine?
                    #print queueNames

                    if localQueueName in queueURLs.keys():
                        qName = localQueueName
                        queueURL = queueURLs.pop(qName)
                    else:
                        logger.error('Could not find local node server')


                    try:
                        #ask the queue for tasks
                        #TODO - make the server actually return a list of tasks, not just one (or implement pipelining in another way)
                        #try:
                        s = clusterIO._getSession(queueURL)
                        r = s.get(queueURL + 'node/tasks?workerID=%s&numWant=50' % self.procName)#, timeout=0)
                        if r.status_code == 200:
                            resp = r.json()
                            if resp['ok']:
                                res = resp['result']
                                if isinstance(res, list):
                                    tasks += [(queueURL, t) for t in res]
                                else:
                                    tasks.append((queueURL, res))
                    except requests.Timeout:
                        logger.info('Read timout requesting tasks from %s' % queueURL)

                    except Exception:
                        import traceback
                        logger.exception(traceback.format_exc())


                if len(tasks) == 0: #no queues had tasks
                    time.sleep(1) #put ourselves to sleep to avoid constant polling
                else:
                    for t in tasks:
                        self.inputQueue.put(t)

    def computeLoop(self):
        while self._loop_alive:
            #loop over tasks - we pop each task and then delete it after processing
            #to keep memory usage down

            queueURL, taskDescr = self.inputQueue.get()
            if taskDescr['type'] == 'localization':
                try:
                    task = remFitBuf.createFitTaskFromTaskDef(taskDescr)
                    res = task()

                    self.resultsQueue.put(queueURL, taskDescr, res)

                except:
                    import traceback
                    traceback.print_exc()
                    logger.exception(traceback.format_exc())

                    self.resultsQueue.put(queueURL, taskDescr, None)

        
def on_SIGHUP(signum, frame):
    raise RuntimeError('Recieved SIGHUP')
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-p':
            profile = True
            from PYME.util import mProfile
            mProfile.profileOn(['taskWorkerHTTP.py', 'remFitBuf.py'])
            
            if len(sys.argv) == 3:
                profileOutDir = sys.argv[2]
            else:
                profileOutDir = None
    else: 
        profile = False
        
    signal.signal(signal.SIGHUP, on_SIGHUP)
    
    try:
        #main()
        tW = taskWorker()
        tW.loop_forever()
    finally:
        if profile:
            mProfile.report(display=False, profiledir=profileOutDir)
