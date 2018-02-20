import cherrypy
import threading
import requests

import queue as Queue
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('nodeserver')
logger.setLevel(logging.DEBUG)

import time
import sys

from PYME.misc import computerName
from PYME import config
from PYME.IO import clusterIO

from PYME.ParallelTasks import webframework

import ujson as json

WORKER_GET_TIMEOUT = config.get('nodeserver-worker-get-timeout', 60)

#disable socket timeout to prevent us from generating 408 errors
cherrypy.server.socket_timeout = 0

import requests
import multiprocessing

#TODO - should be defined in one place
STATUS_UNAVAILABLE, STATUS_AVAILABLE, STATUS_ASSIGNED, STATUS_COMPLETE, STATUS_FAILED = range(5)


class Rater(object):
    def __init__(self, rule):
        self.rule = rule
        self.taskIDs = rule['availableTaskIDs']
        self.template = rule['taskTemplate']
        
        logger.debug('Template: %s'  % self.template)
        
        self.n = 0
        
        self._non_local = []
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            taskID = self.taskIDs[self.n]
            self.n += 1
        except IndexError:
            raise StopIteration

        task = json.loads(self.template.format(taskID=taskID))
        
        cost = 1.0
        if task['type'] == 'localization':
            filename, serverfilter = clusterIO.parseURL(task['inputs']['frames'])
            filename = '/'.join([filename.lstrip('/'), 'frame%05d.pzf' % int(task['taskdef']['frameIndex'])])
        
            if clusterIO.isLocal(filename, serverfilter):
                cost = .01
    
        elif task['type'] == 'recipe':
            for URL in task['inputs'].values():
                if clusterIO.isLocal(*clusterIO.parseURL(URL)):
                    cost *= .2
                    
        return taskID, cost
    
    def next(self):
        return self.__next__()
    
    
    def get_local(self):
        cost = 1.0
        while cost > 0.99:
            taskID, cost = self.next()
            if cost > 0.99:
                self._non_local.append((self.rule, taskID, cost))
            else:
                return taskID, cost
            
    def get_any(self):
        try:
            taskID, cost = self._non_local.pop()
            return taskID, cost
        except IndexError:
            return self.next()
                
            
        
        
    

        
        

class NodeServer(object):
    def __init__(self, distributor, ip_address, port, nodeID=computerName.GetComputerName()):
        self._tasks = Queue.Queue()
        self._handins = Queue.Queue()

        self.nodeID = nodeID
        self.distributor_url = distributor
        self.ip_address = ip_address
        self.port = port

        self.workerIDs = set()

        self._lastUpdateTime = 0
        #self._lastAnnounceTime = 0
        #self._anounce_url = self.distributor_url + 'distributor/announce?nodeID=%s&ip=%s&port=%d' % (self.nodeID, self.ip_address, self.port)

        cherrypy.engine.subscribe('stop', self.stop)

        self._do_poll = True
        self._update_tasks_lock = threading.Lock()

        self._num_connection_fails = 0

        #set up threads to poll the distributor and announce ourselves and get and return tasks
        self.handinSession = requests.Session()
        self.pollThread = threading.Thread(target=self._poll)
        self.pollThread.start()

        #self.announceSession = requests.Session()
        #self.announceThread = threading.Thread(target=self._announce_loop)
        #self.announceThread.start()

        logger.debug('Starting to poll for tasks')
        self.taskSession = requests.Session()
        self.taskThread = threading.Thread(target=self._poll_tasks)
        self.taskThread.start()



    @property
    def num_tasks_to_request(self):
        return config.get('nodeserver-chunksize', 50) *multiprocessing.cpu_count()
        #return config.get('nodeserver-chunksize', 50)*len(self.workerIDs)
              
        

    def _update_tasks(self):
        """Update our task queue"""
        logger.debug('Updating tasks')
        with self._update_tasks_lock:
            n_tasks_to_request = self.num_tasks_to_request - self._tasks.qsize()
            
            t = time.time()
            if (t - self._lastUpdateTime) < 0.1:
                return

            self._lastUpdateTime = t

            if self.taskSession is None:
                self.taskSession = requests.Session()

            
            try:
                #get adverts
                url = self.distributor_url + 'task_advertisements'
                r = self.taskSession.get(url, timeout=120)
                rules = json.loads(r.content) #r.json()
                
                #decide which tasks to bid on
                n_tasks = 0
                task_requests = []
                raters = [Rater(rule) for rule in rules]
                templates_by_ID = {rule['ruleID']: rule['taskTemplate'] for rule in rules}
                
                #try to get local tasks
                for rater in raters:
                    taskIDs = []
                    costs = []
                    try:
                        while n_tasks < n_tasks_to_request:
                            taskID, cost = rater.get_local()
                            taskIDs.append(taskID)
                            costs.append(cost)
                            n_tasks += 1
                    except StopIteration:
                        pass
                    
                    task_requests.append(dict(ruleID=rater.rule['ruleID'], taskIDs=taskIDs, costs=costs))
                    
                    if n_tasks >= n_tasks_to_request:
                        break
                        
                if n_tasks < n_tasks_to_request:
                    #bid for non-local tasks
                    for rater in raters:
                        taskIDs = []
                        costs = []
                        try:
                            while n_tasks < n_tasks_to_request:
                                taskID, cost = rater.get_any()
                                taskIDs.append(taskID)
                                costs.append(cost)
                                n_tasks += 1
                        except StopIteration:
                            pass
    
                        task_requests.append(dict(ruleID=rater.rule['ruleID'], taskIDs=taskIDs, costs=costs))
    
                        if n_tasks >= n_tasks_to_request:
                            break
                        
                #print task_requests
                
                #place bids and get results
                url = self.distributor_url +'bid_on_tasks'
                r = self.taskSession.get(url, json=task_requests, timeout=120)
                successful_bids = json.loads(r.content)
                
                for bid in successful_bids:
                    ruleID = bid['ruleID']
                    template = templates_by_ID[ruleID]
                    for taskID in bid['taskIDs']:
                        self._tasks.put(template.format(taskID=taskID, ruleID=ruleID))
               
                
            except requests.Timeout:
                logger.warn('Timeout getting tasks from distributor')
                return
            except requests.ConnectionError:
                self.taskSession = None
                self._num_connection_fails += 1
                if self._num_connection_fails < 2:
                    #don't log subsequent failures to avoid filling up the disk
                    logger.error('Error getting tasks: Could not connect to distributor')
                    
                return
                

    def _do_handins(self):
        #TODO - FIXME to make rule-based
        handins = []

        if self.handinSession is None:
            self.handinSession = requests.Session()

        try:
            while True:
                handins.append(self._handins.get_nowait())
        except Queue.Empty:
            pass
        
        

        if len(handins) > 0:
            handins_by_rule = {}
    
            for handin in handins:
                ruleID, taskID = handin['id'].split('~')
                
                if handin['status'] == 'success':
                    status = STATUS_COMPLETE
                elif handin['status'] == 'failure':
                    status = STATUS_FAILED
                else:
                    logger.error('Unknown handin status: %s, ignoring' % status)
                    continue
                
                taskID = int(taskID)
                
                try:
                    h_ruleID = handins_by_rule[ruleID]
                except KeyError:
                    h_ruleID = {'taskIDs' : [], 'status' : []}
                    handins_by_rule[ruleID] = h_ruleID
                    
                h_ruleID['taskIDs'].append(taskID)
                h_ruleID['status'].append(status)
                
        
                
            try:
                r = self.handinSession.post(self.distributor_url + 'handin', json=handins)
                resp = r.json()
                if not resp['ok']:
                    raise RuntimeError('')
            except:
                logger.error('Error handing in tasks')
                self.handinSession = None


    def _poll(self):
        while self._do_poll:
            #self._announce()
            self._do_handins()
            #self._update_tasks()
            time.sleep(.5)

    def _poll_tasks(self):
        while self._do_poll:
            self._update_tasks()
            time.sleep(.1)


    def stop(self):
        self._do_poll = False


    @webframework.register_endpoint('/node/tasks')
    def _get_tasks(self, workerID, numWant=50):
        self.workerIDs.add(workerID)
        #if self._tasks.qsize() < 10:
        #    self._update_tasks()

        t_f = time.time() + WORKER_GET_TIMEOUT
        tasks = []

        try:
            tasks += [self._tasks.get(timeout=WORKER_GET_TIMEOUT)] #wait for at least 1 task
            nTasks = 1

            while (nTasks < min(int(numWant), int(tasks[-1].get('optimal-chunk-size', numWant)))) and (time.time() < t_f):
                tasks.append(self._tasks.get_nowait())
                nTasks += 1
        except Queue.Empty:
            pass

        logging.debug('Giving %d tasks to %s' % (len(tasks), workerID))

        return json.dumps({'ok': True, 'result': tasks})



    @webframework.register_endpoint('/node/handin')
    def _handin(self, taskID, status):
        self._handins.put({'taskID': taskID, 'status':status})
        return json.dumps({'ok' : True})

    @webframework.register_endpoint('/node/status')
    def _status(self):
        return json.dumps({'Polling' : self._do_poll, 'nQueued' : self._tasks.qsize()})\




class WFNodeServer(webframework.APIHTTPServer, NodeServer):
    def __init__(self, distributor, ip_address, port, nodeID=computerName.GetComputerName()):
        NodeServer.__init__(self, distributor, ip_address, port, nodeID=computerName.GetComputerName())

        server_address = ('', port)
        webframework.APIHTTPServer.__init__(self, server_address)
        self.daemon_threads = True





def run(distributor, port):
    import socket

    externalAddr = socket.gethostbyname(socket.gethostname())
    nodeserver = WFNodeServer('http://' + distributor + '/', port = port, ip_address=externalAddr)

    try:
        logger.info('Starting nodeserver on %s:%d' % (externalAddr, port))
        nodeserver.serve_forever()
    finally:
        nodeserver._do_poll = False
        logger.info('Shutting down ...')
        nodeserver.shutdown()
        nodeserver.server_close()



if __name__ == '__main__':
    distributor, port = sys.argv[1:]
    run(distributor, int(port))
