import cherrypy
import threading
import requests
import queue as Queue
from six.moves import xrange
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger('ruleserver')
logger.setLevel(logging.DEBUG)

import time
import sys
import ujson as json
#import json
import os

from PYME.misc import computerName
from PYME import config
#from PYME.IO import clusterIO
from PYME.ParallelTasks import webframework
import collections

import uuid

import numpy as np

class Rule(object):
    pass

STATUS_UNAVAILABLE, STATUS_AVAILABLE, STATUS_ASSIGNED, STATUS_COMPLETE, STATUS_FAILED = range(5)

class IntegerIDRule(Rule):
    TASK_INFO_DTYPE = np.dtype([('status', 'uint8'), ('nRetries', 'uint8'), ('expiry', 'f4'), ('cost', 'f4')])
    
    
    def __init__(self, ruleID, task_template, inputs_by_task = None,
                 max_task_ID=100000, task_timeout=600, rule_timeout=3600):
        self.ruleID = ruleID
        self._inputs_by_task = inputs_by_task
        self._template = task_template
        self._task_info = np.zeros(max_task_ID, self.TASK_INFO_DTYPE)
        
        self._n_retries =  config.get('ruleserver-retries', 3)
        self._timeout = task_timeout
        
        self._rule_timeout = rule_timeout
        self._cached_advert = None
        
        self.expiry = time.time() + self._rule_timeout
              
        self._info_lock = threading.Lock()
        self._advert_lock = threading.Lock()
        
    def make_range_available(self, start, end):
        '''Make a range of tasks available (to be called once the underlying data is available)'''
        
        if start < 0 or start >= self._task_info.size or end < 0 or end >= self._task_info.size:
            raise RuntimeError('Range (%d, %d) invalid with maxTasks=self._task_info.size' % (start, end))
        
        #TODO - check existing status - it probably makes sense to only apply this to tasks which have STATUS_UNAVAILABLE
        with self._info_lock:
            self._task_info['status'][start:end] = STATUS_AVAILABLE

        self.expiry = time.time() + self._rule_timeout
        
        with self._advert_lock:
            self._cached_advert = None
            
    def bid(self, bid):
        '''Bid on tasks (and return any that match). Note the current implementation is very naive and expects doesn't
        check bid cost
        
        Parameters
        ----------
        
        bid : dictionary containing the following items: ruleID, taskIDs, taskCosts
        
        '''
        
        taskIDs = np.array(bid['taskIDs'], 'i')
        costs = np.array(bid['costs'], 'i')
        with self._info_lock:
            successful_bid_mask = self._task_info['status'][taskIDs] == STATUS_AVAILABLE
            successful_bid_ids = taskIDs[successful_bid_mask]
            self._task_info['status'][successful_bid_ids] = STATUS_ASSIGNED
            self._task_info['cost'][successful_bid_ids] = costs[successful_bid_mask]
            self._task_info['expiry'][successful_bid_ids] = time.time() + self._timeout

        self.expiry = time.time() + self._rule_timeout
        
        with self._advert_lock:
            self._cached_advert = None
            
        return {'ruleID': bid['ruleID'], 'taskIDs':successful_bid_ids.tolist(), 'template' : self._template}
    
    def mark_complete(self, info):
        taskIDs = np.array(info['taskIDs'], 'i')
        status = np.array(info['status'], 'uint8')
        
        with self._info_lock:
            self._task_info['status'][taskIDs] = status

        self.expiry = time.time() + self._rule_timeout
            
    @property
    def advert(self):
        with self._advert_lock:
            if not self._cached_advert:
                availableTasks = np.where(self._task_info['status'] == STATUS_AVAILABLE)[0].tolist()
                self._cached_advert = {'ruleID' : self.ruleID,
                    'taskTemplate': self._template,
                    'availableTaskIDs': availableTasks}
                
                if self._inputs_by_task:
                    self._cached_advert['inputsByTask'] = {taskID: self._inputs_by_task[taskID] for taskID in availableTasks}
                
            return self._cached_advert
    
    @property
    def nAvailable(self):
        return (self._task_info['status'] == STATUS_AVAILABLE).sum()
    
    @property
    def nAssigned(self):
        return (self._task_info['status'] == STATUS_ASSIGNED).sum()

    @property
    def nCompleted(self):
        return (self._task_info['status'] == STATUS_COMPLETE).sum()

    @property
    def nFailed(self):
        return (self._task_info['status'] == STATUS_FAILED).sum()

    @property
    def nTotal(self):
        return (self._task_info['status'] > 0).sum()
    
    @property
    def expired(self):
        return (self.nAvailable == 0) and (self.nAssigned == 0) and (time.time() > self.expiry)
    
    def info(self):
        av_cost = np.mean(self._task_info['cost'][self._task_info['status']>STATUS_AVAILABLE])
        if np.isnan(av_cost):
            av_cost = 0
        else:
            av_cost = float(av_cost)
            
        return {'tasksPosted': self.nTotal,
                  'tasksRunning': self.nAssigned,
                  'tasksCompleted': self.nCompleted,
                  'tasksFailed' : self.nFailed,
                  'averageExecutionCost' : av_cost,
                }
    
    def poll_timeouts(self):
        t = time.time()
        
        with self._info_lock:
            timed_out = np.where((self._task_info['status'] == STATUS_ASSIGNED)*(self._task_info['expiry'] < t))[0]
            
            self._task_info['status'][timed_out] = STATUS_AVAILABLE
            self._task_info['nRetries'][timed_out] += 1

            self._task_info['status'][self._task_info['nRetries'] > self._n_retries] = STATUS_FAILED
            
            with self._advert_lock:
                self._cached_advert = None
        
        
    
        


class RuleServer(object):
    MAX_ADVERTISEMENTS = 5 * 10 * 50 * 12 #only advertise enough for 100 tasks on each core of each cluster node
    def __init__(self):
        self._rules = collections.OrderedDict()
        
        cherrypy.engine.subscribe('stop', self.stop)
        
        self._do_poll = True
        
        self._queueLock = threading.Lock()
        
        self._advert_lock = threading.Lock()
        
        #set up threads to poll the distributor and announce ourselves and get and return tasks
        #self.pollThread = threading.Thread(target=self._poll)
        #self.pollThread.start()
        
        self._cached_advert = None
        self._cached_advert_expiry = 0
        
        
        self._info_lock = threading.Lock()
        self._cached_info = None
        self._cached_info_expiry = 0
        self._cached_info_timeout = 5
        
        self._rule_n = 0
        
        self.rulePollThread = threading.Thread(target=self._poll_rules)
        self.rulePollThread.start()
    
    
    def _poll(self):
        while self._do_poll:
            time.sleep(1)
    
    def _poll_rules(self):
        while self._do_poll:
            for qn in self._rules.keys():
                self._rules[qn].poll_timeouts()
                
                #remore queue if expired (no activity for an hour) to free up memory
                if self._rules[qn].expired:
                    self._rules.pop(qn)
            
            time.sleep(5)
    
    def stop(self):
        self._do_poll = False
        
        #for queue in self._queues.values():
        #    queue.stop()
            
    @webframework.register_endpoint('/task_advertisements')
    def _advertised_tasks(self):
        with self._advert_lock:
            t = time.time()
            if (t > self._cached_advert_expiry):
                adverts = []
                nTasks = 0
                ruleN = 0
                
                rules = self._rules.values()
                while ruleN < len(rules) and nTasks < self.MAX_ADVERTISEMENTS:
                    advert = rules[ruleN].advert
                    adverts.append(advert)
                    nTasks += len(advert['availableTaskIDs'])
                    ruleN += 1
                    
                self._cached_advert = json.dumps(adverts)
                self._cached_advert_expiry = time.time() + 1 #regenerate advert once every second
            
        return self._cached_advert
        
        
    @webframework.register_endpoint('/bid_on_tasks')
    def _bid_on_tasks(self, body=''):
        bids = json.loads(body)
        
        succesfull_bids = []
        
        for bid in bids:
            rule = self._rules[bid['ruleID']]
            
            succesfull_bids.append(rule.bid(bid))
            #task_ids = bid['taskIDs']
            #costs = bid['taskCosts']
            
        #print(succesfull_bids)
        return json.dumps(succesfull_bids)
        
            
        
    @webframework.register_endpoint('/add_integer_id_rule')
    def _add_rule(self, max_tasks= 1e6, release_start=None, release_end = None, ruleID = None, timeout=3600, body=''):
        rule_info = json.loads(body)
        
        if ruleID is None:
            ruleID = '%06d-%s' % (self._rule_n, uuid.uuid4().get_hex())
        
        rule = IntegerIDRule(ruleID, rule_info['template'], max_task_ID=int(max_tasks), rule_timeout=timeout)
        if not release_start is None:
            rule.make_range_available(release_start, release_end)
        
        self._rules[ruleID] = rule
        
        self._rule_n += 1
        
        return json.dumps({'ok': 'True', 'ruleID' : ruleID})

    @webframework.register_endpoint('/release_rule_tasks')
    def _release_rule_tasks(self, ruleID, release_start, release_end, body=''):
        rule = self._rules[ruleID]
    
        rule.make_range_available(int(release_start), int(release_end))
    
    
        return json.dumps({'ok': 'True'})
    
    @webframework.register_endpoint('/handin')
    def _handin(self, body):
        
        #logger.debug('Handing in tasks...')
        for handin in json.loads(body):
            ruleID = handin['ruleID']
            
            rule = self._rules[ruleID]
            
            rule.mark_complete(handin)
            
        return json.dumps({'ok': True})
    
    
    
    @webframework.register_endpoint('/distributor/queues')
    def _get_queues(self):
        with self._info_lock:
            t = time.time()
            if (t > self._cached_info_expiry):
                self._cached_info = json.dumps({'ok': True, 'result': {qn: self._rules[qn].info() for qn in self._rules.keys()}})
                self._cached_info_expiry = time.time() + self._cached_info_timeout
                
        return self._cached_info


# class CPRuleServer(RuleServer):
#     @cherrypy.expose
#     def add_integer_rule(self, queue=None, nodeID=None, numWant=50, timeout=5):
#         cherrypy.response.headers['Content-Type'] = 'application/json'
#
#         body = ''
#         #if cherrypy.request.method == 'GET':
#
#         if cherrypy.request.method == 'POST':
#             body = cherrypy.request.body.read()
#
#         return self._tasks(queue, nodeID, numWant, timeout, body)
#
#     @cherrypy.expose
#     def handin(self, nodeID):
#         cherrypy.response.headers['Content-Type'] = 'application/json'
#
#         body = cherrypy.request.body.read()
#
#         return self._handin(nodeID, body)
#
#     @cherrypy.expose
#     def announce(self, nodeID, ip, port):
#         cherrypy.response.headers['Content-Type'] = 'application/json'
#
#         self._announce(nodeID, ip, port)
#
#     @cherrypy.expose
#     def queues(self):
#         cherrypy.response.headers['Content-Type'] = 'application/json'
#
#         return self._get_queues()


class WFRuleServer(webframework.APIHTTPServer, RuleServer):
    def __init__(self, port):
        RuleServer.__init__(self)
        
        server_address = ('', port)
        webframework.APIHTTPServer.__init__(self, server_address)
        self.daemon_threads = True


# def runCP(port):
#     import socket
#     cherrypy.config.update({'server.socket_port': port,
#                             'server.socket_host': '0.0.0.0',
#                             'log.screen': False,
#                             'log.access_file': '',
#                             'log.error_file': '',
#                             'server.thread_pool': 50,
#                             })
#
#     logging.getLogger('cherrypy.access').setLevel(logging.ERROR)
#
#     #externalAddr = socket.gethostbyname(socket.gethostname())
#
#     distributor = CPRuleServer()
#
#     app = cherrypy.tree.mount(distributor, '/distributor/')
#     app.log.access_log.setLevel(logging.ERROR)
#
#     try:
#
#         cherrypy.quickstart()
#     finally:
#         distributor._do_poll = False


import threading

class ServerThread(threading.Thread):
    def __init__(self, port, profile=False):
        self.port = int(port)
        self._profile = profile
        threading.Thread.__init__(self)
        
    def run(self):
        import socket
        if self._profile:
            from PYME.util import mProfile
        
            mProfile.profileOn(['ruleserver.py', ])
            profileOutDir = config.get('dataserver-root', os.curdir) + '/LOGS/%s/mProf' % computerName.GetComputerName()
        
        self.externalAddr = socket.gethostbyname(socket.gethostname())
        self.distributor = WFRuleServer(self.port)

        logger.info('Starting ruleserver on %s:%d' % (self.externalAddr, self.port))
        try:
            self.distributor.serve_forever()
        finally:
            self.distributor._do_poll = False
            #logger.info('Shutting down ...')
            #self.distributor.shutdown()
            logger.info('Closing server ...')
            self.distributor.server_close()

            if self._profile:
                mProfile.report(False, profiledir=profileOutDir)
            
    
    def shutdown(self):
        self.distributor._do_poll = False
        logger.info('Shutting down ...')
        self.distributor.shutdown()
        logger.info('Closing server ...')
        self.distributor.server_close()


def on_SIGHUP(signum, frame):
    from PYME.util import mProfile
    mProfile.report(False, profiledir=profileOutDir)
    raise RuntimeError('Recieved SIGHUP')


if __name__ == '__main__':
    import signal
    
    port = sys.argv[1]
    
    if (len(sys.argv) == 3) and (sys.argv[2] == '-k'):
        profile = True
        from PYME.util import mProfile
        
        mProfile.profileOn(['ruleserver.py', ])
        profileOutDir = config.get('dataserver-root', os.curdir) + '/LOGS/%s/mProf' % computerName.GetComputerName()
    else:
        profile = False
        profileOutDir = None
    
    if not sys.platform == 'win32':
        #windows doesn't support handling signals ... don't catch and hope for the best.
        #Note: This will make it hard to cleanly shutdown the distributor on Windows, but should be OK for testing and
        #development
        signal.signal(signal.SIGHUP, on_SIGHUP)
    
    try:
        run(int(port))
    finally:
        if profile:
            mProfile.report(False, profiledir=profileOutDir)
        
