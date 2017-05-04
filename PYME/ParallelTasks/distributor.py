import cherrypy
import threading
import requests
import queue as Queue
from six.moves import xrange
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger('distributor')
logger.setLevel(logging.DEBUG)

import time
import sys
import ujson as json
import os

from PYME.misc import computerName
from PYME import config
#from PYME.IO import clusterIO
from PYME.ParallelTasks import webframework
import collections

NODE_TIMEOUT = config.get('distributor-node_registration_timeout', 10)
PROCESS_TIMEOUT = config.get('distributor-execution_timeout', 300)
RATE_TIMEOUT = config.get('distributor-task_rating_timeout', 10)
NUM_TO_RATE = config.get('distributor-rating_chunk_size', 1000)

class TaskInfo(object):
    def __init__(self, task, timeout):
        self.task = task
        self.expiry = time.time() + timeout

class TaskQueue(object):
    def __init__(self, distributor):
        self.rating_queue = collections.deque()
        self.ratings_in_progress =  collections.OrderedDict()
        self.assigned =  collections.OrderedDict()
        self.total_num_tasks =  0
        self.num_tasks_completed =  0
        self.num_tasks_failed = 0

        self.num_rated = 0
        self.total_cost = 0

        self.handins = collections.deque()

        self.failure_counts = {}

        self._lock = threading.Lock()
        self.distributor = distributor

        self._do_poll = True
        self.pollThread = threading.Thread(target=self._poll_loop)
        self.pollThread.start()

    def info(self):
        return {'tasksPosted': self.total_num_tasks,
                  'tasksRunning': len(self.assigned),
                  'tasksCompleted': self.num_tasks_completed,
                  'tasksFailed' : self.num_tasks_failed,
                  'averageExecutionCost' : self.total_cost/(self.num_rated + .01),
                }

    def stop(self):
        self._do_poll = False

    def posttask(self, task):
        self.rating_queue.append(task)
        self.total_num_tasks += 1

    def handin(self, h):
        self.handins.append(h)

    def _getForRating(self, numToRate = 50):
        tasks = []
        try:
            for i in xrange(numToRate):
                t = self.rating_queue.popleft()
                self.ratings_in_progress[t['id']] = TaskInfo(t, RATE_TIMEOUT)
                tasks.append(t)
        except IndexError:
            pass
        return tasks

    def _rate_tasks(self, tasks, node, rated_queue):
        import zlib
        server = self.distributor.nodes[node]
        url = 'http://%s:%d/node/rate' % (server['ip'], int(server['port']))
        #logger.debug('Requesting rating from %s' % url)
        try:
            r = requests.post(url, data=zlib.compress(tasks),
                          headers={'Content-Type': 'application/json', 'Content-Encoding' : 'gzip'}, timeout=RATE_TIMEOUT)
            resp = r.json()
            if resp['ok']:
                rated_queue.append((node, resp['result']))

            #logger.debug('Ratings returned from %s' % url)
        except requests.ConnectionError:
            pass #fail silently for now TODO - unsubscribe node

    def _rateAndAssignTasks(self):
        tasks = self._getForRating(NUM_TO_RATE)
        if len(tasks) > 0:
            rated_queue = collections.deque()
            task_list = json.dumps(tasks)
            r_threads = [threading.Thread(target=self._rate_tasks, args=(task_list, node, rated_queue)) for node in self.distributor.nodes.keys()]

            #logger.debug('Asking nodes for rating')
            for t in r_threads: t.start()
            for t in r_threads: t.join(timeout=RATE_TIMEOUT)
            #logger.debug('Ratings returned')

            costs = collections.OrderedDict()
            min_cost = collections.OrderedDict()

            for node, ratings in rated_queue:
                for rating in ratings:
                    id = rating['id']
                    cost = float(rating['cost'])
                    if cost < costs.get(id, 900000):
                        costs[id] = cost
                        min_cost[id] = node

            #logger.debug('%d ratings returned' % len(min_cost))
            #assign our rated items to a node
            for id, node in min_cost.items():
                self.num_rated += 1
                self.total_cost += costs[id]
                t = TaskInfo(self.ratings_in_progress.pop(id).task, PROCESS_TIMEOUT)
                self.assigned[id] = t
                self.distributor.nodes[node]['taskQueue'].append(t)

            #logger.debug('%d total tasks rated' % self.num_rated)

        #push all the unassigned items to the back into the rating queue
        try:
            while True:
                id, t = self.ratings_in_progress.popitem(False)
                self.rating_queue.append(t.task)
                #logger.debug('Resubmitting unrated task for rating')
        except KeyError:
            pass

    def _requeue_timed_out(self):
        t = time.time()
        try:
            id = self.assigned.keys()[-1]
            while self.assigned[id].expiry < t:
                task = self.assigned.pop(id).task
                self.rating_queue.append(task)
                id = self.assigned.keys()[-1]
                logger.debug('Task timed out, resubmitting')
        except IndexError:
            pass

    def _process_handins(self):
        try:
            while True:
                h = self.handins.popleft()

                try:
                    t = self.assigned.pop(h['taskID'])

                    if h['status'] == 'success':
                        self.num_tasks_completed += 1
                    elif h['status'] == 'failure':
                        self.failure_counts[h['taskID']] = self.failure_counts.get(h['taskID'], 0) + 1

                        if self.failure_counts[h['taskID']] < 5:
                            self.rating_queue.append(t.task)
                            logger.debug('task failed, retrying ... ')
                        else:
                            self.num_tasks_failed += 1
                            logger.error('task failed > 5 times, discarding')
                    else: #didn't process
                        self.rating_queue.append(t.task)

                except KeyError:
                    logger.error('Trying to hand in unasigned task')

        except IndexError:
            pass


    def _poll_loop(self):
        while self._do_poll:
            self._rateAndAssignTasks()
            self._process_handins()
            self._requeue_timed_out()

            time.sleep(.1)




class Distributor(object):
    def __init__(self):
        self._queues = {}

        self.nodes = {}

        cherrypy.engine.subscribe('stop', self.stop)

        self._do_poll = True

        self._queueLock = threading.Lock()

        #set up threads to poll the distributor and announce ourselves and get and return tasks
        self.pollThread = threading.Thread(target=self._poll)
        self.pollThread.start()


    def _update_nodes(self):
        t = time.time()
        for nodeID, node in self.nodes.items():
            if node['expiry'] < t:
                self.nodes.pop(nodeID)
                logger.info('unsubscribing %s and reassigning tasks' % nodeID)

                q = node['taskQueue']

                try:
                    while True:
                            task = q.popleft().task
                            handin = {'taskID': task['id'], 'status' : 'notExecuted'}
                            queue = handin['taskID'].split('-')[0]
                            self._queues[queue].handin(handin)
                except IndexError:
                    pass


    def _poll(self):
        while self._do_poll:
            self._update_nodes()
            time.sleep(1)

    def stop(self):
        self._do_poll = False

        for queue in self._queues.values():
            queue.stop()



    @webframework.register_endpoint('/distributor/tasks')
    def _tasks(self, queue=None, nodeID=None, numWant=50, timeout=5, body=''):
        if len(body) == 0:
            return json.dumps(self._get_tasks(nodeID, int(numWant), float(timeout)))
        else:
            return json.dumps(self._post_tasks(queue, body))

    def _get_tasks(self, nodeID, numWant, timeout):
        tasks = []

        t = time.time()
        t_finish = t + timeout

        nTasks = 0

        while (nTasks < numWant) and (t < t_finish):
            try:
                tasks.append(self.nodes[nodeID]['taskQueue'].popleft().task)
            except IndexError:
                time.sleep(.2)
                pass
            except KeyError:
                logger.debug('tasks requested from unknown node: %s' % nodeID)
                return {'ok' : False}
            t = time.time()


        if nTasks > 0:
            logger.debug('Gave %d tasks to %s' % (len(tasks), nodeID))

        return {'ok': True, 'result': tasks}

    def _post_tasks(self, queue, body):
        with self._queueLock:
            try:
                q = self._queues[queue]
            except KeyError:
                q = TaskQueue(self)
                self._queues[queue] = q

        tasks = json.loads(body)
        for task in tasks:
            task['id'] = queue + '-' + task['id']
            q.posttask(task)

        logger.debug('accepted %d tasks' % len(tasks))

        return {'ok' : True}


    @webframework.register_endpoint('/distributor/handin')
    def _handin(self, nodeID, body):

        #logger.debug('Handing in tasks...')
        for handin in json.loads(body):
            queue = handin['taskID'].split('-')[0]
            self._queues[queue].handin(handin)
        return json.dumps({'ok': True})

    @webframework.register_endpoint('/distributor/announce')
    def _announce(self, nodeID, ip, port):
        try:
            node = self.nodes[nodeID]
        except KeyError:
            node = {'taskQueue': collections.deque()}
            self.nodes[nodeID] = node

        node.update({'ip': ip, 'port': port, 'expiry' : time.time() + NODE_TIMEOUT})

        #logging.debug('Got announcement from %s' % nodeID)

        return json.dumps({'ok': True})

    @webframework.register_endpoint('/distributor/queues')
    def _get_queues(self):
        return json.dumps({'ok': True, 'result': {qn: q.info() for qn, q in self._queues.items()}})



class CPDistributor(Distributor):
    @cherrypy.expose
    def tasks(self, queue=None, nodeID=None, numWant=50, timeout=5):
        cherrypy.response.headers['Content-Type'] = 'application/json'

        body = ''
        #if cherrypy.request.method == 'GET':

        if cherrypy.request.method == 'POST':
            body = cherrypy.request.body.read()

        return self._tasks(queue, nodeID, numWant, timeout, body)

    @cherrypy.expose
    def handin(self, nodeID):
        cherrypy.response.headers['Content-Type'] = 'application/json'

        body = cherrypy.request.body.read()

        return self._handin(nodeID, body)

    @cherrypy.expose
    def announce(self, nodeID, ip, port):
        cherrypy.response.headers['Content-Type'] = 'application/json'

        self._announce(nodeID, ip, port)

    @cherrypy.expose
    def queues(self):
        cherrypy.response.headers['Content-Type'] = 'application/json'

        return self._get_queues()

class WFDistributor(webframework.APIHTTPServer, Distributor):
    def __init__(self, port):
        Distributor.__init__(self)

        server_address = ('', port)
        webframework.APIHTTPServer.__init__(self, server_address)
        self.daemon_threads = True

def runCP(port):
    import socket
    cherrypy.config.update({'server.socket_port': port,
                            'server.socket_host': '0.0.0.0',
                            'log.screen': False,
                            'log.access_file': '',
                            'log.error_file': '',
                            'server.thread_pool': 50,
                            })

    logging.getLogger('cherrypy.access').setLevel(logging.ERROR)

    #externalAddr = socket.gethostbyname(socket.gethostname())

    distributor = CPDistributor()

    app = cherrypy.tree.mount(distributor, '/distributor/')
    app.log.access_log.setLevel(logging.ERROR)

    try:

        cherrypy.quickstart()
    finally:
        distributor._do_poll = False


def run(port):
    import socket

    externalAddr = socket.gethostbyname(socket.gethostname())
    distributor = WFDistributor(port)

    try:
        logger.info('Starting distributor on %s:%d' % (externalAddr, port))
        distributor.serve_forever()
    finally:
        distributor._do_poll = False
        logger.info('Shutting down ...')
        distributor.shutdown()
        distributor.server_close()


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
        mProfile.profileOn(['distributor.py',])
        profileOutDir = config.get('dataserver-root', os.curdir) + '/LOGS/%s/mProf' % computerName.GetComputerName()
    else:
        profile = False
        profileOutDir=None

    signal.signal(signal.SIGHUP, on_SIGHUP)

    try:
        run(int(port))
    finally:
        if profile:
            mProfile.report(False, profiledir=profileOutDir)


