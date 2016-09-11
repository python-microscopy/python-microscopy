import cherrypy
import threading
import requests
import Queue
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('distributor')
logger.setLevel(logging.DEBUG)

import time
import sys
import json

#from PYME.misc import computerName
from PYME import config
#from PYME.IO import clusterIO
import collections

NODE_TIMEOUT = config.get('distributor-node_registration_timeout', 10)
PROCESS_TIMEOUT = config.get('distributor-execution_timeout', 60)
RATE_TIMEOUT = config.get('distributor-task_rating_timeout', 10)
NUM_TO_RATE = config.get('distributor-rating_chunk_size', 50)

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
        self.pollThread = threading.Thread(target=self._do_poll)
        self.pollThread.start()

    def info(self):
        return {'tasksPosted': self.total_num_tasks,
                  'tasksRunning': len(self.assigned),
                  'tasksCompleted': self.num_tasks_completed,
                  'tasksFailed' : self.num_tasks_failed,
                  'averageExecutionCost' : 1.0,
                }

    def stop(self):
        self._do_poll = False

    def posttask(self, task):
        self.rating_queue.append(task)
        self.total_num_tasks += 1

    def handin(self, handins):
        for h in handins:
            self.handins.append(h)

    def _getForRating(self, numToRate = 50):
        tasks = []
        try:
            for i in xrange(numToRate):
                t = self.rating_queue.popleft()
                self.ratings_in_progress[t.id] = TaskInfo(t, RATE_TIMEOUT)
                tasks.append(t)
        except IndexError:
            pass
        return tasks

    def _rate_tasks(self, tasks, node, rated_queue):
        server = self.distributor.nodes[node]
        r = requests.post('http://%s:%d/node/rate' % (server['ip'], server['port']), json=tasks, timeout=RATE_TIMEOUT)
        resp = r.json()
        if resp['ok']:
            rated_queue.append(node, resp['results'])

    def _rateAndAssignTasks(self):
        tasks = self._getForRating(NUM_TO_RATE)
        rated_queue = collections.deque()
        r_threads = [threading.Thread(target=self._rate_tasks, args=(tasks, node, rated_queue)) for node in self.distributor.nodes.keys()]
        for t in r_threads: t.start()
        for t in r_threads: t.join(timeout=RATE_TIMEOUT)

        costs = collections.OrderedDict()
        min_cost = collections.OrderedDict()

        for node, ratings in rated_queue:
            for rating in ratings:
                id = rating['id']
                cost = rating['cost']
                if cost < costs.get(id, 900000):
                    cost[id] = cost
                    min_cost[id] = node

        #assign our rated items to a node
        for id, node in min_cost.items():
            self.num_rated += 1
            self.total_cost += min_cost[id]
            t = TaskInfo(self.ratings_in_progress.pop(id).task, PROCESS_TIMEOUT)
            self.assigned[id] = t
            self.distributor.nodes[node]['taskQueue'].put(t)

        #push all the unassigned items to the back into the rating queue
        try:
            while True:
                t = self.ratings_in_progress.popitem(False)
                self.rating_queue.append(t.task)
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
        except IndexError:
            pass

    def _process_handins(self):
        try:
            while True:
                h = self.handins.popleft()

                try:
                    t = self.assigned.pop(h['id'])

                    if h['status'] == 'success':
                        self.num_tasks_completed += 1
                    elif h['status'] == 'failure':
                        self.failure_counts[h['id']] = self.failure_counts.get(h['id'], 0) + 1

                        if self.failure_counts[h['id']] < 5:
                            self.rating_queue.append(t.task)
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
                handins = []
                try:
                    while True:
                        task = q.get_nowait()
                        handins.append( {'id': task['id'], 'status' : 'notExecuted'})
                except Queue.Empty:
                    pass

                self.handin(handins)

    def _poll(self):
        while self._do_poll:
            self._update_nodes()
            time.sleep(1)

    def stop(self):
        self._do_poll = False

        for queue in self._queues:
            queue.stop()


    @cherrypy.expose
    @cherrypy.tools.json_out()
    #@cherrypy.tools.json_in()
    def tasks(self, queue=None, nodeID=None, numWant=50, timeout=5):
        if cherrypy.request.method == 'GET':
            return self._get_tasks(nodeID, int(numWant), float(timeout))
        elif cherrypy.request.method == 'POST':
            return self._post_tasks(queue)

    def _get_tasks(self, nodeID, numWant, timeout):
        tasks = []

        t = time.time()
        t_finish = t + timeout

        nTasks = 0
        try:
            while (nTasks < numWant) and (t < t_finish):
                tasks.append(self.nodes[nodeID]['taskQueue'].get(timeout = (t_finish - t)))
                t = time.time()
        except Queue.Empty:
            pass

        logger.debug('Gave %d tasks to %s' % (len(tasks), nodeID))

        return {'ok': True, 'result': tasks}

    def _post_tasks(self, queue):
        try:
            q = self._queues[queue]
        except KeyError:
            q = TaskQueue(self)
            self._queues[queue] = q

        tasks = json.loads(cherrypy.request.body.read())
        for task in tasks:
            task['id'] = queue + '-' + task['id']
            q.posttask(task)

        logger.debug('accepted %d tasks' % len(tasks))

        return {'ok' : True}




    @cherrypy.expose
    #@cherrypy.tools.json_in()
    #@cherrypy.tools.json_out()
    def handin(self, nodeID):
        logger.debug('Handing in tasks...')
        for handin in json.loads(cherrypy.request.body.read()):
            queue = handin.split('-')
            self._queues[queue].handin(handin)
        return #{'ok': True}

    @cherrypy.expose
    def announce(self, nodeID, ip, port):
        try:
            node = self.nodes[nodeID]
        except KeyError:
            node = {'taskQueue': Queue.Queue()}
            self.nodes[nodeID] = node

        node.update({'ip': ip, 'port': port, 'expiry' : time.time() + NODE_TIMEOUT})

        #logging.debug('Got announcement from %s' % nodeID)

        return

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def queues(self):
        #logging.debug('queues: ' + repr(self._queues))
        return {'ok': True, 'result': {qn: q.info() for qn, q in self._queues.items()}}




def run(port):
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

    distributor = Distributor()

    app = cherrypy.tree.mount(distributor, '/distributor/')
    app.log.access_log.setLevel(logging.ERROR)

    try:

        cherrypy.quickstart()
    finally:
        distributor._do_poll = False


if __name__ == '__main__':
    port = sys.argv[1]
    run(int(port))
