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

from PYME.misc import computerName
from PYME import config
from PYME.IO import clusterIO
import collections

NODE_TIMEOUT = config.get('distributor-node_registration_timeout', 10)
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

        self._lock = threading.Lock()
        self.distributor = distributor

    def info(self):
        return {'tasksPosted': self.total_num_tasks,
                  'tasksRunning': len(self.assigned),
                  'tasksCompleted': self.num_tasks_completed,
                  'tasksFailed' : self.num_tasks_failed,
                  'averageExecutionCost' : 1.0,
                }

    def _getForRating(self, numToRate = 50):
        tasks = []
        try:
            for i in xrange(50):
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

    def rateTasks(self):
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
                if cost < costs.get('id', 900000):
                    cost['id'] = cost
                    min_cost['id'] = node

        for id, node in min_cost.items():
            t = self.ratings_in_progress.pop(id)
            self.distributor.nodes[node]['taskQueue'].put(t)








class Distributor(object):
    def __init__(self):
        self._tasks = Queue.Queue()
        self._handins = Queue.Queue()

        #self.nodeID = nodeID
        #self.distributor_url = distributor
        #self.ip_address = ip_address
        #self.port = port
        self.queues = {}

        self.nodes = {}
        self._rating_queue = Queue.Queue()


        cherrypy.engine.subscribe('stop', self.stop)

        self._do_poll = True

        #set up threads to poll the distributor and announce ourselves and get and return tasks
        self.pollThread = threading.Thread(target=self._poll)
        self.pollThread.start()

        self.announceThread = threading.Thread(target=self._announce_loop)
        self.announceThread.start()

        self.taskThread = threading.Thread(target=self._poll_tasks)
        self.taskThread.start()




    def _update_tasks(self):
        """Update our task queue"""
        t = time.time()
        if (t - self._lastUpdateTime) < 0.1:
            return

        self._lastUpdateTime = t

        url = self.distributor_url + 'distributor/tasks?nodeID=%s&numWant=%d&timeout=10' % (self.nodeID, self.num_tasks_to_request)
        try:
            r = requests.get(url, timeout=10)
            resp = r.json()
            if resp['ok']:
                for task in resp['result']:
                    self._tasks.put(task)
        except requests.Timeout:
            pass
        except requests.ConnectionError:
            logger.error('Could not connect to distributor')


    def _poll(self):
        while self._do_poll:
            #self._announce()
            self._do_handins()
            #self._update_tasks()
            time.sleep(1)

    def _poll_tasks(self):
        while self._do_poll:
            self._update_tasks()
            time.sleep(1)

    def _announce_loop(self):
        while self._do_poll:
            self._announce()
            time.sleep(1)

    def stop(self):
        self._do_poll = False


    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def tasks(self, queue=None, nodeID=None, numWant=50, timeout=5):
        if cherrypy.request.method == 'GET':
            return self._get_tasks(nodeID, numWant, timeout)
        elif cherrypy.request.method == 'POST':
            return self._post_tasks(queue)

    def _get_tasks(self, nodeID, numWant, timeout):
        tasks = []

        t = time.time()
        t_finish = t + timeout

        nTasks = 0
        while (nTasks < numWant) and (t < t_finish):
            tasks.append(self.nodes[nodeID]['taskQueue'].get(timeout = (t_finish - t)))
            t = time.time()

        return {'ok': True, 'result': tasks}

    def _post_tasks(self, queue):
        try:
            q = self.queues[queue]
        except KeyError:
            q = TaskQueue()
            self.queues[queue] = q

        for task in cherrypy.request.json:
            q['rating_queue'].put(task)
            q['num_tasks'] += 1

        return {'ok' : True}




    @cherrypy.expose
    @cherrypy.tools.json_in()
    #@cherrypy.tools.json_out()
    def handin(self, nodeID):
        for handin in cherrypy.request.json:
            self._handins.put(handin)
        return #{'ok': True}

    @cherrypy.expose
    def announce(self, nodeID, ip, port):
        try:
            node = self.nodes[nodeID]
        except KeyError:
            node = {'taskQueue': Queue.Queue()}
            self.nodes[nodeID] = node

        node.update({'ip': ip, 'port': port, 'expiry' : time.time() + NODE_TIMEOUT})

    @cherrpy.expose
    @cherrypy.tools.json_out()
    def queues(self):
        return [q.info() for q in self.queues]


def run(distributor, port):
    import socket
    cherrypy.config.update({'server.socket_port': port,
                            'server.socket_host': '0.0.0.0',
                            'log.screen': False,
                            'log.access_file': '',
                            'log.error_file': '',
                            'server.thread_pool': 50,
                            })

    logging.getLogger('cherrypy.access').setLevel(logging.ERROR)

    externalAddr = socket.gethostbyname(socket.gethostname())

    nodeserver = NodeServer('http://' + distributor + '/', port = port, ip_address=externalAddr)

    try:

        cherrypy.quickstart(nodeserver, '/node/')
    finally:
        nodeserver._do_poll = False


if __name__ == '__main__':
    distributor, port = sys.argv[1:]
    run(distributor, int(port))
