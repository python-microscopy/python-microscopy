import cherrypy
import threading
import requests
import Queue
import logging
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('nodeserver')
logger.setLevel(logging.DEBUG)

import time
import sys

from PYME.misc import computerName
from PYME import config
from PYME.IO import clusterIO

class NodeServer(object):
    def __init__(self, distributor, ip_address, port, nodeID=computerName.GetComputerName()):
        self._tasks = Queue.Queue()
        self._handins = Queue.Queue()

        self.nodeID = nodeID
        self.distributor_url = distributor
        self.ip_address = ip_address
        self.port = port

        self.workerIDs = []

        self._lastUpdateTime = 0
        self._lastAnnounceTime = 0
        self._anounce_url = self.distributor_url + 'distributor/announce?nodeID=%s&ip=%s&port=%d' % (self.nodeID, self.ip_address, self.port)

        cherrypy.engine.subscribe('stop', self.stop)

        self._do_poll = True

        #set up threads to poll the distributor and announce ourselves and get and return tasks
        self.pollThread = threading.Thread(target=self._poll)
        self.pollThread.start()

        self.announceThread = threading.Thread(target=self._announce_loop)
        self.announceThread.start()

        self.taskThread = threading.Thread(target=self._poll_tasks)
        self.taskThread.start()


    def _announce(self):
        t = time.time()
        if True:#(t - self._lastAnnounceTime) > .5:
            self._lastAnnounceTime = t

            logger.debug('Announcing to %s' % self.distributor_url)

            try:
                requests.post(self._anounce_url, timeout=1)
            except (requests.Timeout, requests.ConnectionError):
                logger.error('Could not connect to distributor %s' % self.distributor_url)


    @property
    def num_tasks_to_request(self):
        return config.get('nodeserver-chunksize', 50)*len(self.workerIDs)

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

    def _do_handins(self):
        handins = []

        try:
            while True:
                handins.append(self._handins.get_nowait())
        except Queue.Empty:
            pass

        if len(handins) > 0:
            requests.post(self.distributor_url + 'distributor/handin?nodeID=%s' % self.nodeID, json=handins)

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
    def tasks(self, workerID):
        self.workerIDs.append(workerID)
        if self._tasks.qsize() < 10:
            self._update_tasks()

        tasks = [self._tasks.get()] #wait for at leas 1 task
        nTasks = 1

        try:
            while nTasks < 50:
                tasks.append(self._tasks.get_nowait())
                nTasks += 1
        except Queue.Empty:
            pass

        return {'ok': True, 'result': tasks}



    @cherrypy.expose
    def handin(self, taskID, status):
        self._handins.put({'taskID': taskID, 'status':status})


    def _rateTask(self, task):
        cost = 1.0
        if task['type'] == 'localization':
            filename, serverfilter = clusterIO.parseURL(task['inputs']['frames'])
            filename = '/'.join([filename.lstrip('/'), 'frame%05d.pzf' % int(task['taskdef']['frameIndex'])])

            if clusterIO.isLocal(filename, serverfilter):
                cost = .01

        return {'id' : task['id'], 'cost': cost}

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def rate(self):
        logger.debug('Rating tasks')
        tasks = cherrypy.request.json

        logging.debug(tasks)

        ratings = [self._rateTask(task) for task in tasks]
        logger.debug('Returning %d ratings ... ' % len(ratings))

        return {'ok': True, 'result': ratings}


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
