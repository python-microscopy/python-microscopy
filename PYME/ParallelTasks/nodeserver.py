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

        cherrypy.engine.subscribe('stop', self.stop)

        self._do_poll = True
        self.pollThread = threading.Thread(target=self._poll)
        self.pollThread.start()

    def _announce(self):
        t = time.time()
        if (t - self._lastAnnounceTime) > 1:
            self._lastAnnounceTime = t

            logger.debug('Announcing to %s' % self.distributor_url)
            url = self.distributor_url + 'distributor/announce?nodeID=%s&ip=%s&port=%d' % (self.nodeID, self.ip_address, self.port)

            try:
                requests.post(url, timeout=10)
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
            self._announce()
            self._do_handins()
            self._update_tasks()
            time.sleep(5)

    def stop(self):
        self._do_poll = False


    @cherrypy.expose
    @cherrypy.tools.json_out()
    def tasks(self, workerID):
        self.workerIDs.append(workerID)
        if self._tasks.qsize() < 10:
            self._update_tasks()

        task = self._tasks.get()
        return {'ok' : True, 'result' :task}

    @cherrypy.expose
    def handin(self, taskID, status):
        self._handins.put({'taskID': taskID, 'status':status})


    def _rateTask(self, task):
        cost = 1.0
        if task['type'] == 'localization':
            filename, serverfilter = clusterIO.parseURL(task['input']['frames'])
            filename = '/'.join([filename.lstrip('/'), 'frame%05d.pzf' % task['taskdef']['frameIndex']])

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

    externalAddr = socket.gethostbyname(socket.gethostname())

    nodeserver = NodeServer('http://' + distributor + '/', port = port, ip_address=externalAddr)

    try:

        cherrypy.quickstart(nodeserver, '/node/')
    finally:
        nodeserver._do_poll = False


if __name__ == '__main__':
    distributor, port = sys.argv[1:]
    run(distributor, int(port))
