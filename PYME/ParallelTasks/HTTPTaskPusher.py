# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 23:59:45 2015

@author: david
"""
import time
import numpy as np
import threading
import requests
from PYME.IO import DataSources
from PYME.IO import clusterResults
import json
import socket
import random

from PYME.misc import pyme_zeroconf as pzc
from PYME.misc.computerName import GetComputerName
compName = GetComputerName()

def _getTaskQueueURI():
    ns = pzc.getNS('_pyme-taskdist')

    queueURLs = {}
    for name, info in ns.advertised_services.items():
        queueURLs[name] = 'http://%s:%d' % (socket.inet_ntoa(info.address), info.port)

    try:
        #try to grab the distributor on the local computer
        return queueURLs[compName]
    except KeyError:
        #if there is no local distributor, choose one at random
        return queueURLs.items()[random.randint(0, len(queueURLs)-1)]

import logging
logging.basicConfig()

class HTTPTaskPusher(object):
    def __init__(self, dataSourceID, metadata, resultsFilename, queueName = None, startAt = 10, dataSourceModule=None, serverfilter=''):
        if queueName is None:
            queueName = resultsFilename

        self.queueID = queueName
        self.dataSourceID = dataSourceID
        self.resultsURI = 'PYME-CLUSTER://%s/__aggregate_h5r/%s' % (serverfilter, resultsFilename)

        self.taskQueueURI = _getTaskQueueURI()

        self.mdh = metadata

        #load data source
        if dataSourceModule is None:
            DataSource = DataSources.getDataSourceForFilename(dataSourceID)
        else:
            DataSource = __import__('PYME.IO.DataSources.' + dataSourceModule, fromlist=['PYME', 'io', 'DataSources']).DataSource #import our data source
        self.ds = DataSource(self.dataSourceID)
        
        #set up results file:
        clusterResults.fileResults(self.resultsURI + '/MetaData', metadata)
        clusterResults.fileResults(self.resultsURI + '/Events', self.ds.getEvents())

        self.currentFrameNum = startAt
        
        self.doPoll = True
        
        self.pollT = threading.Thread(target=self._updatePoll)
        self.pollT.start()

    def fileTasksForFrames(self):
        numTotalFrames = self.ds.getNumSlices()
        if  numTotalFrames > (self.currentFrameNum + 1):
            mdstring = self.mdh.to_JSON() #TODO - use a URI instead

            tasks = [{'id':self.queueID,
                      'type':'localization',
                      'taskdef': {'frameIndex': frameNum, 'metadata':mdstring},
                      'inputs' : {'frames': self.dataSourceID},
                      'outputs' : {'results': self.resultsURI}
                      } for frameNum in range(self.currentFrameNum, numTotalFrames)]

            task_list = json.dumps(tasks)

            r = requests.post('%s/distributor/tasks?queue=%s' % (self.taskQueueURI, self.queueID), data=task_list)
            if r.status_code == 200 and r.json()['Ok']:
                logging.debug('Successfully posted tasks')
                self.currentFrameNum = numTotalFrames - 1
            else:
                logging.error('Failed on posting tasks with status code: %d' % r.status_code)


    
    def _updatePoll(self):
        while (self.doPoll == True):
            self.fileTasksForFrames()
            if self.ds.isComplete():
                self.doPoll = False
            else:
                time.sleep(1)
        
    def cleanup(self):
        self.doPoll = False
