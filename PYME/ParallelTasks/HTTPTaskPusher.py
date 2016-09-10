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
import logging

from PYME.misc import pyme_zeroconf as pzc
from PYME.misc.computerName import GetComputerName
compName = GetComputerName()

def _getTaskQueueURI():
    """Discover the distributors using zeroconf and choose one"""
    ns = pzc.getNS('_pyme-taskdist')

    queueURLs = {}
    for name, info in ns.advertised_services.items():
        if name.startswith('PYMEDistributor'):
            queueURLs[name] = 'http://%s:%d' % (socket.inet_ntoa(info.address), info.port)

    try:
        #try to grab the distributor on the local computer
        return queueURLs[compName]
    except KeyError:
        #if there is no local distributor, choose one at random
        logging.info('no local distributor, choosing one at random')
        return random.choice(queueURLs.values())

import logging
logging.basicConfig()

class HTTPTaskPusher(object):
    def __init__(self, dataSourceID, metadata, resultsFilename, queueName = None, startAt = 10, dataSourceModule=None, serverfilter=''):
        """
        Create a pusher and push tasks for each frame in a series. For use with the new cluster distribution architecture

        Parameters
        ----------
        dataSourceID : str
            The URI of the data source - e.g. PYME-CLUSTER://serverfilter/path/to/data
        metadata : PYME.IO.MetaDataHandler object
            The acquisition and analysis metadata
        resultsFilename : str
            The cluster relative path to the results file. e.g. "<username>/analysis/<date>/seriesname.h5r"
        queueName : str
            a name to give the queue. The results filename is used if no name is given.
        startAt : int
            which frame to start at. TODO - read from metadata instead of taking as a parameter.
        dataSourceModule : str [optional]
            The name of the module to use for reading the raw data. If not given, it will be inferred from the dataSourceID
        serverfilter : str
            A cluster filter, for use when multiple PYME clusters are visible on the same network segment.
        """
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
        logging.debug('resultsURI: ' + self.resultsURI)
        clusterResults.fileResults(self.resultsURI + '/MetaData', metadata)
        clusterResults.fileResults(self.resultsURI + '/Events', self.ds.getEvents())
        
        #wait until clusterIO caches clear to avoid replicating the results file.
        time.sleep(1)

        self.currentFrameNum = startAt
        
        self.doPoll = True
        
        self.pollT = threading.Thread(target=self._updatePoll)
        self.pollT.start()

    def _postTasks(self, task_list):
        r = requests.post('%s/distributor/tasks?queue=%s' % (self.taskQueueURI, self.queueID), data=task_list)
        if r.status_code == 200 and r.json()['ok']:
            logging.debug('Successfully posted tasks')
        else:
            logging.error('Failed on posting tasks with status code: %d' % r.status_code)

    def fileTasksForFrames(self):
        numTotalFrames = self.ds.getNumSlices()
        logging.debug('numTotalFrames: %s, currentFrameNum: %d' % (numTotalFrames, self.currentFrameNum))
        numFramesOutstanding = 0
        if  numTotalFrames > (self.currentFrameNum + 1):
            logging.debug('we have unpublished frames - push them')

            #turn our metadata to a string once (outside the loop)
            mdstring = self.mdh.to_JSON() #TODO - use a URI instead
            
            newFrameNum = min(self.currentFrameNum + 1000, numTotalFrames)

            #create task definitions for each frame
            tasks = [{'id':str(frameNum),
                      'type':'localization',
                      'taskdef': {'frameIndex': str(frameNum), 'metadata':mdstring},
                      'inputs' : {'frames': self.dataSourceID},
                      'outputs' : {'fitResults': self.resultsURI+'/FitResults',
                                   'driftResults':self.resultsURI+'/DriftResults'}
                      } for frameNum in range(self.currentFrameNum, newFrameNum)]

            task_list = json.dumps(tasks)

            # r = requests.post('%s/distributor/tasks?queue=%s' % (self.taskQueueURI, self.queueID), data=task_list)
            # if r.status_code == 200 and r.json()['ok']:
            #     logging.debug('Successfully posted tasks')
            #     #self.currentFrameNum = newFrameNum
            # else:
            #     logging.error('Failed on posting tasks with status code: %d' % r.status_code)

            threading.Thread(target=self._postTasks, args=(task_list,)).start()

            self.currentFrameNum = newFrameNum

            numFramesOutstanding = numTotalFrames - self.currentFrameNum

        return  numFramesOutstanding


    
    def _updatePoll(self):
        logging.debug('task pusher poll loop started')
        while (self.doPoll == True):
            framesOutstanding = self.fileTasksForFrames()
            if self.ds.isComplete() and not (framesOutstanding > 0):
                logging.debug('all tasks pushed, ending loop.')
                self.doPoll = False
            else:
                time.sleep(1)
        
    def cleanup(self):
        self.doPoll = False
