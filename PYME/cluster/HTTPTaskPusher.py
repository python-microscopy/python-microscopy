# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 23:59:45 2015

@author: david
"""
import hashlib
import json
import random
import socket
import threading
import time

from PYME.IO import DataSources
from PYME.IO import clusterIO
from PYME.IO import clusterResults
from PYME.misc import pyme_zeroconf as pzc
from PYME.misc import hybrid_ns
from PYME.misc.computerName import GetComputerName
from six import string_types

compName = GetComputerName()

import logging
logger = logging.getLogger(__name__)

def _getTaskQueueURI(n_retries=2):
    """Discover the distributors using zeroconf and choose one"""
    ns = hybrid_ns.getNS('_pyme-taskdist')

    queueURLs = {}
    
    def _search():
        for name, info in ns.get_advertised_services():
            if name.startswith('PYMEDistributor'):
                queueURLs[name] = 'http://%s:%d' % (socket.inet_ntoa(info.address), info.port)
                
    _search()
    while not queueURLs and (n_retries > 0):
        logging.info('could not find a distributor, waiting 5s and trying again')
        time.sleep(5)
        n_retries -= 1
        _search()

    try:
        #try to grab the distributor on the local computer
        local_queues = [q for q in queueURLs if compName in q]
        logger.debug('local_queues: %s' % local_queues)
        return queueURLs[local_queues[0]]
    except (KeyError, IndexError):
        #if there is no local distributor, choose one at random
        logger.info('no local distributor, choosing one at random')
        return random.choice(queueURLs.values())

def verify_cluster_results_filename(resultsFilename):
    """
    Checks whether a results file already exists on the cluster, and returns an available version of the results
    filename. Should be called before writing a new results file.

    Parameters
    ----------
    resultsFilename : str
        cluster path, e.g. pyme-cluster:///example_folder/name.h5r
    Returns
    -------
    resultsFilename : str
        cluster path which may have _# appended to it if the input resultsFileName is already in use, e.g.
        pyme-cluster:///example_folder/name_1.h5r

    """
    from PYME.IO import clusterIO
    import os
    if clusterIO.exists(resultsFilename):
        di, fn = os.path.split(resultsFilename)
        i = 1
        stub = os.path.splitext(fn)[0]
        while clusterIO.exists(os.path.join(di, stub + '_%d.h5r' % i)):
            i += 1

        resultsFilename = os.path.join(di, stub + '_%d.h5r' % i)

    return resultsFilename


def launch_localize(analysisMDH, seriesName):
    """
    Pushes an analysis task for a given series to the distributor

    Parameters
    ----------
    analysisMDH : dictionary-like
        MetaDataHandler describing the analysis tasks to launch
    seriesName : str
        cluster path, e.g. pyme-cluster:///example_folder/series
    Returns
    -------

    """
    import logging
    import json
    from PYME.IO import MetaDataHandler
    from PYME.Analysis import MetaData
    from PYME.IO.FileUtils.nameUtils import genClusterResultFileName
    from PYME.IO import unifiedIO

    resultsFilename = verify_cluster_results_filename(genClusterResultFileName(seriesName))
    logging.debug('Results file: ' + resultsFilename)

    resultsMdh = MetaDataHandler.NestedClassMDHandler()
    # NB - anything passed in analysis MDH will wipe out corresponding entries in the series metadata
    resultsMdh.update(json.loads(unifiedIO.read(seriesName + '/metadata.json')))
    resultsMdh.update(analysisMDH)

    resultsMdh['EstimatedLaserOnFrameNo'] = resultsMdh.getOrDefault('EstimatedLaserOnFrameNo',
                                                                    resultsMdh.getOrDefault('Analysis.StartAt', 0))
    MetaData.fixEMGain(resultsMdh)
    # resultsMdh['DataFileID'] = fileID.genDataSourceID(image.dataSource)

    # TODO - do we need to keep track of the pushers in some way (we currently rely on the fact that the pushing thread
    # will hold a reference
    pusher = HTTPTaskPusher.HTTPTaskPusher(dataSourceID=seriesName,
                                           metadata=resultsMdh, resultsFilename=resultsFilename)

    logging.debug('Queue created')


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
        if '~' in self.dataSourceID or '~' in self.queueID or '~' in resultsFilename:
            raise RuntimeError('File, queue or results name must NOT contain dashes')

        self.resultsURI = 'PYME-CLUSTER://%s/__aggregate_h5r/%s' % (serverfilter, resultsFilename)

        resultsMDFilename = resultsFilename + '.json'
        self.results_md_uri = 'PYME-CLUSTER://%s/%s' % (serverfilter, resultsMDFilename)

        self.taskQueueURI = _getTaskQueueURI()

        self.mdh = metadata
        self.mdh['Analysis.DataFileURI'] = self.dataSourceID  # for convenient linking after results files get downloaded/moved

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

        # set up metadata file which is used for deciding how to launch the analysis
        clusterIO.put_file(resultsMDFilename, self.mdh.to_JSON(), serverfilter=serverfilter)
        
        #wait until clusterIO caches clear to avoid replicating the results file.
        #time.sleep(1.5) #moved inside polling thread so launches will run quicker

        self.currentFrameNum = startAt

        self._task_template = None
        
        self.doPoll = True
        
        self.pollT = threading.Thread(target=self._updatePoll)
        self.pollT.start()

    def _postTasks(self, task_list):
        if isinstance(task_list[0], string_types):
            task_list = '[' + ',\n'.join(task_list) + ']'
        else:
            task_list = json.dumps(task_list)

        s = clusterIO._getSession(self.taskQueueURI)
        r = s.post('%s/distributor/tasks?queue=%s' % (self.taskQueueURI, self.queueID), data=task_list,
                          headers={'Content-Type': 'application/json'})

        if r.status_code == 200 and r.json()['ok']:
            logging.debug('Successfully posted tasks')
        else:
            logging.error('Failed on posting tasks with status code: %d' % r.status_code)

    @property
    def _taskTemplate(self):
        if self._task_template is None:
            tt = {'id': '{frameNum:04d}',
                      'type':'localization',
                      'taskdef': {'frameIndex': '{frameNum:d}', 'metadata':self.results_md_uri},
                      'inputs' : {'frames': self.dataSourceID},
                      'outputs' : {'fitResults': self.resultsURI+'/FitResults',
                                   'driftResults':self.resultsURI+'/DriftResults'}
                      }
            self._task_template = json.dumps(tt)

        return self._task_template


    def fileTasksForFrames(self):
        numTotalFrames = self.ds.getNumSlices()
        logging.debug('numTotalFrames: %s, currentFrameNum: %d' % (numTotalFrames, self.currentFrameNum))
        numFramesOutstanding = 0
        while  numTotalFrames > (self.currentFrameNum + 1):
            logging.debug('we have unpublished frames - push them')

            #turn our metadata to a string once (outside the loop)
            #mdstring = self.mdh.to_JSON() #TODO - use a URI instead
            
            newFrameNum = min(self.currentFrameNum + 1000, numTotalFrames)

            #create task definitions for each frame
            tasks = [{'id': '%04d' % frameNum,
                      'type':'localization',
                      'taskdef': {'frameIndex': str(frameNum), 'metadata':self.results_md_uri},
                      'inputs' : {'frames': self.dataSourceID},
                      'outputs' : {'fitResults': self.resultsURI+'/FitResults',
                                   'driftResults':self.resultsURI+'/DriftResults'}
                      } for frameNum in range(self.currentFrameNum, newFrameNum)]

            task_list = tasks #json.dumps(tasks)

            #task_list = [self._taskTemplate.format(frameNum=frameNum) for frameNum in range(self.currentFrameNum, newFrameNum)]

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
        #wait until clusterIO caches clear to avoid replicating the results file.
        time.sleep(1.5)
        
        while (self.doPoll == True):
            framesOutstanding = self.fileTasksForFrames()
            if self.ds.is_complete and not (framesOutstanding > 0):
                logging.debug('all tasks pushed, ending loop.')
                self.doPoll = False
            else:
                time.sleep(1)
        
    def cleanup(self):
        self.doPoll = False


class RecipePusher(object):
    def __init__(self, recipe=None, recipeURI=None):
        from PYME.recipes import Recipe
        if recipe:
            if isinstance(recipe, string_types):
                self.recipe_text = recipe
                self.recipe = Recipe.fromYAML(recipe)
            else:
                self.recipe_text = recipe.toYAML()
                self.recipe = recipe

            self.recipeURI = None
        else:
            self.recipe = None
            if recipeURI is None:
                raise ValueError('recipeURI must be defined if no recipe given')
            else:
                from PYME.IO import unifiedIO
                self.recipeURI = recipeURI
                self.recipe = Recipe.fromYAML(unifiedIO.read(recipeURI))

        self.taskQueueURI = _getTaskQueueURI()

        #generate a queue ID as a hash of the recipe and the current time
        h = hashlib.md5(self.recipeURI if self.recipeURI else self.recipe_text)
        h.update('%s' % time.time())
        self.queueID = h.hexdigest()

    def _postTasks(self, task_list):
        if isinstance(task_list[0], string_types):
            task_list = '[' + ',\n'.join(task_list) + ']'
        else:
            task_list = json.dumps(task_list)

        s = clusterIO._getSession(self.taskQueueURI)
        r = s.post('%s/distributor/tasks?queue=%s' % (self.taskQueueURI, self.queueID), data=task_list,
                   headers={'Content-Type': 'application/json'})

        if r.status_code == 200 and r.json()['ok']:
            logging.debug('Successfully posted tasks')
        else:
            logging.error('Failed on posting tasks with status code: %d' % r.status_code)


    def _generate_task(self, **kwargs):
        input_names = kwargs.keys()

        #our id will be a hash of our recipe text (or name), the time, and the input names
        h = hashlib.md5(self.recipeURI if self.recipeURI else self.recipe_text)
        h.update('%s' % time.time())
        h.update(''.join([kwargs[input_name] for input_name in input_names]))

        task = {'id': h.hexdigest(),
              'type': 'recipe',
              'inputs': {input_name: kwargs[input_name] for input_name in input_names},
              #'outputs': {output_name: kwargs[output_name] for output_name in self.recipe.outputs}
              }

        if self.recipeURI:
            task['taskdefRef'] = self.recipeURI
        else:
            task['taskdef'] = self.recipe_text

        return task

    def fileTasksForInputs(self, **kwargs):
        from PYME.IO import clusterIO
        input_names = kwargs.keys()
        inputs = {k : kwargs[k] if isinstance(kwargs[k], list) else clusterIO.cglob(kwargs[k], include_scheme=True) for k in input_names}

        numTotalFrames = len(list(inputs.values())[0])
        self.currentFrameNum = 0

        logger.debug('numTotalFrames = %d' % numTotalFrames)
        logger.debug('inputs = %s' % inputs)

        while numTotalFrames > (self.currentFrameNum + 1):
            logging.debug('we have unpublished frames - push them')

            newFrameNum = min(self.currentFrameNum + 1000, numTotalFrames)

            #create task definitions for each frame
            tasks = [self._generate_task(**{k : inputs[k][frameNum] for k in inputs.keys()}) for frameNum in range(self.currentFrameNum, newFrameNum)]

            task_list = tasks #json.dumps(tasks)

            #print tasks


            threading.Thread(target=self._postTasks, args=(task_list,)).start()

            self.currentFrameNum = newFrameNum



