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
from PYME.IO import clusterIO
import json
import socket
import random

import hashlib

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
            if name.startswith('PYMERuleServer'):
                print(info, info.address)
                queueURLs[name] = 'http://%s:%d' % (socket.inet_ntoa(info.address), info.port)
                
    _search()
    while not queueURLs and (n_retries > 0):
        logging.info('could not find a rule server, waiting 5s and trying again')
        time.sleep(5)
        n_retries -= 1
        _search()

    try:
        #try to grab the distributor on the local computer
        return queueURLs[compName]
    except KeyError:
        #if there is no local distributor, choose one at random
        logging.info('no local rule server, choosing one at random')
        return random.choice(list(queueURLs.values()))

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

def setup_localization_result_files(series_uri, analysis_metadata, server_filter=''):
    """

    Parameters
    ----------
    series_uri: str
        cluster path, e.g. pyme-cluster:///example_folder/series
    analysis_metadata: PYME.IO.MetaDataHandler.MDHandlerBase
         describes the analysis rule to create
    server_filter: str
        cluster name used when finding servers - used to facilitate the operation for multiple clusters on the one
        network segment.

    Returns
    -------
    results_uri: str
        cluster path of the localization results file
    results_md_uri: str
        cluster path of the localization results metadata, which will be used to define the localization rule/tasks.

    """
    import json
    from PYME.IO import MetaDataHandler
    from PYME.Analysis import MetaData
    from PYME.IO.FileUtils.nameUtils import genClusterResultFileName
    from PYME.IO import unifiedIO

    unifiedIO.assert_uri_ok(series_uri)

    results_filename = verify_cluster_results_filename(genClusterResultFileName(series_uri))
    logger.info('Results file: ' + results_filename)

    results_mdh = MetaDataHandler.NestedClassMDHandler()
    # NB - anything passed in analysis MDH will wipe out corresponding entries in the series metadata
    results_mdh.update(json.loads(unifiedIO.read(series_uri + '/metadata.json')))
    results_mdh.update(analysis_metadata)

    results_mdh['EstimatedLaserOnFrameNo'] = results_mdh.getOrDefault('EstimatedLaserOnFrameNo',
                                                                    results_mdh.getOrDefault('Analysis.StartAt', 0))
    MetaData.fixEMGain(results_mdh)

    if '~' in series_uri or '~' in results_filename:
        raise RuntimeError('filenames on the cluster must NOT contain ~')

    # create results file, and metadata table
    results_uri = clusterResults.pickResultsServer('__aggregate_h5r/%s' % results_filename, server_filter)
    logging.debug('results URI: ' + results_uri)
    clusterResults.fileResults(results_uri + '/MetaData', analysis_metadata)

    # create metadata file which will be used in the rule/task definition
    results_md_filename = results_filename + '.json'
    results_md_uri = 'PYME-CLUSTER://%s/%s' % (server_filter, results_md_filename)
    clusterIO.put_file(results_md_filename, results_mdh.to_JSON().encode(), serverfilter=server_filter)

    return results_filename, results_uri, results_md_uri


class HTTPRulePusher(object):
    def __init__(self, series_uri, metadata, queue_name=None, server_filter=clusterIO.local_serverfilter):
        """
        Create a pusher and push tasks for each frame in a series. For use with the new cluster distribution architecture

        Parameters
        ----------
        series_uri : str
            The URI of the data source - e.g. PYME-CLUSTER://serverfilter/path/to/data
        metadata : PYME.IO.MetaDataHandler.MDHandlerBase
            analysis metadata, note that for the localization reuslts, anything passed in analysis MDH will wipe out
            corresponding entries in the series metadata
        queue_name : str
            [optional] a name to give the queue. The results filename is used if no name is given.
        server_filter : str
            A cluster filter, for use when multiple PYME clusters are visible on the same network segment.
        """
        self.results_filename, self.results_uri, self.results_md_uri = setup_localization_result_files(series_uri,
                                                                                                       metadata,
                                                                                                       server_filter)
        if queue_name is None:
            queue_name = self.results_filename

        self.queueID = queue_name
        self.dataSourceID = series_uri
        self.taskQueueURI = _getTaskQueueURI()
        self.mdh = metadata
        self.current_frame_number = metadata.getOrDefault('Analysis.StartAt', 0)

        DataSource = DataSources.getDataSourceForFilename(series_uri)
        self.data_source = DataSource(series_uri)

        self._task_template = None
        self._ruleID = None
        self.doPoll = True
        
        #post our rule
        self.post_rule()
        self.pollT = threading.Thread(target=self._updatePoll)
        self.pollT.start()


    @property
    def _taskTemplate(self):
        if self._task_template is None:
            tt = {'id': '{{ruleID}}~{{taskID}}',
                      'type':'localization',
                      'taskdef': {'frameIndex': '{{taskID}}', 'metadata':self.results_md_uri},
                      'inputs' : {'frames': self.dataSourceID},
                      'outputs' : {'fitResults': self.results_uri+'/FitResults',
                                   'driftResults':self.results_uri+'/DriftResults'}
                      }
            self._task_template = json.dumps(tt)

        return self._task_template
    
    def post_rule(self):
        rule = {'template' : self._taskTemplate}

        if self.data_source.isComplete():
            queueSize = self.data_source.getNumSlices()
        else:
            queueSize = 1e6
        
        s = clusterIO._getSession(self.taskQueueURI)
        r = s.post('%s/add_integer_id_rule?timeout=300&max_tasks=%d' % (self.taskQueueURI,queueSize), data=json.dumps(rule),
                   headers={'Content-Type': 'application/json'})

        if r.status_code == 200:
            resp = r.json()
            self._ruleID = resp['ruleID']
            logging.debug('Successfully created rule')
        else:
            logging.error('Failed creating rule with status code: %d' % r.status_code)


    def fileTasksForFrames(self):
        numTotalFrames = self.data_source.getNumSlices()
        logging.debug('numTotalFrames: %s, currentFrameNum: %d' % (numTotalFrames, self.current_frame_number))
        numFramesOutstanding = 0
        while  numTotalFrames > (self.current_frame_number + 1):
            logging.debug('we have unpublished frames - push them')

            #turn our metadata to a string once (outside the loop)
            #mdstring = self.mdh.to_JSON() #TODO - use a URI instead
            
            newFrameNum = min(self.current_frame_number + 100000, numTotalFrames - 1)

            #create task definitions for each frame

            s = clusterIO._getSession(self.taskQueueURI)
            r = s.get('%s/release_rule_tasks?ruleID=%s&release_start=%d&release_end=%d' % (self.taskQueueURI, self._ruleID, self.current_frame_number, newFrameNum),
                      data='',
                      headers={'Content-Type': 'application/json'})

            if r.status_code == 200 and r.json()['ok']:
                logging.debug('Successfully posted tasks')
            else:
                logging.error('Failed on posting tasks with status code: %d' % r.status_code)

            self.current_frame_number = newFrameNum

            numFramesOutstanding = numTotalFrames  - 1 - self.current_frame_number

        return  numFramesOutstanding


    
    def _updatePoll(self):
        logging.debug('task pusher poll loop started')
        #wait until clusterIO caches clear to avoid replicating the results file.
        time.sleep(1.5)
        
        while (self.doPoll == True):
            framesOutstanding = self.fileTasksForFrames()
            if self.data_source.isComplete() and not (framesOutstanding > 0):
                logging.debug('all tasks pushed, ending loop.')
                self.doPoll = False
                # save events
                clusterResults.fileResults(self.results_uri + '/Events', self.data_source.getEvents())
            else:
                time.sleep(1)
        
    def cleanup(self):
        self.doPoll = False


class RecipePusher(object):
    def __init__(self, recipe=None, recipeURI=None, output_dir = None):
        from PYME.recipes.modules import ModuleCollection
        if recipe:
            if isinstance(recipe, string_types):
                self.recipe_text = recipe
                self.recipe = ModuleCollection.fromYAML(recipe)
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
                self.recipe = ModuleCollection.fromYAML(unifiedIO.read(recipeURI))
                
        self.output_dir = output_dir

        self.taskQueueURI = _getTaskQueueURI()

        #generate a queue ID as a hash of the recipe and the current time
        h = hashlib.md5(self.recipeURI if self.recipeURI else self.recipe_text)
        h.update('%s' % time.time())
        self.queueID = h.hexdigest()


    @property
    def _taskTemplate(self):
        task = '''{"id": "{{ruleID}}~{{taskID}}",
              "type": "recipe",
              "inputs" : {{taskInputs}},
              %s,
              %s
              }'''

        if self.output_dir is None:
            output_dir_n = ''
        else:
            output_dir_n = '"output_dir": "%s",' % self.output_dir
        
        if self.recipeURI:
            task = task % ('"taskdefRef" : "%s"' % self.recipeURI, output_dir_n)
        else:
            task = task % ('"taskdef" : {"recipe": "%s"}' % self.recipe_text, output_dir_n)
            

        return task

