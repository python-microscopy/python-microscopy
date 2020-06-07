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


def _get_ruleserver_uri(n_retries=2):
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
        # try to grab the distributor on the local computer
        return queueURLs[compName]
    except KeyError:
        # if there is no local distributor, choose one at random
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


class Rule(object):
    def __init__(self):
        self._template = {'id': '{{ruleID}}~{{taskID}}'}

    @property
    def template(self):
        return self._template

    def post(self):
        raise NotImplementedError

    def chain_rule(self, chained_rule):
        """
        replace uri's in chained rule inputs with output uri's from current rule

        Parameters
        ----------
        chained_rule: Rule

        """
        chained_rule.update_inputs(self.outputs)  # fixme - does outputs get us all outputs for e.g. recipe with many tasks?
        # fixme - need to make sure that update_inputs(outputs) works for localizations (tasks produce single output) and recipes (each task can produce set of outputs)
        self._template['on_completion'] = {
            'template': chained_rule.template
        }

    @property
    def outputs(self):
        try:
            return self.template['outputs']
        except KeyError:
            return {}

    @property
    def inputs(self):
        try:
            return self._template['inputs']
        except KeyError:
            return {}

    def update_inputs(self, inputs):
        """

        Parameters
        ----------
        inputs: dict, list of dict
            mapping from keys to input URIs. For localization rules this is often simply {'frames': datasource_uri}. For
            recipes this is more generally a list of dictionaries to apply the same recipe to more than one input
            dataset.

        """
        self._template['inputs'] = inputs

class LocalizationRule(Rule):
    def __init__(self, series_uri, analysis_metadata, server_filter=''):
        """

        Parameters
        ----------
        series_uri: str
            cluster path, e.g. pyme-cluster:///example_folder/series
        analysis_metadata: PYME.IO.MetaDataHandler.MDHandlerBase
             describes the analysis rule to create
        server_filter: str
            [optional] cluster name used when finding servers for analysis - used to facilitate the operation for
            multiple clusters on one network segment. NB - server filter for data series location should be included in
            its uri, this parameter only affects which cluster performs the localization
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
                                                                          results_mdh.getOrDefault('Analysis.StartAt',
                                                                                                   0))
        MetaData.fixEMGain(results_mdh)

        if '~' in series_uri or '~' in results_filename:
            raise RuntimeError('filenames on the cluster must NOT contain ~')

        # create results file, and metadata table
        results_uri = clusterResults.pickResultsServer('__aggregate_h5r/%s' % results_filename, server_filter)
        logging.debug('results URI: ' + results_uri)
        clusterResults.fileResults(results_uri + '/MetaData', analysis_metadata)

        # create metadata file which will be used in the rule/task definition
        results_md_filename = results_filename + '.json'
        clusterIO.put_file(results_md_filename, results_mdh.to_JSON().encode(), serverfilter=server_filter)

        Rule.__init__(self)
        self._template.update({
            'type': 'localization',
            'inputs': {
                'frames': series_uri
            },
            'taskdef': {
                'frameIndex': '{{taskID}}',
                'metadata': 'PYME-CLUSTER://%s/%s' % (server_filter, results_md_filename),
            },
            'outputs': {
                'fitResults': results_uri + '/FitResults',
                'driftResults': results_uri + '/DriftResults'
            }
        })

    def __del__(self):
        self._posting_poll = False

    def post(self):
        self.ruleserver_uri = _get_ruleserver_uri()
        self.datasource = DataSources.getDataSourceForFilename(self.template['inputs']['frames'])

        self._max_frames = max(self._max_frames, self.datasource.getNumSlices())

        s = clusterIO._getSession(self.ruleserver_uri)
        r = s.post('%s/add_integer_id_rule?timeout=300&max_tasks=%d' % (self.ruleserver_uri, self._max_frames),
                   data=json.dumps({'template': self.template}), headers={'Content-Type': 'application/json'})

        if r.status_code == 200:
            resp = r.json()
            self._rule_id = resp['ruleID']
            logging.debug('Successfully created rule')
        else:
            logging.error('Failed creating rule with status code: %d' % r.status_code)

        # set up some values for the per-frame task releasing thread
        self._posting_poll = True
        self._current_frame = 0

        # release task for each frame
        threading.Thread(target=self._poll)

    def _release_frame_tasks_for_bidding(self):
        n_frames = self.datasource.getNumSlices()
        logging.debug('datasource frames: %d, tasks filed: %d' % (n_frames, self._current_frame))
        n_outstanding = 0

        while n_frames > self._current_frame + 1:

            new_current_frame = min(self._current_frame + 100000, n_frames - 1)

            # release frames, creating tasks for each
            s = clusterIO._getSession(self.ruleserver_uri)
            r = s.get('%s/release_rule_tasks?ruleID=%s&release_start=%d&release_end=%d' % (self.ruleserver_uri,
                                                                                           self._rule_id,
                                                                                           self._current_frame,
                                                                                           new_current_frame),
                       data='', headers={'Content-Type': 'application/json'})

            if r.status_code == 200 and r.json()['ok']:
                logging.debug('Successfully created tasks from rule')
            else:
                logging.error('Failed on posting tasks with status code: %d' % r.status_code)

            self._current_frame = new_current_frame

            n_outstanding = n_frames - 1 - self._current_frame

        return n_outstanding


    def _poll(self):
        logger.debug('generating tasks for each frame')
        time.sleep(1.5)  # wait until clusterIO caches clear to avoid replicating the results file.

        while self._posting_poll:
            frames_outstanding = self._release_frame_tasks_for_bidding()
            if self.datasource.is_complete and frames_outstanding < 1:
                logging.debug('all tasks generated, ending loop')
                self._posting_poll = False
            else:
                time.sleep(1)


class RecipeRule(Rule):
    def __init__(self, recipe, inputs, output_dir=None):
        """

        Parameters
        ----------
        recipe: str
            recipe URI or recipe text
        inputs: list
            list of dictionaries mapping recipe inputs to file URIs. Each input dictionary will create a separate task,
            however a chained rule/follow-on rule will only be called after all tasks for the preceding rule are
            finished, making this a potential control-flow for running a recipe on a batch of intermediate recipe
            outputs.
        output_dir: str
        """
        Rule.__init__(self)
        self._template.update({
            'type': 'recipe',
            'inputs': '{{taskInputs}}'
        })

        if '\n' not in recipe:
            self._template['taskdefRef'] = recipe
        else:
            self._template['taskdef'] = {'recipe': recipe}

        if output_dir is not None:
            self._template['output_dir'] = output_dir

        self._task_inputs = inputs
        self._n_tasks = len(self._task_inputs)
        self._inputs_by_task = {t_ind: task for t_ind, task in enumerate(inputs)}



    def post(self):
        ruleserver_uri = _get_ruleserver_uri()

        logger.debug('inputs = %s' % self._inputs_by_task)

        rule = {'template': self.template, 'inputsByTask': self.inputs_by_task}

        s = clusterIO._getSession(ruleserver_uri)
        r = s.post('%s/add_integer_id_rule?max_tasks=%d&release_start=%d&release_end=%d' % (ruleserver_uri,
                                                                                            self._n_tasks, 0,
                                                                                            self._n_tasks),
                   data=json.dumps(rule), headers={'Content-Type': 'application/json'})

        if r.status_code == 200:
            resp = r.json()
            self._rule_id = resp['ruleID']
            logging.debug('Successfully created rule')
        else:
            logging.error('Failed creating rule with status code: %d' % r.status_code)

    @property
    def outputs(self):
        raise


class RuleChain(list):
    def post_all(self):
        for ri in range(len(self)):
            self[ri].post_rule()
            if ri != len(self):
                self[ri].chain_rule(self[ri + 1])

