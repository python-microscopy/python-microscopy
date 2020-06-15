
import time
import threading
from PYME.IO import DataSources, clusterIO, unifiedIO, clusterResults
import json
import socket
import random
from PYME.misc import hybrid_ns
from PYME.misc.computerName import GetComputerName
import logging

logger = logging.getLogger(__name__)


def _get_ruleserver_uri(n_retries=2):
    """Discover the distributors using zeroconf and choose one"""
    ns = hybrid_ns.getNS('_pyme-taskdist')

    servers = {}

    def _search():
        for name, info in ns.get_advertised_services():
            if name.startswith('PYMERuleServer'):
                print(info, info.address)
                servers[name] = 'http://%s:%d' % (socket.inet_ntoa(info.address), info.port)

    _search()
    while not servers and (n_retries > 0):
        logging.info('could not find a rule server, waiting 5s and trying again')
        time.sleep(5)
        n_retries -= 1
        _search()

    # if we only have one option, go with it
    server_uris = list(servers.values())
    if len(server_uris) == 1:
        return server_uris[0]
    # otherwise, prefer a local server if available
    try:
        return servers[GetComputerName()]
    except KeyError:
        logging.info('no local rule server; found %d, choosing one at random' % len(server_uris))
        return random.choice(server_uris)

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
        Link two rules, hardcoding the output URIs from the first as inputs to the second

        Parameters
        ----------
        chained_rule: Rule

        """
        chained_rule.chain_inputs(self.outputs)
        self._template['on_completion'] = {
            'template': chained_rule.template
        }

    @property
    def outputs(self):
        """
        Return task outputs in list with a dictionary mapping output names to URIs.

        Returns
        -------
        outputs: list
            returns list of dictionaries mapping output names to URIs. One dict per task specified by this rule.

        """
        raise NotImplementedError

    def chain_inputs(self, inputs):
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
    def __init__(self, analysis_metadata, series_uri=None, server_filter=''):
        """

        Parameters
        ----------
        analysis_metadata: PYME.IO.MetaDataHandler.MDHandlerBase
             describes the analysis rule to create
        series_uri: str
            cluster path, e.g. pyme-cluster:///example_folder/series
        server_filter: str
            [optional] cluster name used when finding servers for analysis - used to facilitate the operation for
            multiple clusters on one network segment. NB - server filter for data series location should be included in
            its uri, this parameter only affects which cluster performs the localization
        """
        Rule.__init__(self)
        self._template['type'] = 'localization'
        self.analysis_metadata = analysis_metadata
        self.server_filter = server_filter

        self._results_prepared = False
        if series_uri:
            self.prepare_results_files(series_uri)
            self._results_prepared = True

    def prepare_results_files(self, series_uri):
        import json
        from PYME.IO import MetaDataHandler
        from PYME.Analysis import MetaData
        from PYME.IO.FileUtils.nameUtils import genClusterResultFileName
        unifiedIO.assert_uri_ok(series_uri)
        self.results_filename = clusterResults.verify_cluster_results_filename(genClusterResultFileName(series_uri))
        logger.info('Results file: ' + self.results_filename)
        logger.info('Results file: ' + self.results_filename)
        results_mdh = MetaDataHandler.NestedClassMDHandler()
        # NB - anything passed in analysis MDH will wipe out corresponding entries in the series metadata
        results_mdh.update(json.loads(unifiedIO.read(series_uri + '/metadata.json')))
        results_mdh.update(self.analysis_metadata)

        results_mdh['EstimatedLaserOnFrameNo'] = results_mdh.getOrDefault('EstimatedLaserOnFrameNo',
                                                                          results_mdh.getOrDefault('Analysis.StartAt',
                                                                                                   0))
        MetaData.fixEMGain(results_mdh)

        if '~' in series_uri or '~' in self.results_filename:
            raise RuntimeError('filenames on the cluster must NOT contain ~')

        # create results file, and metadata table
        results_uri = clusterResults.pickResultsServer('__aggregate_h5r/%s' % self.results_filename, self.server_filter)
        logging.debug('results URI: ' + results_uri)
        clusterResults.fileResults(results_uri + '/MetaData', self.analysis_metadata)

        # create metadata file which will be used in the rule/task definition
        results_md_filename = self.results_filename + '.json'
        clusterIO.put_file(results_md_filename, results_mdh.to_JSON().encode(), serverfilter=self.server_filter)

        self._template.update({
            'inputs': {
                'frames': series_uri
            },
            'taskdef': {
                'frameIndex': '{{taskID}}',
                'metadata': 'PYME-CLUSTER://%s/%s' % (self.server_filter, results_md_filename),
            },
            'outputs': {
                'fitResults': results_uri + '/FitResults',
                'driftResults': results_uri + '/DriftResults'
            }
        })

    def __del__(self):
        self._posting_poll = False

    def post(self):
        if not self._results_prepared:
            raise RuntimeError('results files not initiated, call prepare_results_files first')
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

    @property
    def outputs(self):
        return [{
            'fitResults': self.results_filename + '/FitResults',
            'driftResults': self.results_filename + '/DriftResults'
        }]

    def chain_inputs(self, inputs):
        # currently only support single input/output per LocalizationRule
        if len(inputs) > 1 or ('frames' not in inputs[0].keys() and 'input' not in inputs[0].keys()):
            raise RuntimeError('Malformed input; LocalizationRule does not yet support multiple/fancy input chaining')

        inp = inputs[0]
        try:
            inp['frames']
        except KeyError:
            inp['frames'] = inp['input']

        self._template['inputs'] = inp
        self.prepare_results_files(inp['frames'])



class RecipeRule(Rule):
    def __init__(self, recipe, inputs=None, output_dir=None):
        """

        Parameters
        ----------
        recipe: str
            recipe URI or recipe text
        inputs: list
            list of dictionaries mapping recipe inputs to file URIs. Each input dictionary will create a separate task,
            however a chained rule/follow-on rule will only be called after all tasks for the preceding rule are
            finished.
        output_dir: str
            [optional] dataserver-relative path to recipe output directory
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

    def post(self):
        ruleserver_uri = _get_ruleserver_uri()
        rule, n_tasks = {'template': self.template}, 1
        if self._task_inputs:
            rule['inputsByTask'] = {t_ind: task for t_ind, task in enumerate(self._task_inputs)}
            n_tasks = len(self._task_inputs)
            logger.debug('inputs by task: %s' % rule['inputsByTask'])

        s = clusterIO._getSession(ruleserver_uri)
        r = s.post('%s/add_integer_id_rule?max_tasks=%d&release_start=%d&release_end=%d' % (ruleserver_uri, n_tasks, 0,
                                                                                            n_tasks),
                   data=json.dumps(rule), headers={'Content-Type': 'application/json'})

        if r.status_code == 200:
            resp = r.json()
            self._rule_id = resp['ruleID']
            logging.debug('Successfully created rule')
        else:
            logging.error('Failed creating rule with status code: %d' % r.status_code)

    @property
    def outputs(self):
        from PYME.recipes.modules import ModuleCollection
        from PYME.recipes.base import OutputModule

        if 'taskdef' in self.template.keys():
            recipe = ModuleCollection.fromYAML(self.template['taskdef']['recipe'])
        else:
            recipe = ModuleCollection.fromYAML(unifiedIO.read(self.template['taskdefRef']))

        # find outputs
        outputs = {}
        for mod in recipe.modules:
            if isinstance(mod, OutputModule) and mod.scheme.upper() == 'PYME-CLUSTER://':
                for _in in mod.inputs:
                    # we might override location for things being saved multiple times, but we have no way of
                    # prioritizing locations and shouldn't really care anyway.
                    outputs[_in] = mod.filePattern

        # get input file stub(s) and convert output patterns to URIs
        if self._task_inputs:
            file_stub_key = 'input' if 'input' in self._task_inputs[0].keys() else self._task_inputs[0].keys()[0]
            file_stubs = [task_input[file_stub_key] for task_input in self._task_inputs]
        else:
            file_stubs = [self.template['inputs']]
        output_list = []
        for ind in range(len(file_stubs)):
            file_stub = file_stubs[ind]
            out = {}
            for k in outputs.keys():
                out[k] = outputs[k].format(output_dir=self.template['output_dir'], file_stub=file_stub)
                output_list.append(out)
        return output_list

    def chain_inputs(self, inputs):
        """
        Currently, there is no support for taskInputs in chained recipes, meaning you can only chain a recipe for a
        single task. See PYME.cluster.ruleserver.RuleServer._poll_rules

        Parameters
        ----------
        inputs: dict, list
            dictionary mapping recipe input names to URIs, or length-1 list of a single dict doing the same.

        """
        if isinstance(inputs, list):
            # at the moment we don't support taskInputs, meaning we can only handle a single inputs dictionary
            if not isinstance(inputs[0], dict) or len(inputs > 1):
                raise TypeError('rule chaining recipes do not yet support taskInput style task generation')
            inputs = inputs[0]

        self._template['inputs'] = inputs
        self._task_inputs = None



class RuleChain(list):
    def post_all(self):
        for ri in range(len(self)):
            self[ri].post_rule()
            if ri != len(self):
                self[ri].chain_rule(self[ri + 1])

    def set_chain_input(self, inputs):
        if isinstance(inputs, dict):
            inputs = [inputs]
        self[0].chain_inputs(inputs)
