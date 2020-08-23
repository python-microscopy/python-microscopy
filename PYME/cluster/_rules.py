"""
Initial attempt to refactor rule pushing to make it easier to push chained rules

WARNING: this will change significantly or potentially disappear - best to stick with HTTPRulePusher for most
use cases at present.

Major sticking point of this implementation is that the Rule class behaviour is not 
particularly intuitive (initialised with one series name, rule_id, etc ... , but modified and 
re-used for different series, and that the implementation of chaining does not map well to how 
chaining is implemented in the json rules it generates).

"""
import time
import threading
from PYME.IO import DataSources, clusterIO, unifiedIO, clusterResults
import json
import socket
import random
from PYME.misc import hybrid_ns
from PYME.misc.computerName import GetComputerName
from PYME.contrib import dispatch
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
        self._on_completion = None
        self.ruleserver_uri = None
        self._rule_id = None
        self._max_tasks = 1

    @property
    def template(self):
        return self._template

    @property
    def rule(self):
        rule = {'template': json.dumps(self.template)}  # todo - change rulenodeserver so we don't have to dumps template first
        if self._on_completion:
            rule['on_completion'] = self._on_completion
        return rule

    def post(self, thread_queue=None):
        """
        Post rule to ruleserver. 
        
        Should additionally update self.ruleserver_uri and self._rule_id in the process
        Parameters
        ----------
        thread_queue: queue.Queue
            queue to keep track of any threads used in task posting so they aren't prematurely garbage collected after
            the rule is posted and this method exits
        Returns
        -------
        """
        raise NotImplementedError

    def chain_rule(self, chained_rule):
        """
        Link two rules, note that this does not take care of input/output URI hardcoding
        Parameters
        ----------
        chained_rule: Rule
        """
        self._on_completion = chained_rule.rule

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

    @property
    def max_tasks(self):
        return self._max_tasks

    @max_tasks.setter
    def max_tasks(self, n_max):
        """
        Updates the number of tasks which can be created from this rule, updating the entry on the ruleserver
        as well if the rule has already been posted. Important to call for rules initialially created with 
        an arbitrary max-tasks defined (e.g. for a LocalizationRule created while the series is still spooling).
        Parameters
        ----------
        n_max : int
            Number of tasks which can be created from this rule.
        """
        self._max_tasks = n_max

        if self.ruleserver_uri and self._rule_id:  # rule has been posted, update on ruleserver
            s = clusterIO._getSession(self.ruleserver_uri)
            r = s.post('%s/update_max_tasks?rule_id=%s&n_max=%d' % (self.ruleserver_uri, self._rule_id, n_max),
                    data='', headers={'Content-Type': 'application/json'})

            if r.status_code == 200:
                logger.debug('Successfully updated rule max_tasks')
            else:
                logger.error('Failed to update rule max_tasks with status code: %d' % r.status_code)


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
        from PYME.cluster.HTTPRulePusher import verify_cluster_results_filename  # TODO - David if you like maybe we move this function to clusterResults?
        unifiedIO.assert_uri_ok(series_uri)
        self.results_filename = verify_cluster_results_filename(genClusterResultFileName(series_uri))
        if '~' in series_uri or '~' in self.results_filename:
            raise RuntimeError('filenames on the cluster must NOT contain ~')
        logger.info('Results file: ' + self.results_filename)

        results_mdh = MetaDataHandler.NestedClassMDHandler()
        # NB - anything passed in analysis MDH will wipe out corresponding entries in the series metadata
        results_mdh.update(json.loads(unifiedIO.read(series_uri + '/metadata.json')))
        results_mdh.update(self.analysis_metadata)
        results_mdh['EstimatedLaserOnFrameNo'] = results_mdh.getOrDefault('EstimatedLaserOnFrameNo',
                                                                          results_mdh.getOrDefault('Analysis.StartAt',
                                                                                                   0))
        MetaData.fixEMGain(results_mdh)

        # create results file, and metadata table
        self.results_uri = clusterResults.pickResultsServer('__aggregate_h5r/%s' % self.results_filename, self.server_filter)
        logging.debug('results URI: ' + self.results_uri)
        clusterResults.fileResults(self.results_uri + '/MetaData', results_mdh)

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
                'fitResults': self.results_uri + '/FitResults',
                'driftResults': self.results_uri + '/DriftResults'
            }
        })

        self._results_prepared = True

    def __del__(self):
        self._posting_poll = False

    def post(self, thread_queue=None):
        if not self._results_prepared:
            raise RuntimeError('results files not initiated, call prepare_results_files first')
        self.ruleserver_uri = _get_ruleserver_uri()
        self.datasource = DataSources.getDataSourceForFilename(self.template['inputs']['frames'])
        self.datasource = self.datasource(self.template['inputs']['frames'])

        self.max_tasks = self.datasource.getNumSlices() if self.datasource.is_complete else 1e6

        s = clusterIO._getSession(self.ruleserver_uri)
        r = s.post('%s/add_integer_id_rule?timeout=300&max_tasks=%d' % (self.ruleserver_uri, self.max_tasks),
                   data=json.dumps(self.rule),
                   headers={'Content-Type': 'application/json'})

        if r.status_code == 200:
            resp = r.json()
            self._rule_id = resp['ruleID']
            logging.debug('Successfully created localization rule')
        else:
            logging.error('Failed creating rule with status code: %d' % r.status_code)

        # set up some values for the per-frame task releasing thread
        self._posting_poll = True
        self._current_frame = 0

        # release task for each frame
        self._thread = threading.Thread(target=self._poll)
        self._thread.start()
        if thread_queue is not None:
            # keep track of number of launching threads to make sure they have time to finish before joining
            if thread_queue.full():
                thread_queue.get().join()
            thread_queue.put(self._thread)

    def _release_frame_tasks_for_bidding(self):
        n_frames = self.datasource.getNumSlices()
        logging.debug('datasource frames: %d, tasks filed: %d' % (n_frames, self._current_frame))
        n_outstanding = 0

        while n_frames > self._current_frame:
            # release in batches of 10,000
            new_current_frame = min(self._current_frame + 100000, n_frames)

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
            n_outstanding = n_frames - self._current_frame

        return n_outstanding

    def _poll(self):
        logger.debug('generating tasks for each frame')
        time.sleep(1.5)  # wait until clusterIO caches clear to avoid replicating the results file.

        while self._posting_poll:
            done = self.datasource.is_complete
            frames_outstanding = self._release_frame_tasks_for_bidding()
            if done and frames_outstanding < 1:
                # update max_tasks on ruleserver so the rule can eventually be marked as finished
                self.max_tasks = self.datasource.getNumSlices()
                #set up results file:
                clusterResults.fileResults(self.results_uri + '/Events', self.datasource.getEvents())
                logging.debug('all tasks generated, ending loop')
                self._posting_poll = False
                self._rule_id = None  # disconnect this rule from the ruleserver copy
            else:
                time.sleep(1)

    @property
    def outputs(self):
        cluster_resolved_filename = 'PYME-CLUSTER://%s/' % self.server_filter + self.results_filename
        # return [{  # fixme - recipe loading currently doesn't support get_tabular_part as separate inputs
        #     'fitResults':  cluster_resolved_filename + '/FitResults',
        #     'driftResults': cluster_resolved_filename + '/DriftResults'
        # }]
        return [{'input': cluster_resolved_filename}]

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
    def __init__(self, recipe, inputs=(), output_dir=None):
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

    @property
    def rule(self):
        rule = super(RecipeRule, self).rule

        if len(self._task_inputs) > 0:
            rule['inputsByTask'] = {t_ind: task for t_ind, task in enumerate(self._task_inputs)}
            logger.debug('inputs by task: %s' % rule['inputsByTask'])

        return rule

    def post(self, thread_queue=None):
        ruleserver_uri = _get_ruleserver_uri()

        self.max_tasks = max(len(self._task_inputs), 1)

        s = clusterIO._getSession(ruleserver_uri)
        r = s.post('%s/add_integer_id_rule?max_tasks=%d&release_start=%d&release_end=%d' % (ruleserver_uri, self.max_tasks, 0,
                                                                                            self.max_tasks),
                   data=json.dumps(self.rule), headers={'Content-Type': 'application/json'})

        if r.status_code == 200:
            resp = r.json()
            self._rule_id = resp['ruleID']
            logging.debug('Successfully created recipe rule')
        else:
            logging.error('Failed creating rule with status code: %d' % r.status_code)

        self._rule_id = None  # disconnect this rule from the ruleserver copy

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
            if not isinstance(inputs[0], dict) or len(inputs) > 1:
                raise TypeError('rule chaining recipes do not yet support taskInput style task generation')
            inputs = inputs[0]

        self._template['inputs'] = inputs
        if 'output_dir' not in self._template.keys():
            # ensure we have somewhere to save outputs
            import posixpath  # cluster filenames are posix separated
            self._template['output_dir'] = posixpath.split(list(inputs.values())[0])[0] + '/recipe_outputs'
        self._task_inputs = []


class RuleChain(list):
    """
    List of rules to be posted with inputs/outputs chained. Execution will be serial, and subsequent rules will only be
    posted/made available after all tasks for the preceding rule are completed.
    """
    def __init__(self, thread_queue=None):
        list.__init__(self)
        self.thread_queue = thread_queue
        self.posted = dispatch.Signal()

    def post(self):
        """
        Chains all rules such that the inputs/outputs are connected, and then nests rules such that they will all be
        in the 'on_completion' part of the first rule, which is then posted
        """

        # chain rules in opposite order so we nest the on_completions
        for ri in reversed(range(len(self) - 1)):
            self[ri].chain_rule(self[ri + 1])

        self[0].post(self.thread_queue)
        self.posted.send(self)

    def set_chain_input(self, inputs):
        """
        Set the first input of the chain, and propagate the input/output chaining through the list. Note that for
        LocalizationRule, chaining the input will create a results file on the cluster.
        Parameters
        ----------
        inputs: list
            inputs to be chained to the first rule in the list
        """
        if isinstance(inputs, dict):
            inputs = [inputs]
        self[0].chain_inputs(inputs)

        for ri in range(0, len(self) - 1):
            self[ri + 1].chain_inputs(self[ri].outputs)
