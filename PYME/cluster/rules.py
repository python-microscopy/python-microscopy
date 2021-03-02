"""
Refactored rule pushing. Introduces rule classes which act as a python proxy for the JSON rule objects, and rule
factories for use when constructing equivalent rules (or chains of rules) for multiple series.

Design principles as follows:
- a `Rule` object is a 1:1 mapping with rules on the ruleserver
- you create a new `Rule` object for each rule you push to the server
- a pattern for rule creation is expressed using a `RuleFactory`
- calling `.get_rule() on the first step returns you a fully linked rule suitable for submitting
- very limited inference of inputs etc ... between steps - rely on specifying inputs and outputs using patterns instead

Examples
--------

>>> step1 = LocalisationRuleFactory(analysisMetadata=mdh)
>>> step2 = RecipeRuleFactory(recipeURI='PYME-CLUSTER///RECIPES/render_image.yaml', input_patterns={'input':'{{spool_dir}}/analysis/{{series_stub}}.h5r'})
>>> step1.chain(step2)
>>> step3 = RecipeRuleFactory(recipeURI='PYME-CLUSTER///RECIPES/measure_blobs.yaml', input_patterns={'input':'{{spool_dir}}/analysis/{{series_stub}}.tif'})
>>> step2.chain(step3)

or

>>> step1 = LocalisationRuleFactory(analysisMetadata=mdh,
>>>                                 on_completion=RecipeRuleFactory(recipeURI='PYME-CLUSTER///RECIPES/render_image.yaml',
>>>                                                                 input_patterns={'input':'{{spool_dir}}/analysis/{{series_stub}}.h5r'},
>>>                                                                 on_completion=RecipeRuleFactory(recipeURI='PYME-CLUSTER///RECIPES/measure_blobs.yaml',
>>>                                                                                                 input_patterns={'input':'{{spool_dir}}/analysis/{{series_stub}}.tif'})))

then:

>>> def on_launch_analysis(context={'spool_dir': ..., 'series_stub': ...}):
>>>    step1.get_rule(context=context).push()

"""

import six
import json
import threading
import time
import logging
logger = logging.getLogger(__name__)

from PYME.IO import DataSources, clusterIO, clusterResults, unifiedIO

class NoNewTasks(Exception):
    pass


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


class Rule(object):
    def __init__(self, on_completion=None, **kwargs):
        """
        Create a new rule. Sub-classed for specific rule types


        Parameters
        ----------
        on_completion : RuleFactory instance
            A rule to run after this one has completed (also setable using the `.chain()` method)

        kwargs : any additional arguments, available in the context used when creating the rule template

        """
        
        self.rule = self._populate_rule(context=kwargs, on_completion=on_completion)
    
    def _populate_rule(self, context, on_completion=None):
        """
        Populate a rule using series specific info from context

        Parameters
        ----------
        context : dict
            a dictionary containing series specific info to populate into the task template

        Returns
        -------

        a rule suitable for submitting to the ruleserver `/add_integer_id_rule` endpoint

        """
        rule = {'template': self._task_template(context)}
        
        if on_completion:
            chained_context = dict(**context)
            # standard recipe single-input key is 'input'. Set up 'results'
            # as alt key so we can use the same recipe in clusterUI views
            out_files = self.output_files
            if 'results' in out_files.keys() and 'input' not in out_files.keys():
                out_files = out_files.copy()
                out_files['input'] = out_files.pop('results')
            chained_context['rule_outputs'] = out_files
            rule['on_completion'] = on_completion.get_rule(chained_context).rule
        
        return rule
    
    def _task_template(self, context):
        """
        Populate the task template for a given rule type. Should be implemented in derived classes.

        Parameters
        ----------
        context : dict
            Series specific context information to use in the template

        Returns
        -------

        The task template as a string.

        """
        raise NotImplementedError('This method should be over-ridden in a derived class')
    
    def prepare(self):
        """
        Do any setup work - e.g. uploading metadata required before the rule is triggered
        
        Returns
        -------
        
        post_args : dict
            a dictionary with arguments to pass to RulePusher._post_rule() - specifically timeout, max_tasks, release_start, release_end

        """
        
        return {}
    
    def _output_files(self):
        """ Return a dictionary of output cluster URIs.
        The principle output (if it makes sense to define one) should be under the 'results' key. Each entry should be a
        single output as this gets substituted into the recipe before submission (not in the ruleserver):
        
        For a single output, the dictionary should look like: {'results' : results_URI}
        
        Over-ride in derived rule classes
        """
        return {}
    
    @property
    def output_files(self):
        return self._output_files()
    
    @property
    def complete(self):
        """
        Is this rule complete, or do we need to poll for more input?
        
        Over-ridden in localisation rule
        Returns
        -------

        """
        return True

    @property
    def data_complete(self):
        """ Is the underlying data complete?"""
        return True
    
    def on_data_complete(self):
        '''Over-ride in derived rules so that, e.g. events can be written at the end of a real-time acquisition '''
        pass
    
    def get_new_tasks(self):
        """
        Over-ridden in rules where all the data is not guaranteed to be present when the rule is created
        
        Returns
        -------
        
        release_start, release_end : the indices of starting and ending tasks to release

        """
        raise NoNewTasks('get_new_tasks() called in base class which assumes all tasks are released on rule creation')
    
    @classmethod
    def _getTaskQueueURI(cls, n_retries=2):
        """Discover the distributors using zeroconf and choose one"""
        from PYME.misc import hybrid_ns
        import socket
        import random
        import time
        from PYME.misc.computerName import GetComputerName
        compName = GetComputerName()
    
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
            local_queues = [q for q in queueURLs if compName in q]
            logger.debug('local_queues: %s' % local_queues)
            return queueURLs[local_queues[0]]
        except (KeyError, IndexError):
            #if there is no local distributor, choose one at random
            logger.info('no local rule server, choosing one at random')
            return random.choice(list(queueURLs.values()))
    
    def push(self):
        self._ruleID = None
        self.taskQueueURI = self._getTaskQueueURI()
        self._current_task_num = 0
    
        #create any needed files - e.g. metadata
        post_args = self.prepare()
    
        #post our rule
        self._post_rule(**post_args)
    
        if not self.complete:
            #we haven't released all the frames yet, start a loop to poll and release frames as they become available.
            self.doPoll = True
            self.pollT = threading.Thread(target=self._poll_loop)
            self.pollT.start()
        else:
            self.on_data_complete()

    def _post_rule(self, timeout=3600, max_tasks=1e6, release_start=None, release_end=None):
        """ wrapper around add_integer_rule api endpoint"""
        from PYME.IO import clusterIO
    
        s = clusterIO._getSession(self.taskQueueURI)
        if release_start is None:
            cmd = '%s/add_integer_id_rule?timeout=%d&max_tasks=%d' % (self.taskQueueURI, timeout, max_tasks)
        else:
            # TODO - can we get rid of this special casing?
            cmd = '%s/add_integer_id_rule?timeout=%d&max_tasks=%d&release_start=%d&release_end=%d' % (
            self.taskQueueURI, timeout, max_tasks, release_start, release_end)
    
        r = s.post(cmd,
                   data=json.dumps(self.rule),
                   headers={'Content-Type': 'application/json'})
    
        if r.status_code == 200:
            resp = r.json()
            self._ruleID = resp['ruleID']
            logger.debug('Successfully created rule')
        else:
            logger.error('Failed creating rule with status code: %d' % r.status_code)

    def _release_tasks(self, release_start, release_end):
        """ Thin wrapper around release_rule_tasks api endpoint"""
        from PYME.IO import clusterIO
        s = clusterIO._getSession(self.taskQueueURI)
        r = s.get('%s/release_rule_tasks?ruleID=%s&release_start=%d&release_end=%d' % (
            self.taskQueueURI, self._ruleID, release_start, release_end),
                  data='',
                  headers={'Content-Type': 'application/json'})
    
        if r.status_code == 200 and r.json()['ok']:
            logging.debug('Successfully released tasks (%d:%d)' % (release_start, release_end))
        else:
            logging.error('Failed on releasing tasks with status code: %d' % r.status_code)

    def _mark_complete(self):
        """ Thin wrapper around release_rule_tasks api endpoint"""
        from PYME.IO import clusterIO
        s = clusterIO._getSession(self.taskQueueURI)
        r = s.get('%s/mark_release_complete?ruleID=%s' % (
            self.taskQueueURI, self._ruleID),
                  data='',
                  headers={'Content-Type': 'application/json'})
    
        if r.status_code == 200 and r.json()['ok']:
            logging.debug('Successfully marked rule as complete')
        else:
            logging.error('Failed to mark rule complete with status code: %d' % r.status_code)

    def _poll_loop(self):
        logging.debug('task pusher poll loop started')
    
        while (self.doPoll == True):
            try:
                rel_start, rel_end = self.get_new_tasks()
                self._release_tasks(rel_start, rel_end)
            except NoNewTasks:
                pass
        
            if self.data_complete and not hasattr(self, '_data_comp_callback_res'):
                logger.debug('input data complete, calling on_data_complete')
                self._data_comp_callback_res=self.on_data_complete()
            
            if self.complete:
                logger.debug('input data complete')
                try:  # check/relase one more time to avoid race condition
                    rel_start, rel_end = self.get_new_tasks()
                    self._release_tasks(rel_start, rel_end)
                except NoNewTasks:
                    pass
                logger.debug('all tasks pushed, marking rule as complete')
                self._mark_complete()
                logging.debug('ending polling loop.')
                self.doPoll = False
            else:
                time.sleep(1)

    def cleanup(self):
        self.doPoll = False


class RecipeRule(Rule):
    def __init__(self, recipe=None, recipeURI=None, output_dir=None, **kwargs):
        """
        Create a recipe rule

        Parameters
        ----------
        recipe : str or recipes.modules.ModuleCollection
            The recipe as YAML text or as a recipe instance (alternatively provide recipeURL)
        recipeURI : str
            A cluster URI for the recipe text (if `recipe` not provided directly)
        output_dir : str
            The directory to put the recipe output TODO: should this be templated based on context?
        kwargs : dict
            Parameters for the base `Rule` class (notably `on_completion`, which, if provided, should be a RuleFactory instance).
            Additional parameters, not consumed by `Rule` are accessible in the context used for creating rule templates.  
            
            One (and only one) of the following keyword parameters should be provided to specify the recipe inputs:
                inputs : dict
                    keys are recipe namespace keys, values are either lists of file URIs, or globs which will be expanded 
                    to a list of URIs. Corresponds to the `inputsByTask` property in the recipe description (see `ruleserver` docs).
                    `inputsByTask` will be used by the server to populate the inputs for individual tasks 
                input_templates : dict
                    simplar to inputs, except that dictionary substitution with the rule context is perfromed on the values before
                    they are written to `inputsByTasks`
                rule_outputs :  dict
                    used with chained recipes, this is the outputs of the previous recipe step. Unlike `inputs` and `input_templates`
                    this is a dict of str, rather than of list (or glob-implied list) and is written directly to the `"inputs"` section
                    of the template, rather than to the `inputsByTask` property of the rule, short-circuiting server side input filling.
        
        TODO - support for templated recipes? Subclass?
        """
        
        if recipe:
            if isinstance(recipe, six.string_types):
                self.recipe_text = recipe
            else:
                self.recipe_text = recipe.toYAML()
            
            self.recipeURI = None
        else:
            if recipeURI is None:
                raise ValueError('recipeURI must be defined if no recipe given')
            else:
                self.recipeURI = recipeURI
        
        self.output_dir = output_dir

        Rule.__init__(self, **kwargs)
    
    def _populate_rule(self, context, on_completion=None):
        # over-ride here because we need to add input info
        
        # try (in order of increasing precedence):
        #
        # - an `input_templates` variable in the context
        # - hard coded `inputs` in the context
        inputs = {} # = context.get('rule_outputs', {})
        input_templates = context.get('input_templates', None)
        if input_templates is not None:
            inputs.update({k: v.format(context) for k, v in input_templates.items()})
        
        inputs.update(context.get('inputs', {}))

        if not inputs:
            if not context.get('rule_outputs', False):
                # NOTE - rule_outputs is handled in _task_template.
                raise RuntimeError('No inputs found, one of "inputs", "input_templates", or "rule_outputs" must be present in context')
            inputs_by_task = None
        else:
            input_names = inputs.keys()
            
            def _to_input_list(v):
                # TODO - move this fcn definition??
                if isinstance(v, list):
                    return v
                else:
                    # value is a string glob
                    name, serverfilter = unifiedIO.split_cluster_url(v)
                    return clusterIO.cglob(name, serverfilter)
                    
            inputs = {k: _to_input_list(inputs[k]) for k in input_names}
    
            self._num_recipe_tasks = len(list(inputs.values())[0])
    
            logger.debug('numTotalFrames = %d' % self._num_recipe_tasks)
            logger.debug('inputs = %s' % inputs)
    
            inputs_by_task = {frameNum: {k: inputs[k][frameNum] for k in inputs.keys()} for frameNum in
                              range(self._num_recipe_tasks)}
            
            
        rule = Rule._populate_rule(self, context, on_completion=on_completion)
        
        if inputs_by_task:
            rule['inputsByTask'] = inputs_by_task
        
        return rule
    
    def prepare(self):
        return dict(max_tasks=self._num_recipe_tasks, release_start=0, release_end=self._num_recipe_tasks)
    
    def _task_template(self, context):
        task = '''{"id": "{{ruleID}}~{{taskID}}",
                    "type": "recipe",
                    "inputs" : {{taskInputs}},
                    %s
                }'''
        
        if self.output_dir is None:
            output_dir_n = ''
        else:
            output_dir_n = '"output_dir": "%s",\n    ' % self.output_dir
    
        if self.recipeURI:
            task = task % (output_dir_n + '"taskdefRef" : "%s"' % self.recipeURI)
        else:
            task = task % (output_dir_n + '"taskdef" : {"recipe": "%s"}' % self.recipe_text)
            
        #if we are a chained rule, hard-code inputs
        rule_outputs = context.get('rule_outputs', None)
        if rule_outputs is not None:
            #logger.debug(rule_outputs)
            task = task.replace('{{taskInputs}}', json.dumps(rule_outputs))
            logger.debug(task)
        
        return task



class LocalisationRule(Rule):
    def __init__(self, seriesName, analysisMetadata, resultsFilename=None, startAt=0, dataSourceModule=None, serverfilter=clusterIO.local_serverfilter, **kwargs):
        from PYME.IO import MetaDataHandler
        from PYME.Analysis import MetaData
        from PYME.IO.FileUtils.nameUtils import genClusterResultFileName
        from PYME.IO import unifiedIO
    
        unifiedIO.assert_uri_ok(seriesName)
    
        if resultsFilename is None:
            resultsFilename = genClusterResultFileName(seriesName)
        
        resultsFilename = verify_cluster_results_filename(resultsFilename)
        logger.info('Results file: ' + resultsFilename)
    
        resultsMdh = MetaDataHandler.NestedClassMDHandler()
        # NB - anything passed in analysis MDH will wipe out corresponding entries in the series metadata
        resultsMdh.update(json.loads(unifiedIO.read(seriesName + '/metadata.json')))
        resultsMdh.update(analysisMetadata)
    
        resultsMdh['EstimatedLaserOnFrameNo'] = resultsMdh.getOrDefault('EstimatedLaserOnFrameNo',
                                                                        resultsMdh.getOrDefault('Analysis.StartAt', 0))
        MetaData.fixEMGain(resultsMdh)
        
        
        self._setup(seriesName, resultsMdh, resultsFilename, startAt, dataSourceModule, serverfilter)
        
        Rule.__init__(self, **kwargs)
        
    def _output_files(self):
        return {'results' : self.resultsURI}
    
    def _setup(self, dataSourceID, metadata, resultsFilename, startAt=0, dataSourceModule=None, serverfilter=clusterIO.local_serverfilter):
        self.dataSourceID = dataSourceID
        if '~' in self.dataSourceID or '~' in resultsFilename:
            raise RuntimeError('File, queue or results name must NOT contain ~')
    
        
        #where the results are when we want to read them
        self.resultsURI = 'PYME-CLUSTER://%s/%s' % (serverfilter, resultsFilename)
        
        # it's faster (and safer for race condition avoidance) to pick a server in advance and give workers the direct
        # HTTP endpoint to write to. This should also be an aggregate endpoint, as multiple writes are needed.
        self.worker_resultsURI = clusterResults.pickResultsServer('__aggregate_h5r/%s' % resultsFilename, serverfilter)
    
        self.resultsMDFilename = resultsFilename + '.json'
        self.results_md_uri = 'PYME-CLUSTER://%s/%s' % (serverfilter, self.resultsMDFilename)
    
        self.mdh = metadata
        self.start_at = startAt
        self.serverfilter = serverfilter
    
        #load data source
        if dataSourceModule is None:
            DataSource = DataSources.getDataSourceForFilename(dataSourceID)
        else:
            DataSource = __import__('PYME.IO.DataSources.' + dataSourceModule,
                                    fromlist=['PYME', 'io', 'DataSources']).DataSource #import our data source
    
        self.ds = DataSource(self.dataSourceID)
    
        logger.debug('DataSource.__class__: %s' % self.ds.__class__)
    
    def _task_template(self, context):
        tt = {'id': '{{ruleID}}~{{taskID}}',
              'type': 'localization',
              'taskdef': {'frameIndex': '{{taskID}}', 'metadata': self.results_md_uri},
              'inputs': {'frames': self.dataSourceID},
              'outputs': {'fitResults': self.worker_resultsURI + '/FitResults',
                          'driftResults': self.worker_resultsURI + '/DriftResults'}
              }
        return json.dumps(tt)

    def prepare(self):
        """
        Do any setup work - e.g. uploading metadata required before the rule is triggered

        Returns
        -------

        post_args : dict
            a dictionary with arguments to pass to RulePusher._post_rule() - specifically timeout, max_tasks, release_start, release_end

        """
        #set up results file:
        logging.debug('resultsURI: ' + self.worker_resultsURI)
        clusterResults.fileResults(self.worker_resultsURI + '/MetaData', self.mdh)
        
        # defer copying events to after series completion
        #clusterResults.fileResults(self.worker_resultsURI + '/Events', self.ds.getEvents())

        # set up metadata file which is used for deciding how to launch the analysis
        clusterIO.put_file(self.resultsMDFilename, self.mdh.to_JSON().encode(), serverfilter=self.serverfilter)

        #wait until clusterIO caches clear to avoid replicating the results file.
        #time.sleep(1.5) #moved inside polling thread so launches will run quicker

        self._next_release_start = self.start_at
        self.frames_outstanding=self.total_frames - self._next_release_start
        if self.data_complete:
            return dict(max_tasks=self.total_frames)
        return {}
    
    @property
    def total_frames(self):
        return self.ds.getNumSlices()

    @property
    def data_complete(self):
        return self.ds.is_complete
    
    @property
    def complete(self):
        """
        Is this rule complete, or do we need to poll for more input?

        Over-ridden in localisation rule
        Returns
        -------

        """
        return self.data_complete and not (self.frames_outstanding  > 0)
    
    def on_data_complete(self):
        logger.debug('Data complete, copying events to output file')
        clusterResults.fileResults(self.worker_resultsURI + '/Events', self.ds.getEvents())

    def get_new_tasks(self):
        """
        Over-ridden in rules where all the data is not gauranteed to be present when the rule is created

        Returns
        -------

        release_start, release_end : the indices of starting and ending tasks to release

        """
        numTotalFrames = self.total_frames
        logging.debug('numTotalFrames: %s, _next_release_start: %d' % (numTotalFrames, self._next_release_start))

        if numTotalFrames <= self._next_release_start:
            raise NoNewTasks('not new localisation tasks available at this time')
        else:
            logging.debug('we have unpublished frames - push them')
            release_end = min(self._next_release_start + 100000, numTotalFrames)
            release_start = self._next_release_start
            self._next_release_start = release_end #note - we use standard slice indexing where release_end = last_frame_idx +1 = the start idx of the next release
            self.frames_outstanding = numTotalFrames - self._next_release_start
            
            return release_start, release_end
        

class SpoolLocalLocalizationRule(LocalisationRule):
    def __init__(self, spooler, seriesName, analysisMetadata, resultsFilename=None, startAt=0, serverfilter=clusterIO.local_serverfilter, **kwargs):
        # TODO - reduce duplication of `LocalisationRule.__init__()` and `LocalisationRule._setup()`
        from PYME.IO import MetaDataHandler
        from PYME.Analysis import MetaData
        from PYME.IO.FileUtils.nameUtils import genClusterResultFileName
        from PYME.IO import unifiedIO

        self.spooler = spooler
    
        if resultsFilename is None:
            resultsFilename = genClusterResultFileName(seriesName)
        
        resultsFilename = verify_cluster_results_filename(resultsFilename)
        logger.info('Results file: ' + resultsFilename)
    
        resultsMdh = MetaDataHandler.DictMDHandler()
        # NB - anything passed in analysis MDH will wipe out corresponding entries in the series metadata
        resultsMdh.update(self.spooler.md)
        resultsMdh.update(analysisMetadata)
        resultsMdh['EstimatedLaserOnFrameNo'] = resultsMdh.getOrDefault('EstimatedLaserOnFrameNo',
                                                                        resultsMdh.getOrDefault('Analysis.StartAt', 0))
        MetaData.fixEMGain(resultsMdh)
        
        self._setup(resultsMdh, resultsFilename, startAt, serverfilter)
        
        Rule.__init__(self, **kwargs)

    def _setup(self, metadata, resultsFilename, startAt=0, serverfilter=clusterIO.local_serverfilter):
        #where the results are when we want to read them
        self.resultsURI = 'PYME-CLUSTER://%s/%s' % (serverfilter, resultsFilename)
        
        # it's faster (and safer for race condition avoidance) to pick a server in advance and give workers the direct
        # HTTP endpoint to write to. This should also be an aggregate endpoint, as multiple writes are needed.
        self.worker_resultsURI = clusterResults.pickResultsServer('__aggregate_h5r/%s' % resultsFilename, serverfilter)
    
        self.resultsMDFilename = resultsFilename + '.json'
        self.results_md_uri = 'PYME-CLUSTER://%s/%s' % (serverfilter, self.resultsMDFilename)
    
        self.mdh = metadata
        self.start_at = startAt
        self.serverfilter = serverfilter

    @property
    def total_frames(self):
        return self.spooler.get_n_frames()

    @property
    def data_complete(self):
        try:
            return self.spooler.finished()
        except RuntimeError as e:
            logger.error(str(e))
            logger.info('marking data as complete')
            return True
    
    def on_data_complete(self):
        logger.debug('Data complete, copying events to output file')
        clusterResults.fileResults(self.worker_resultsURI + '/Events', self.spooler.evtLogger.to_JSON())


class RuleFactory(object):
    _type = ''
    def __init__(self, on_completion=None, rule_class = Rule, **kwargs):
        """
        Create a new rule factory. Sub-classed for specific rule types
        
        
        Parameters
        ----------
        on_completion : RuleFactory instance
            A rule to run after this one has completed (also setable using the `.chain()` method)
             
        kwargs : any additional arguments (ignored in base class)
        
        """
        self._on_completion = on_completion
        self._rule_class = rule_class
        self._rule_kwargs = kwargs
        
    def get_rule(self, context):
        """
        Populate a rule using series specific info from context. Note that
        the the rule class initialization arguments should be passed in the
        RuleFactory initialization as kwargs, but can also be passed here in
        context if, e.g. the series name is not known when creating the 
        
        Parameters
        ----------
        context : dict
            a dictionary containing series specific info to populate into the task template

        Returns
        -------
        Rule
            a rule suitable for submitting to the ruleserver `/add_integer_id_rule`
            endpoint

        """
        
        d = dict(self._rule_kwargs)
        d.update(context)
        
        return self._rule_class(on_completion=self._on_completion, **d)
    
    def chain(self, on_completion):
        """
        Chain a rule (as an alternative to passing on_completion to the rule constructor, allows us to create rule chains
        from front to back rather than back to front
        
        Parameters
        ----------
        
        on_completion : RuleFactory instance
             The rule to run after this one has completed
        """
        assert(isinstance(on_completion, RuleFactory))
        self._on_completion = on_completion

    @property
    def rule_type(self):
        # TODO - rename to something like `display_name` of `display_type` to indicate that this is a UI helper function.
        # TODO - do we need a function for this, or can we just have a property?
        return self._type


class RecipeRuleFactory(RuleFactory):
    _type = 'recipe'
    def __init__(self, **kwargs):
        RuleFactory.__init__(self, rule_class=RecipeRule, **kwargs)
        
class LocalisationRuleFactory(RuleFactory):
    _type = 'localization'
    def __init__(self, **kwargs):
        """
        See `LocalisationRule` for full initialization arguments. Required 
        kwargs are
            seriesName : str
            analysisMetadata : PYME.IO.MetaDataHandler.MDHandlerBase
        """
        RuleFactory.__init__(self, rule_class=LocalisationRule, **kwargs)   

class SpoolLocalLocalizationRule(RuleFactory):
    _type = 'localization'
    def __init__(self, **kwargs):
        """
        See `SpoolLocalLocalizationRule` for full initialization arguments. 
        Required kwargs are
            seriesName : str
            analysisMetadata : PYME.IO.MetaDataHandler.MDHandlerBase
        """
        RuleFactory.__init__(self, rule_class=SpoolLocalLocalizationRule, **kwargs)   
