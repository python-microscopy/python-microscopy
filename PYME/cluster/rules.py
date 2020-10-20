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
>>>    RulePusher(step1.get_rule(context=context))

"""

import six
import json
import threading
import logging
logger = logging.getLogger(__name__)

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

        kwargs : any additional arguments (ignored in base class)

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
            rule['on_completion'] = on_completion.get_rule(context).rule
        
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
    
    @property
    def complete(self):
        """
        Is this rule complete, or do we need to poll for more input?
        
        Over-ridden in localisation rule
        Returns
        -------

        """
        return True
    
    def get_new_tasks(self):
        """
        Over-ridden in rules where all the data is not gauranteed to be present when the rule is created
        
        Returns
        -------
        
        release_start, release_end : the indices of starting and ending tasks to release

        """
        raise NoNewTasks('get_new_tasks() called in base class which assumes all tasks are released on rule creation')


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
        kwargs : any additional args to get passed to base class - e.g. on_completion

        TODO - support for templated recipes? Subclass?
        """
        Rule.__init__(**kwargs)
        
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
    
    def _populate_rule(self, context, on_completion=None):
        #over-ride here because we need to add input info
        rule = Rule._populate_rule(self, context, on_completion=on_completion)
        rule['inputsByTask'] = inputs_by_task
        
        return rule
    
    def _task_template(self, context):
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


from PYME.IO import DataSources, clusterIO, clusterResults
class LocalisationRule(Rule):
    def __init__(self, seriesName, analysisMetadata, resultsFilename=None, startAt=10, dataSourceModule=None, serverfilter=clusterIO.local_serverfilter, **kwargs):
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
    
    def _setup(self, dataSourceID, metadata, resultsFilename, startAt=10, dataSourceModule=None, serverfilter=clusterIO.local_serverfilter):
        self.dataSourceID = dataSourceID
        if '~' in self.dataSourceID or '~' in resultsFilename:
            raise RuntimeError('File, queue or results name must NOT contain ~')
    
        self.resultsURI = clusterResults.pickResultsServer('__aggregate_h5r/%s' % resultsFilename, serverfilter)
    
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
    
    def task_template(self, context):
        tt = {'id': '{{ruleID}}~{{taskID}}',
              'type': 'localization',
              'taskdef': {'frameIndex': '{{taskID}}', 'metadata': self.results_md_uri},
              'inputs': {'frames': self.dataSourceID},
              'outputs': {'fitResults': self.resultsURI + '/FitResults',
                          'driftResults': self.resultsURI + '/DriftResults'}
              }
        self._task_template = json.dumps(tt)

    def prepare(self):
        """
        Do any setup work - e.g. uploading metadata required before the rule is triggered

        Returns
        -------

        post_args : dict
            a dictionary with arguments to pass to RulePusher._post_rule() - specifically timeout, max_tasks, release_start, release_end

        """
        #set up results file:
        logging.debug('resultsURI: ' + self.resultsURI)
        clusterResults.fileResults(self.resultsURI + '/MetaData', self.mdh)
        clusterResults.fileResults(self.resultsURI + '/Events', self.ds.getEvents())

        # set up metadata file which is used for deciding how to launch the analysis
        clusterIO.put_file(self.resultsMDFilename, self.mdh.to_JSON().encode(), serverfilter=self.serverfilter)

        #wait until clusterIO caches clear to avoid replicating the results file.
        #time.sleep(1.5) #moved inside polling thread so launches will run quicker

        self.currentFrameNum = self.start_at
        numTotalFrames = self.ds.getNumSlices()
        self.frames_outstanding=numTotalFrames - 1 - self.currentFrameNum
        
        return {}

    @property
    def complete(self):
        """
        Is this rule complete, or do we need to poll for more input?

        Over-ridden in localisation rule
        Returns
        -------

        """
        return self.ds.is_complete and not (self.frames_outstanding  > 0)

    def get_new_tasks(self):
        """
        Over-ridden in rules where all the data is not gauranteed to be present when the rule is created

        Returns
        -------

        release_start, release_end : the indices of starting and ending tasks to release

        """
        numTotalFrames = self.ds.getNumSlices()
        logging.debug('numTotalFrames: %s, currentFrameNum: %d' % (numTotalFrames, self.currentFrameNum))

        if numTotalFrames <= (self.currentFrameNum + 1):
            raise NoNewTasks('not new localisation tasks available at this time')
        else:
            logging.debug('we have unpublished frames - push them')
            newFrameNum = min(self.currentFrameNum + 100000, numTotalFrames - 1)
            cur_frame = self.currentFrameNum
            self.currentFrameNum = newFrameNum
            self.frames_outstanding = numTotalFrames - 1 - self.currentFrameNum
            
            return cur_frame, newFrameNum
        
        
        

class RuleFactory(object):
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
        Populate a rule using series specific info from context
        
        Parameters
        ----------
        context : dict
            a dictionary containing series specific info to populate into the task template

        Returns
        -------
        
        a rule suitable for submitting to the ruleserver `/add_integer_id_rule` endpoint

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
        

class RecipeRuleFactory(RuleFactory):
    def __init__(self, **kwargs):
        RuleFactory.__init__(self, rule_class=RecipeRule, **kwargs)
        
class LocalisationRuleFactory(RuleFactory):
    def __init__(self, **kwargs):
        RuleFactory.__init__(self, rule_class=LocalisationRule, **kwargs)
        



import time
class RulePusher(object):
    def __init__(self, rule):
        self._rule = rule
        
        self._ruleID = None
        self.taskQueueURI = self._getTaskQueueURI()
        
        self._current_task_num = 0
        #self._complete = False
    
        #create any needed files - e.g. metadata
        post_args = self._rule.prepare()
        
        #post our rule
        self._post_rule(**post_args)

        if not self._rule.complete:
            #we haven't released all the frames yet, start a loop to poll and release frames as they become available.
            self.doPoll = True
            self.pollT = threading.Thread(target=self._poll_loop())
            self.pollT.start()
        
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

    def _post_rule(self, timeout=3600, max_tasks=1e6, release_start=None, release_end=None):
        """ wrapper around add_integer_rule api endpoint"""
        from PYME.IO import clusterIO
    
        s = clusterIO._getSession(self.taskQueueURI)
        if release_start is None:
            cmd = '%s/add_integer_id_rule?timeout=%d&max_tasks=%d' % (self.taskQueueURI, timeout, max_tasks)
        else:
            # TODO - can we get rid of this special casing?
            cmd = '%s/add_integer_id_rule?timeout=%d&max_tasks=%d&release_start=%d&release_end=%d' % (self.taskQueueURI, timeout, max_tasks, release_start, release_end)
            
        r = s.post(cmd,
                   data=json.dumps(self._rule.rule),
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
        self.taskQueueURI, self._ruleID, release_start, release_start),
                  data='',
                  headers={'Content-Type': 'application/json'})
    
        if r.status_code == 200 and r.json()['ok']:
            logging.debug('Successfully released tasks')
        else:
            logging.error('Failed on releasing tasks with status code: %d' % r.status_code)
        
            
    def _poll_loop(self):
        logging.debug('task pusher poll loop started')
        #wait until clusterIO caches clear to avoid replicating the results file.
        time.sleep(1.5)
    
        while (self.doPoll == True):
            try:
                rel_start, rel_end = self._rule.get_new_tasks()
                self._release_tasks(rel_start, rel_end)
            except NoNewTasks:
                pass
                
            if self._rule.complete:
                logging.debug('all tasks pushed, ending loop.')
                self.doPoll = False
            else:
                time.sleep(1)

    def cleanup(self):
        self.doPoll = False