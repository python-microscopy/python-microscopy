#!/usr/bin/python

##################
# taskWorkerM.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

#!/usr/bin/python
#import Pyro.core
#import Pyro.naming
import os
import platform
import time

import matplotlib

matplotlib.use('SVG')

import queue as Queue
import threading

#import PYME.misc.pyme_zeroconf as pzc
from PYME import config
from PYME.misc.computerName import GetComputerName
compName = GetComputerName()

import logging
import logging.handlers
dataserver_root = config.get('dataserver-root')
if dataserver_root:
    log_dir = '%s/LOGS/%s/taskWorkerHTTP' % (dataserver_root, compName)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except:
            #as we are launching multiple processes at once, there is a race condition here and we might
            #have already created the directory between our test and the makedirs call
            pass
        
    #fh = logging.FileHandler('%s/%d.log' % (log_dir, os.getpid()), 'w')
    #fh.setLevel(logging.DEBUG)
    #logger.addHandler(fh)
    #logging.basicConfig(filename ='%s/%d.log' % (log_dir, os.getpid()), level=logging.DEBUG)
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    fh = logging.handlers.RotatingFileHandler(filename ='%s/%d.log' % (log_dir, os.getpid()), mode='w', maxBytes=1e6)
    logger.addHandler(fh)
else:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('')


import requests
import sys
import signal
import yaml

def str_presenter(dumper, data):
  if len(data.splitlines()) > 1:  # check for multiline string
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
  return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)
#import socket

from PYME.localization import remFitBuf
from PYME.cluster import distribution

#import here to pre-populate the zeroconf nameserver
from PYME.IO import clusterIO
from PYME.IO import unifiedIO
from PYME.IO import clusterResults
#time.sleep(3)


LOCAL = False
if 'PYME_LOCAL_ONLY' in os.environ.keys():
    LOCAL = os.environ['PYME_LOCAL_ONLY'] == '1'
    
class TaskError(object):
    template = '''===========================================================================
Error in rule: {rule_id} running task: {task_id} on {comp_name}:{pid}

taskDescr:
----------
{taskDescr}

Traceback:
----------
{traceback}

'''
    html_error_template = '''
<h5>Error in rule: {rule_id} running task: {task_id} on {comp_name}:{pid}</h5>

<h6>taskDescr:</h6>
<pre style="text-size:8pt;">{taskDescr}</pre>

<h6>Traceback:</h6>
<pre style="text-size:8pt;">{traceback}</pre>

<hrule>
'''

    def __init__(self, taskDescr, traceback, exception=None):
        self.taskDescr = taskDescr
        self.traceback = traceback
        self.exception = exception
        
    @property
    def log_url(self):
        rule_id = self.taskDescr['id'].split('~')[0]
        return 'PYME-CLUSTER://%s/__aggregate_txt/LOGS/rules/%s.log' % (clusterIO.local_serverfilter, rule_id)

    @property
    def html_log_url(self):
        rule_id = self.taskDescr['id'].split('~')[0]
        return 'PYME-CLUSTER://%s/__aggregate_txt/LOGS/rules/%s.html' % (clusterIO.local_serverfilter, rule_id)
        
    def format_taskDesc(self):
        return yaml.dump(self.taskDescr)

    def extra_template_kwargs(self, mode='txt'):
        """ Over-ride in derived classes to provide extra info to templates"""
        return {}
    
    def to_string(self, mode='txt'):
        rule_id, task_id = self.taskDescr['id'].split('~')

        if mode == 'html':
            template = self.html_error_template
        else:
            template = self.template
        
        return template.format(rule_id = rule_id, task_id = task_id, comp_name = compName, pid=os.getpid(),
                                    taskDescr= self.format_taskDesc(), traceback = self.traceback, **self.extra_template_kwargs(mode=mode))

class RecipeTaskError(TaskError):
    template = '''===========================================================================
Error in rule: {rule_id} running task: {task_id} on {comp_name}:{pid}

taskDescr:
----------
{taskDescr}

Recipe:
-------
{recipe}

Traceback:
----------
{traceback}

'''
    html_error_template = '''
<h5>Error in rule: {rule_id} running task: {task_id} on {comp_name}:{pid}</h5>

<h6>taskDescr:</h6>
<pre style="font-size: 8pt;">{taskDescr}</pre>

<h6>Recipe:</h6>
{recipe}

<h6>Traceback:</h6>
<pre style="font-size: 8pt;color: darkred;">{traceback}</pre>

<hrule>
'''
    def format_taskDesc(self):
        import copy
        # copy description, as we are going to override the recipe entry
        desc = copy.deepcopy(self.taskDescr)
        if 'recipe' in desc['taskdef']:
            desc['taskdef']['recipe'] = 'see below ...'
        
        return yaml.dump(desc)

    def extra_template_kwargs(self, mode='text'):
        from PYME.recipes.recipe import RecipeExecutionError
        if (mode == 'html'):
            if isinstance(self.exception, RecipeExecutionError) and self.exception.recipe:
                return {'recipe' : self.exception.recipe.to_html()}
            else:
                return {'recipe': '<pre style="font-size: 8pt;">%s</pre>' % self.taskDescr['taskdef'].get('recipe', 'see taskdef.recipeURI')}
        return {'recipe': self.taskDescr['taskdef'].get('recipe', 'see taskdef.recipeURI')}
        

class taskWorker(object):
    def __init__(self, nodeserver_port):
        self.inputQueue = Queue.Queue()
        self.resultsQueue = Queue.Queue()

        self.procName = '%s_%d' % (compName, os.getpid())
        
        self._nodeserver_port = nodeserver_port
        self._local_queue_url = 'http://127.0.0.1:%d/' % self._nodeserver_port

        self._loop_alive = True

    def loop_forever(self):
        self.tCompute = threading.Thread(target=self.computeLoop)
        self.tCompute.daemon = True
        self.tCompute.start()

        self.tI = threading.Thread(target=self.tasksLoop)
        self.tI.daemon = True
        self.tI.start()

        self.tO = threading.Thread(target=self.returnLoop)
        self.tO.daemon = True
        self.tO.start()

        try:
            while True:
                time.sleep(1)
        finally:
            self._loop_alive = False

    def _return_task_results(self):
        """

        File all results that this worker has completed

        Returns
        -------

        """
        while True:  # loop over results queue until it's empty
            # print 'getting results'
            try:
                queueURL, taskDescr, res = self.resultsQueue.get_nowait()
                outputs = taskDescr.get('outputs', {})
            except Queue.Empty:
                # queue is empty
                return

            if isinstance(res, TaskError):
                # failure
                clusterResults.fileResults(res.log_url, res.to_string().encode())
                clusterResults.fileResults(res.html_log_url, res.to_string(mode='html').encode())
                
                s = clusterIO._getSession(queueURL)
                r = s.post(queueURL + 'node/handin?taskID=%s&status=failure' % taskDescr['id'])
                if not r.status_code == 200:
                    logger.error('Returning task failed with error: %s' % r.status_code)
            elif res is None:
                # failure
                s = clusterIO._getSession(queueURL)
                r = s.post(queueURL + 'node/handin?taskID=%s&status=failure' % taskDescr['id'])
                if not r.status_code == 200:
                    logger.error('Returning task failed with error: %s' % r.status_code)
            
            elif res == True:  # isinstance(res, ModuleCollection): #recipe output
                # res.save(outputs) #abuse outputs dictionary as context

                s = clusterIO._getSession(queueURL)
                r = s.post(queueURL + 'node/handin?taskID=%s&status=success' % taskDescr['id'])
                if not r.status_code == 200:
                    logger.error('Returning task failed with error: %s' % r.status_code)

            else:
                # success
                try:
                    if 'results' in outputs.keys():
                        # old style pickled results
                        clusterResults.fileResults(outputs['results'], res)
                    else:
                        if len(res.results) > 0:
                            clusterResults.fileResults(outputs['fitResults'], res.results)

                        if len(res.driftResults) > 0:
                            clusterResults.fileResults(outputs['driftResults'], res.driftResults)
                except requests.Timeout:
                    logger.exception('Filing results failed on timeout.')
                    s = clusterIO._getSession(queueURL)
                    r = s.post(queueURL + 'node/handin?taskID=%s&status=failure' % taskDescr['id'])
                    if not r.status_code == 200:
                        logger.error('Returning task failed with error: %s' % r.status_code)
                else:
                    s = clusterIO._getSession(queueURL)
                    r = s.post(queueURL + 'node/handin?taskID=%s&status=success' % taskDescr['id'])
                    if not r.status_code == 200:
                        logger.error('Returning task failed with error: %s' % r.status_code)

    def _get_tasks(self):
        """

        Query nodeserver for tasks and place them in the queue for this worker,
        if available

        Returns
        -------
        new_tasks : bool
            flag to report whether _get_tasks added new tasks to the taskWorker queue

        """
        tasks = []
        queueURL = self._local_queue_url

        try:
            # ask the queue for tasks
            s = clusterIO._getSession(queueURL)
            r = s.get(queueURL + 'node/tasks?workerID=%s&numWant=50' % self.procName)
            if r.status_code == 200:
                resp = r.json()
                if resp['ok']:
                    res = resp['result']
                    if isinstance(res, list):
                        tasks += [(queueURL, t) for t in res]
                    else:
                        tasks.append((queueURL, res))
        except requests.Timeout:
            logger.info('Read timout requesting tasks from %s' % queueURL)

        except Exception:
            import traceback
            logger.exception(traceback.format_exc())

        if len(tasks) != 0:
            for t in tasks:
                self.inputQueue.put(t)
            return True
        else:
            # flag that there were no new tasks
            return False

    def tasksLoop(self):
        """

        Loop forever asking for tasks to queue up for this worker

        Returns
        -------

        """
        while True:
            if not self._loop_alive:
                break

            # if our queue for computing is empty, try to get more tasks
            if self.inputQueue.empty():
                # if we don't have any new tasks, sleep to avoid constant polling
                if not self._get_tasks():
                    # no queues had tasks
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
                    
    def returnLoop(self):
        """

        Loop forever returning task results

        Returns
        -------

        """
        while True:
            # turn in completed tasks
            try:
                self._return_task_results()
            except:
                import traceback
                logger.exception(traceback.format_exc())

            if not self._loop_alive:
                break

            time.sleep(1)


    def computeLoop(self):
        while self._loop_alive:
            #loop over tasks - we pop each task and then delete it after processing
            #to keep memory usage down

            queueURL, taskDescr = self.inputQueue.get()
            if taskDescr['type'] == 'localization':
                try:
                    task = remFitBuf.createFitTaskFromTaskDef(taskDescr)
                    res = task()

                    self.resultsQueue.put((queueURL, taskDescr, res))

                except:
                    import traceback
                    traceback.print_exc()
                    tb = traceback.format_exc()
                    logger.exception(tb)
                    self.resultsQueue.put((queueURL, taskDescr, TaskError(taskDescr, tb)))
                    #self.resultsQueue.put((queueURL, taskDescr, None))

            elif taskDescr['type'] == 'recipe':
                from PYME.recipes import Recipe
                from PYME.recipes import modules

                try:
                    taskdefRef = taskDescr.get('taskdefRef', None)
                    if taskdefRef: #recipe is defined in a file - go find it
                        recipe_yaml = unifiedIO.read(taskdefRef)
                        
                    else: #recipe is defined in the task
                        recipe_yaml = taskDescr['taskdef']['recipe']

                    recipe = Recipe.fromYAML(recipe_yaml)

                    #initial context
                    context = {'data_root' : clusterIO.local_dataroot,
                               'task_id' : taskDescr['id'].split('~')[0]}
                    
                    #load recipe inputs
                    logging.debug(taskDescr)
                    for key, url in taskDescr['inputs'].items():
                        if key == '__sim':
                            # special case for no-input simulation recipes
                            # for now, essentially ignore `__sim` inputs, but propagate into context just in case
                            # TODO?? find a way of encoding simulation parameters?
                            context['sim_tag'] = url
                        else:    
                            logging.debug('RECIPE: loading %s as %s' % (url, key))
                            recipe.loadInput(url, key)

                    #print recipe.namespace
                    recipe.execute()

                    #update context with file stub and input directory
                    try:
                        principle_input = taskDescr['inputs']['input'] #default input
                        context['file_stub'] = os.path.splitext(os.path.basename(principle_input))[0]
                        context['input_dir'] = unifiedIO.dirname(principle_input)
                    except KeyError:
                        pass

                    try:
                        od = taskDescr['output_dir']
                        # make sure we have a trailing slash
                        # TODO - this should be fine for most windows use cases, as you should generally
                        # use POSIX urls for the cluster/cluster of one, but might need checking
                        if not od.endswith('/'):
                            od = od + '/'
                            
                        context['output_dir'] = unifiedIO.dirname(od)
                    except KeyError:
                        pass

                    #print taskDescr['inputs']
                    #print context

                    #abuse outputs as context
                    outputs = taskDescr.get('outputs', None)
                    if not outputs is None:
                        context.update(outputs)
                    #print context, context['input_dir']
                    recipe.save(context)

                    self.resultsQueue.put((queueURL, taskDescr, True))

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    tb = traceback.format_exc()
                    logger.exception(tb)
                    self.resultsQueue.put((queueURL, taskDescr, RecipeTaskError(taskDescr, tb, e)))

        
def on_SIGHUP(signum, frame):
    raise RuntimeError('Recieved SIGHUP')
    

from argparse import ArgumentParser

if __name__ == '__main__':
    op = ArgumentParser(description="PYME rule server for task distribution. This should run once per cluster.")
    
    op.add_argument('-p', dest='profile', default=False, action='store_true',
                    help="enable profiling")
    op.add_argument('--profile-dir', dest='profile_dir')
    op.add_argument('-s', '--server-port', dest='server_port', type=int, default=config.get('nodeserver-port', 15347),
                    help='Optionally restrict advertisements to local machine')
    
    args = op.parse_args()
    
    if args.profile:
        from PYME.util import mProfile
        mProfile.profileOn(['taskWorkerHTTP.py', 'remFitBuf.py'])
        
    if not platform.platform().startswith('Windows'):
        signal.signal(signal.SIGHUP, on_SIGHUP)
    
    try:
        #main()
        tW = taskWorker(nodeserver_port=args.server_port)
        tW.loop_forever()
    except KeyboardInterrupt:
        #supress error message here -  we only want to know if something bad happened
        pass
    finally:
        if args.profile:
            mProfile.report(display=False, profiledir=args.profile_dir)
