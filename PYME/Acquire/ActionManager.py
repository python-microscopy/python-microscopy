# -*- coding: utf-8 -*-
"""
Created on Sat May 28 23:12:24 2016

@author: david
"""
try:
    #python 2.x
    # noinspection PyCompatibility
    import Queue
except ImportError:
    #python 3.x
    import queue as Queue
    
import time
from PYME.contrib import dispatch
import weakref
import threading
from PYME.util import webframework
import numpy as np
import logging
logger = logging.getLogger(__name__)

from .actions import *

class ActionManager(object):
    """This implements a queue for actions which should be called sequentially.
    
    The main purpose of the ActionManager is to facilitate automated imaging by 
    allowing multiple operations to be queued. Rather than being strictly FIFO,
    different tasks can be asigned different priorities, with higher priority 
    tasks bubbling up and and being executed before lower priority tasks. This 
    allows high priority "imaging" tasks to be inserted into a stream of lower
    priority "monitoring" tasks if something interesting is detected during 
    monitoring.
    
    An individual action is a function which can be found within the scope of
    our microscope object (for more details see the QueueAction method).
    
    To function correctly the Tick() method should be called regularly - e.g.
    from a GUI timer.
    """
    def __init__(self, scope):
        """Initialise our action manager
        
        Parameters
        ----------
        
        scope : PYME.Acquire.microscope.Microscope object
            The microscope. The function object to call for an action should be 
            accessible within the scope namespace, and will be resolved by
            calling eval('scope.functionName')
        
        """
        self.actionQueue = Queue.PriorityQueue() # queue for immediate execution
        self.scheduledQueue = Queue.PriorityQueue() # queue for scheduled execution
        self.scope = weakref.ref(scope)
        
        #this will be assigned to a callback to indicate if the last task has completed        
        self.isLastTaskDone = None
        self.paused = False
        
        self.currentTask = None
        
        self.onQueueChange = dispatch.Signal()
        
        self._timestamp = 0

        self._monitoring = True
        self._monitor = threading.Thread(target=self._monitor_defunct)
        self._monitor.daemon = True
        self._monitor.start()
        
        self._lock = threading.Lock()
        
    def QueueAction(self, functionName, args, nice=10, timeout=1e6, 
                    max_duration=np.finfo(float).max, execute_after=0):
        """Add an action to the queue. Legacy version for string based actions. Most applications should use queue_actions() below instead
        
        Parameters
        ----------
        
        functionName : string
            The name of a function relative to the microscope object.
            e.g. to `call scope.spoolController.start_spooling()`, you would use
            a functionName of 'spoolController.start_spooling'.
            
            The function should either return `None` if the operation has already
            completed, or function which evaluates to True once the operation
            has completed. See `scope.spoolController.start_spooling()` for an
            example.
            
        args : dict
            a dictionary of arguments to pass the function    
        nice : int (or float)
            The priority with which to execute the function. Functions with a
            lower nice value execute first. Nice should have a value between 0 and 20. nice=20 is reserved by convention
            for tidy-up tasks which should run
            after all other tasks and put the microscope in a 'safe' state.
        timeout : float
            A timeout in seconds from the current time at which the action
            becomes irrelevant and should be ignored.
        max_duration : float
            A generous estimate, in seconds, of how long the task might take, 
            after which the lasers will be automatically turned off and the 
            action queue paused. This will not interrupt the current task, 
            though it has presumably already failed at that point. Intended as a
            safety feature for automated acquisitions, the check is every 3 s 
            rather than fine-grained.
        execute_after: float
            A timestamp in system time before which the action should not be executed.
            If this is before the current time, the action will be queued for immediate
            execution, otherwise it will be placed in a queue of scheduled acquisitions
            which is polled by the Tick() method and added to the execution queue when
            the time is right. The default of 0 means that the action will be queued for
            immediate execution.
            
        """
        # make sure nice is in supported range.
        assert ((nice >= 0) and (nice <= 20))
        
        curTime = time.time()
        expiry = curTime + timeout
        
        #make sure our timestamps strictly increment
        self._timestamp = max(curTime, self._timestamp + 1e-3)
        
        #ensure FIFO behaviour for events with the same priority
        nice_ = nice + self._timestamp*1e-10
        
        if execute_after < curTime:
            logger.debug('Queuing action %s for immediate execution (%s, %s)' % (functionName, curTime, execute_after))
            self.actionQueue.put_nowait((nice_, FunctionAction(functionName, args), expiry, max_duration))
        else:
            logger.debug('Queuing action %s for delayed execution (%s, %s)' % (functionName, curTime, execute_after))
            self.scheduledQueue.put_nowait((execute_after, (nice_, FunctionAction(functionName, args), expiry, max_duration)))
        
        self.onQueueChange.send(self)
        
    def queue_actions(self, actions, nice=10, timeout=1e6, max_duration=np.finfo(float).max, execute_after=0):
        '''
        Queue a number of actions for subsequent execution
        
        Parameters
        ----------
        actions : list
            A list of Action instances
        nice : int (or float)
            The priority with which to execute the function. Functions with a
            lower nice value execute first. Nice should have a value between 0 and 20.
        timeout : float
            A timeout in seconds from the current time at which the action
            becomes irrelevant and should be ignored.
        max_duration : float
            A generous estimate, in seconds, of how long the task might take,
            after which the lasers will be automatically turned off and the
            action queue paused. This will not interrupt the current task,
            though it has presumably already failed at that point. Intended as a
            safety feature for automated acquisitions, the check is every 3 s
            rather than fine-grained.
        execute_after: float
            A timestamp in system time before which the action should not be executed.
            If this is before the current time, the action will be queued for immediate
            execution, otherwise it will be placed in a queue of scheduled acquisitions
            which is polled by the Tick() method and added to the execution queue when
            the time is right. The default of 0 means that the action will be queued for
            immediate execution.

        Returns
        -------
        
        
        Examples
        --------
        
        >>> my_actions = [UpdateState(state={'Camera.ROI' : [50, 50, 200, 200]}),
        >>>      SpoolSeries(maxFrames=500, stack=False),
        >>>      UpdateState(state={'Camera.ROI' : [100, 100, 250, 250]}).then(SpoolSeries(maxFrames=500, stack=False)),
        >>>      ]
        >>>
        >>>ActionManager.queue_actions(my_actions)
        
        Note that the first two tasks are independant -

        '''
        # make sure nice is in supported range.
        assert((nice >= 0) and (nice <= 20))
        
        with self._lock:
            # lock to prevent 'nice' collisions when queueing from separate threads.
            
            for j, action in enumerate(actions):
                curTime = time.time()
                expiry = curTime + timeout
            
                #make sure our timestamps strictly increment
                self._timestamp = max(curTime, self._timestamp + 1e-3)
            
                #ensure FIFO behaviour for events with the same priority
                nice_ = nice + self._timestamp * 1e-10

                if np.isscalar(execute_after):
                    after = execute_after
                else:
                    after = execute_after[j]

                if after < curTime:
                    #logger.debug('Queuing action %s for immediate execution (%s, %s)' % (functionName, curTime, execute_after))
                    self.actionQueue.put_nowait((nice_, action, expiry, max_duration))
                else:
                    #logger.debug('Queuing action %s for delayed execution (%s, %s)' % (functionName, curTime, execute_after))
                    self.scheduledQueue.put_nowait((after, (nice_, action, expiry, max_duration)))
            
        self.onQueueChange.send(self)
        
        
    def Tick(self, **kwargs):
        """Polling function to check if the current action is finished and, if so, start the next
        action if available.
        
        Should be called regularly for a timer or event loop.
        """
        if self.paused:
            return
            
        # queue scheduled actions which are now due
        _n_queued = 0
        while (self.scheduledQueue.qsize() > 0) and ((self.scheduledQueue.queue[0][0]) < time.time()):
            execute_after, action = self.scheduledQueue.get_nowait()
            print(time.time(), execute_after, action)
            self.actionQueue.put_nowait(action)
            _n_queued += 1

        if _n_queued > 0:
            self.onQueueChange.send(self) # TODO - avoid sending this signal here and below??
        
        if (self.isLastTaskDone is None) or self.isLastTaskDone():
            try:
                self.currentTask.finalise(self.scope())
                self.currentTask = None
            except AttributeError:
                pass

            try:
                self.currentTask = self.actionQueue.get_nowait()
                nice, action, expiry, max_duration = self.currentTask
                self._cur_task_kill_time = time.time() + max_duration
                self.onQueueChange.send(self)
            except Queue.Empty:
                self.currentTask = None
                return
            
            if expiry > time.time():
                logger.debug('Executing action: %s, %s' % (self.currentTask, action))
                #fcn = eval('.'.join(['self.scope()', functionName]))
                self.isLastTaskDone = action(self.scope())
            else:
                past_expire = time.time() - expiry
                logger.debug('Action expired %f s ago, ignoring %s' % (past_expire,
                                                                     self.currentTask))
    
    def _monitor_defunct(self):
        """
        polling thread method to check that if a task is being executed through
        the action manager it isn't taking longer than its `max_duration`.
        """
        while self._monitoring:
            if self.currentTask is not None:
                #logger.debug('here, %f s until kill' % (self._cur_task_kill_time - time.time()))
                if time.time() > self._cur_task_kill_time:
                    self.scope().turnAllLasersOff()
                    # pause and reset so we can start up again later
                    self.paused = True
                    self.isLastTaskDone = None
                    self.currentTask = None
                    self.onQueueChange.send(self)
                    logger.error('task exceeded specified max duration')
        
            time.sleep(3)
    
    def __del__(self):
        self._monitoring = False


class ActionManagerWebWrapper(object):
    def __init__(self, action_manager):
        """ Wraps an action manager instance with server endpoints

        Parameters
        ----------
        action_manager : ActionManager
            action manager instance to wrap
        """
        self.action_manager = action_manager

    @webframework.register_endpoint('/queue_actions', output_is_json=False)
    def queue_actions(self, body, nice=10, timeout=1e6, max_duration=np.finfo(float).max, execute_after=0):
        """
        Add a list of actions to the queue
        
        Parameters
        ----------
        body - json formatted list of serialised actions (see example below)
        nice
        timeout
        max_duration

        Returns
        -------
        
        
        Examples
        --------
        
        Body json
        ^^^^^^^^^
        
        .. code-block:: json

            [{'UpdateState':{'foo':'bar', 'then': {'SpoolSeries' : {...}}}]

        """
        import json
        actions = [action_from_dict(a) for a in json.loads(body)]

        self.action_manager.queue_actions(actions, nice=int(nice), 
                                          timeout=float(timeout), 
                                          max_duration=float(max_duration),
                                          execute_after=execute_after)
        
    
    @webframework.register_endpoint('/queue_action', output_is_json=False)
    def queue_action(self, body):
        """
        adds an action to the queue

        Parameters
        ----------
        body: str
            json.dumps(dict) with the following keys:
                function_name : str
                    The name of a function relative to the microscope object.
                    e.g. to `call scope.spoolController.start_spooling()`, you 
                    would use a functionName of 'spoolController.start_spooling'.
                    
                    The function should either return `None` if the operation 
                    has already completed, or function which evaluates to True 
                    once the operation has completed. See 
                    `scope.spoolController.start_spooling()` for an example.
                args : dict, optional
                    a dictionary of arguments to pass to `function_name`
                nice : int, optional
                    priority with which to execute the function, by default 10. 
                    Functions with a lower nice value execute first.
                timeout : float, optional
                    A timeout in seconds from the current time at which the 
                    action becomes irrelevant and should be ignored. By default
                    1e6.
                max_duration : float
                    A generous estimate, in seconds, of how long the task might
                    take, after which the lasers will be automatically turned 
                    off and the action queue paused.
        """
        import json
        params = json.loads(body)
        function_name = params['function_name']
        args = params.get('args', {})
        nice = params.get('nice', 10.)
        timeout = params.get('timeout', 1e6)
        max_duration = params.get('max_duration', np.finfo(float).max)
        execute_after = params.get('execute_after', 0)

        self.action_manager.QueueAction(function_name, args, nice, timeout,
                                        max_duration, execute_after)


class ActionManagerServer(webframework.APIHTTPServer, ActionManagerWebWrapper):
    def __init__(self, action_manager, port, bind_address=''):
        """
        Server process to expose queue_action functionality to everything on the
        cluster network.

        NOTE - this will likely not be around long, as it would be preferable to
        add the ActionManagerWebWrapper to
        `PYME.acquire_server.AcquireHTTPServer` and run a single server process
        on the microscope computer.

        Parameters
        ----------
        action_manager : ActionManager
            already initialized
        port : int
            port to listen on
        bind_address : str, optional
            specifies ip address to listen on, by default '' will bind to local 
            host.
        """
        webframework.APIHTTPServer.__init__(self, (bind_address, port))
        ActionManagerWebWrapper.__init__(self, action_manager)
        
        self.daemon_threads = True
        self._server_thread = threading.Thread(target=self._serve)
        self._server_thread.daemon_threads = True
        self._server_thread.start()

    def _serve(self):
        try:
            logger.info('Starting ActionManager server on %s:%s' % (self.server_address[0], self.server_address[1]))
            self.serve_forever()
        finally:
            logger.info('Shutting down ActionManager server ...')
            self.shutdown()
            self.server_close()

