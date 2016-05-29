# -*- coding: utf-8 -*-
"""
Created on Sat May 28 23:12:24 2016

@author: david
"""
import Queue
import time
import dispatch
import weakref

class ActionManager(object):
    '''This implements a queue for actions which should be called sequentially.
    
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
    '''
    def __init__(self, scope):
        '''Initialise our action manager
        
        Parameters
        ----------
        
        scope : PYME.Acquire.microscope.microscope object
            The microscope. The function object to call for an action should be 
            accessible within the scope namespace, and will be resolved by
            calling eval('scope.functionName')
        
        '''
        self.actionQueue = Queue.PriorityQueue()
        self.scope = weakref.ref(scope)
        
        #this will be assigned to a callback to indicate if the last task has completed        
        self.isLastTaskDone = None
        self.paused = False
        
        self.currentTask = None
        
        self.onQueueChange = dispatch.Signal()
        
    def QueueAction(self, functionName, args, nice=10, timeout=1e6):
        '''Add an action to the queue
        
        Parameters
        ----------
        
        functionName : string
            The name of a function relative to the microscope object.
            e.g. to `call scope.spoolController.StartSpooling()`, you would use
            a functionName of 'spoolController.StartSpooling'.
            
            The function should either return `None` if the operation has already
            completed, or function which evaluates to True once the operation
            has completed. See `scope.spoolController.StartSpooling()` for an
            example.
            
        args : dict
            a dictionary of arguments to pass the function    
        nice : int (or float)
            The priority with which to execute the function. Functions with a
            lower nice value execute first.
        timeout : float
            A timeout in seconds from the current time at which the action
            becomes irrelevant and should be ignored.
            
        '''
            
        expiry = time.time() + timeout
        self.actionQueue.put_nowait((nice, functionName, args, expiry))
        self.onQueueChange.send(self)
        
        
    def Tick(self, **kwargs):
        '''Polling function to check if the current action is finished and, if so, start the next 
        action if available.
        
        Should be called regularly for a timer or event loop.
        '''
        if self.paused:
            return
            
        if (self.isLastTaskDone is None) or self.isLastTaskDone():
            try:
                self.currentTask = self.actionQueue.get_nowait()
                nice, functionName, args, expiry = self.currentTask
                self.onQueueChange.send(self)
            except Queue.Empty:
                self.currentTask = None
                return
            
            if expiry > time.time():
                fcn = eval('.'.join(['self.scope()', functionName]))
                self.isLastTaskDone = fcn(**args)
                

        
                