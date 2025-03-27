"""
Implements an event-loop and timers for headless usage of PYMEAcquire

"""
import threading
try:
    import queue
except ImportError:
    import Queue as queue
import time
import sys
import weakref

import logging
logger = logging.getLogger(__name__)

class _Timer(object):
    def __init__(self):
        self._next_trigger = sys.float_info.max
        self._single_shot = False
        self._delay_s = -1
        
        logger.debug('Created timer')
    
    def notify(self):
        raise NotImplementedError('Over-ride in derived class')
    
    def check(self, t):
        if t > self._next_trigger:
            #logger.debug('Timer triggered')
            self._next_trigger = sys.float_info.max
            try:
                self.notify()
            except:
                logger.exception('Error in timer event')
            finally:
                if not self._single_shot:
                    self._next_trigger = t + self._delay_s
                return True
        else:
            return False
    
    def start(self, delay_ms, single_shot=False):
        self._single_shot = single_shot
        self._delay_s = 0.001*delay_ms
        self._next_trigger = time.time() + self._delay_s
    
    def stop(self):
        self._next_trigger = sys.float_info.max
        
class _SingleTargetTimer(_Timer):
    def __init__(self, target):
        _Timer.__init__(self)
        self._target = target
        
    def notify(self):
        self._target()


class _MultiTargetTimer(_Timer):
    """
    Timer which calls multiple handlers
    """
    
    def __init__(self, PROFILE=False):
        _Timer.__init__(self)
        self.WantNotification = []
    
    def notify(self):
        for a in self.WantNotification:
            a()
    
    def register_callback(self, callback):
        self.WantNotification.append(callback)

class EventLoop(object):
    def __init__(self):
        self._loop_active = False
        
        #keep a list of stuff to execute
        self._deferreds = queue.Queue()
        self._timers = weakref.WeakSet()
        
    def loop_forever(self):
        self._loop_active = True
        while self._loop_active:
            try:
                # process stuff which needs to run in this thread
                callable, args, kwargs = self._deferreds.get(timeout=0.01)
                logger.debug('Calling %s' % repr((callable, args, kwargs)))
                callable(*args, **kwargs)
            except(queue.Empty):
                #do timer stuff
                t = time.time()
                did_stuff = False
                for tm in self._timers:
                    did_stuff = did_stuff or tm.check(t)

                if not did_stuff:
                    # sleep a bit (to limit CPU usage)
                    time.sleep(0.001)
            
    def stop(self):
        self._loop_active = False

    def call_in_main_thread(self, callable, *args, **kwargs):
        self._deferreds.put((callable, args, kwargs))
        
    #factory functions for timers - create a timer and add it to _timers
    def SingleTargetTimer(self, target):
        tm = _SingleTargetTimer(target)
        self._timers.add(tm)
        return tm
    
    def MultiTargetTimer(self):
        tm = _MultiTargetTimer()
        self._timers.add(tm)
        return tm
        