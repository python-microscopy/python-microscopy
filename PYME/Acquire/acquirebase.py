''' Base class for both wx and HTML based acquisition interfaces'''
import threading
import time

from PYME.Acquire import microscope
from PYME.Acquire import protocol
from PYME.Acquire import event_loop

import logging
logger = logging.getLogger(__name__)

class PYMEAcquireBase(object):
    def __init__(self, options=None, evt_loop=None) -> None:
        if evt_loop is None:
            self.evt_loop = event_loop.EventLoop()
        else:
            self.evt_loop = evt_loop

        self.options = options

        self.snapNum = 0
        
        self.MainFrame = self #reference to this window for use in scripts etc...
        protocol.MainFrame = self

        self.initDone = False
        self.postInit = [] #for protocol compat
        
        self.scope = microscope.Microscope()

        self.roi_on = False
        self.bin_on = False

        # non-GUI timer (replaces time1 for non-GUI sceduled events)
        self._timer0 = self.evt_loop.MultiTargetTimer()
        self._timer0.start(50)
        
        self._is_running = False
        
        # variables to facilitate long-polling for frame updates
        self._current_frame = None
        self._new_frame_condition = threading.Condition()
        
        self._state_valid = False
        self._state_updated_condition = threading.Condition()
        
        self.scope.state.stateChanged.connect(self._on_scope_state_change)
    
    def main_loop(self):
        """
        Infinite loop which polls hardware
        
        Returns
        -------

        """
        #self._is_running = True

        logger.debug('Starting initialisation')
        self._initialize_hardware()
        #poll to see if the init script has run
        self._wait_for_init_complete()
        
        
        logger.debug('Starting post-init')

        self.non_gui_post_init()

        logger.debug('Finished post-init')
        
        try:
            logger.debug('Starting event loop')
            self.evt_loop.loop_forever()
        except:
            logger.exception('Exception in event loop')
        finally:
            logger.debug('Shutting down')
            self._shutdown()

    def non_gui_post_init(self):
        if self.scope.cam.CamReady():# and ('chaninfo' in self.scope.__dict__)):
            self._start_polling_camera()

        self._timer0.register_callback(self.scope.actions.Tick)
        self.initDone = True


    def _initialize_hardware(self):
        """
        Launch microscope hardware initialization and start polling for completion

        """
        #this spawns a new thread to run the initialization script
        self.scope.initialize(self.options.initFile, self.__dict__)

        logger.debug('Init run, waiting on background threads')

    def _wait_for_init_complete(self):
        self.scope.wait_for_init()
        logger.debug('Backround initialization done')
        
    def _on_frame_group(self, *args, **kwargs):
        #logger.debug('_on_frame_group')
        with self._new_frame_condition:
            self._current_frame = self.scope.frameWrangler.currentFrame
            #logger.debug(repr(self.scope.frameWrangler.currentFrame))
            self._new_frame_condition.notify()
            
    def _on_scope_state_change(self, *args, **kwargs):
        with self._state_updated_condition:
            self._state_valid = False
            self._state_updated_condition.notify()
    
    def _start_polling_camera(self):
        self.scope.startFrameWrangler(event_loop=self.evt_loop)
        self.scope.frameWrangler.onFrameGroup.connect(self._on_frame_group)

    def _shutdown(self):
        self.scope.frameWrangler.stop()
        
        if 'cameras' in dir(self.scope):
            for c in self.scope.cameras.values():
                c.Shutdown()
        else:
            self.scope.cam.Shutdown()
            
        for f in self.scope.CleanupFunctions:
            f()
            
        logger.info('All cleanup functions called')
        
        time.sleep(1)
        
        import threading
        msg = 'Remaining Threads:\n'
        for t in threading.enumerate():
            msg += '%s, %s\n' % (t.name, t._target)
            
        logger.info(msg)