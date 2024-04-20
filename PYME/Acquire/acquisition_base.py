"""
Base class for acquisition modules 

This exists to document the interface which acquisition types should implement in order 
to be able to be used with spooling etc ...
"""

import abc
from PYME.contrib import dispatch


class AcquisitionBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        '''Create an Acquisition object'''
        
        # A signal to be emitted when progress is made in the acquisition
        # this triggers a GUI update, and implementations can choose whether to emit this signal
        # on every frame, or to throttle it. 
        self.on_progress = dispatch.Signal()

        # A signal to be emitted when the acquisition is stopped
        # this should always be emitted when the acquisition is stopped, regardless of the reason
        self.on_stop = dispatch.Signal()

        # A flag to indicate whether the spool is complete. Ths should be set to True once spooling is finished 
        # (it is watched by the ActionManager to determine if the acquisition is complete and the next action should be executed)
        self.spool_complete = False

    @classmethod
    @abc.abstractmethod
    def from_spool_settings(cls, scope, settings, backend, backend_kwargs={}, series_name=None, spool_controller=None):
        '''Create an Acquisition object from settings and a backend.
        
        
        Parameters
        ----------

        scope : microscope.Microscope
            The microscope object to use for acquisition. Supplied in order to allow hardware control and access to the frameWrangler.

        settings : dict
            A dictionary of settings for the acquisition

        backed : class
            The class of the backend to use for the acquisition. Generally one of PYME.IO.acquisition_backends

        backend_kwargs : dict
            A dictionary of keyword arguments to pass to the backend constructor

        series_name : str
            The name of the series to create. series_name should be defined unless the MemoryBackend is used, in which case it is ignored.

        spool_controller : object
            The spool controller object to use for the acquisition. Generally scope.spoolController. Currently used by ProtocolAcquisition 
            to access default protocol settings held by the spoolController, but a bit of a legacy hack. May be removed in the future.
        
        '''
        pass

    @classmethod
    def get_frozen_settings(cls, scope, spool_controller=None):
        '''Return a dictionary of settings for the acquisition
        
        Used to "freeze" the state of the spool_controller and/or other settings objects when queueing
        acquisitions for subsequent execution via the ActionManager
        '''

        return {}
    
    @abc.abstractmethod
    def start(self):
        '''Start the acquisition
        
        This will usually record any metadata, connect self.on_frame to a frame source (i.e. scope.frameWrangler.onFrame), and initialise
        the backend. It should not block, with the bulk of the acquisition logic taking place in frame source handler.
        
        '''
        pass

    @abc.abstractmethod
    def stop(self):
        '''Stop the acquisition
        
        This should disconnect from the frame souce, flush any remaining buffers, return hardware to starting state (if modified),
        do any cleanup, and finalise the backend.

        It is desirable to send a final on_progress signal so that the GUI reflects the actual number of frames spooled etc... .

        Once everything is complete, the on_stop signal should be emitted and the spool_complete flag set.
        
        '''
        pass

    
    def abort(self):
        '''Abort the acquisition
        
        This should stop the acquisition as quickly as practical. In many cases this can just be a call to stop(),
        but there is no expectation that buffers will be flushed, remaining protocol steps executed or hardware returned 
        to a starting state. Especially for cases which involve moving translation stages or other hardware it is desirable
        that the abort method not result in further movement of hardware unless to a known safe state (e.g. lasers off).
        
        Abort should, however, finalise the backend and emit the on_stop signal. 
        TODO - should abort set spool_complete? 
        '''
        self.stop()

    @abc.abstractmethod
    def md(self):
        ''' Return acqusition metadata (a PYME.IO.MetaDataHandler object)
        
        generally just a short wrapper around the backend metadata

        #TODO - rename this to .mdh, or remove entirely and standardise backend access?
        '''
        pass
        
    @abc.abstractmethod
    def on_frame(self, sender, frameData, **kwargs):
        '''Frame source handler
        
        This method should be connected to the frame source signal (i.e. scope.frameWrangler.onFrame) and be responsible for 
        handling incoming frames, passing them off to storage and performing anything that needs to be done (e.g. hardware movements
        or protocol task handling) before the next frame. NOTE - this method is called when the frame is retrieved by the frame wrangler
        - if the camera is running in continuous mode, this may be several frames after the camera has actually acquired the frame. 
        
        '''
        pass
