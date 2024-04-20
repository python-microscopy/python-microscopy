# protocol_acquisition.py 

# replaces Spooler.py and the various backend-specific spoolers, factoring out everything which is specific to 
# asynchronous protocol based acquisition.
#
# Protocol based asynchronous acquisition should be used where it is important that the camera runs as fast as possible.
# To allow this, we do not wait for various hardware events to complete, but rather just record timestamps ("Events") and work things
# out in post-processing.


import datetime
import time
import os
import uuid
import sys

from PYME.contrib import dispatch

from PYME import config
from PYME.IO import MetaDataHandler
from PYME.IO import acquisition_backends
from PYME.IO.events import HDFEventLogger, MemoryEventLogger

from PYME.Acquire import eventLog
from PYME.Acquire import protocol as p
from PYME.Acquire.acquisition_base import AcquisitionBase


try:
    from PYME.Acquire import sampleInformation
except:
    sampleInformation= None


import logging
logger = logging.getLogger(__name__)



def getReducedFilename(filename):
    #rname = filename[len(nameUtils.datadir):]
        
    sname = '/'.join(filename.split(os.path.sep))
    if sname.startswith('/'):
        sname = sname[1:]
    
    return sname


class ProtocolAcquisition(AcquisitionBase):
    """Spooler base class"""
    def __init__(self, filename, frameSource, protocol = p.NullProtocol, 
                 fakeCamCycleTime=None, maxFrames = p.maxint, backend='hdf', backend_kwargs={}, **kwargs):
        """Create a new spooler.
        
        Parameters
        ----------
        scope : PYME.Acquire.microscope.Microscope object
            The microscope providing the data
        filename : string
            The file into which to spool
        frameSource : dispatch.Signal object
            A source of frames we can subscribe to. It should implement a "connect"
            method allowing us to register a callback and then call the callback with
            the frame data in a "frameData" kwarg.
        protocol : PYME.Acquire.protocol.TaskListProtocol object
            The acquisition protocol
        guiUpdateCallback : function
            a function to call when the spooling GUI needs updating
            
        """
        AcquisitionBase.__init__(self)

        self.filename=filename
        self.frameSource = frameSource
        self.seriesName = getReducedFilename(filename)
        
        self.protocol = protocol
        
        self.maxFrames = maxFrames
        
        stack_settings = kwargs.get('stack_settings', None)
        if stack_settings:
            # only record stack settings if provided (letting protocol fall through to global stack settings,
            # if not provided / None)
            self.stack_settings = stack_settings

        
        self.spoolOn = False
        self.frame_num = 0
        
        self.spool_complete = False
        
        self._spooler_uuid = uuid.uuid4()
            
        self._fakeCamCycleTime = fakeCamCycleTime

        self._last_gui_update = 0

        self._create_backend(backend_type=backend, **backend_kwargs)

    @classmethod
    def from_spool_settings(cls, scope, settings, backend, backend_kwargs={}, series_name=None, spool_controller=None):
        '''Create an XYZTCAcquisition object from a spool_controller settings object'''
        from PYME.IO import acquisition_backends

        if isinstance(backend, acquisition_backends.MemoryBackend):
            # TODO - make this softer and allow memory backend for fixed length protocol acquisitions???
            raise RuntimeError('Memory spooling not supported for protocol-based acquisitions')
        
        protocol = spool_controller.protocol_settings.get_protocol_for_acquistion(settings=settings)
        
        preflight_mode = settings.get('preflight_mode', 'interactive')
        if (preflight_mode != 'skip'):
            from PYME.Acquire.ui import preflight
            if not preflight.ShowPreflightResults(protocol.PreflightCheck(), preflight_mode):
                logger.debug('Bailing from preflight check')
                return None #bail if we failed the pre flight check, and the user didn't choose to continue
        
        #fix timing when using fake camera
        if scope.cam.__class__.__name__ == 'FakeCamera':
            fakeCycleTime = scope.cam.GetIntegTime()
        else:
            fakeCycleTime = None
        
        #logger.info('Creating spooler for %s' % series_name)
        return cls(filename=series_name,
                    frameSource=scope.frameWrangler.onFrame,
                    protocol=protocol,
                    fakeCamCycleTime=fakeCycleTime, 
                    maxFrames=settings.get('max_frames', sys.maxsize),
                    stack_settings=settings.get('stack_settings', None),
                    backend=backend, backend_kwargs=backend_kwargs,) 

    @classmethod
    def get_frozen_settings(cls, scope, spool_controller=None):
        settings = {'z_stepped' : spool_controller.protocol_settings.z_stepped,
                'z_dwell' : spool_controller.protocol_settings.z_dwell,} 
        
        if not spool_controller.protocol_settings.protocol in [p.NullProtocol, p.NullZProtocol]:
            settings['protocol_name'] = spool_controller.protocol_settings.protocol.filename

        return settings           


    def _create_backend(self, backend_type=acquisition_backends.HDFBackend, **kwargs):
        logger.debug('Creating backend of type %s' % backend_type)
        
        if backend_type in  ['cluster', 'Cluster', acquisition_backends.ClusterBackend]:
            self._aggregate_h5 = kwargs.get('cluster_h5', False)
            
            self.clusterFilter = kwargs.get('serverfilter', config.get('dataserver-filter', ''))
            
            chunk_size = config.get('httpspooler-chunksize', 50)
            
            def dist_fcn(n_servers, i=None):
                if i is None:
                    # distribute at random
                    import random
                    return random.randrange(n_servers)
                
                return int(i/chunk_size) % n_servers
            
            
            self._backend = acquisition_backends.ClusterBackend(self.seriesName, 
                                                                distribution_fcn=dist_fcn, 
                                                                compression_settings=kwargs.get('compression_settings', {}),
                                                                cluster_h5=self._aggregate_h5,
                                                                serverfilter=self.clusterFilter,
                                                                shape=[-1,-1,1,-1,1], #spooled aquisitions are time series (for now)
                                                                evt_time_fcn=self._time_fcn)
            
        else: # assume hdf
            self._backend = acquisition_backends.HDFBackend(self.filename, complevel=kwargs.get('complevel', 6), complib=kwargs.get('complib','zlib'),
                            shape=[-1,-1,1,-1,1], # spooled series are time-series (for now)
                            evt_time_fcn=self._time_fcn)
        
        
        self._stopping = False

    @property
    def md(self):
        return self._backend.mdh

    def StartSpool(self):
        from PYME import warnings
        warnings.warn('StartSpool is deprecated. Use start instead', DeprecationWarning)
        self.start()

    def StopSpool(self):
        from PYME import warnings
        warnings.warn('StopSpool is deprecated. Use stop instead', DeprecationWarning)
        self.stop()
    
    def start(self):
        """ Perform protocol 'frame -1' tasks, log start metadata, then connect
        to the frame source.
        """
        self.watchingFrames = True
        eventLog.register_event_handler(self._backend.event_logger)

        self.frame_num = 0
        
        # set tStart here for simulator so that events in init phase get time stamps. Real start time is set below
        # **after** protocol.Init() call
        self.tStart = time.time()

        self.protocol.Init(self)
        
        # record start time when we start receiving frames.
        self.tStart = time.time()
        self._collect_start_metadata()
        self.frameSource.connect(self.on_frame, dispatch_uid=self._spooler_uuid)
        
        self.spoolOn = True

        logger.debug('Starting spooling: %s' %self.seriesName)

        self._backend.initialise()
       
    def stop(self):
        #try:
        logger.debug('Disconnecting from frame source')
        self.frameSource.disconnect(self.on_frame, dispatch_uid=self._spooler_uuid)
        logger.debug('Frame source should be disconnected')
        
        #there is a race condition on disconnect - ignore any additional frames
        self.watchingFrames = False 
        
        #except:
        #    pass

        try:
            self.protocol.OnFinish()#this may still cause events
            self.FlushBuffer()
            self._collect_stop_metadata()
        except:
            import traceback
            traceback.print_exc()
            
        try:
            eventLog.remove_event_handler(self._backend.event_logger)
        except ValueError:
            pass
        
        self.spoolOn = False

        self.on_progress.send(self)
        
        self._stopping=True
        self.finalise()
        self.on_stop.send(self)
        self.spool_complete = True
        
    def finalise(self):
        """
        Over-ride in derived classes to do any spooler specific tidy up - e.g. sending events to server

        """
        self._backend.finalise()

        
    def abort(self):
        """
        Tidy up if something goes horribly wrong. Disconnects frame source and event logger  and then calls cleanup()

        """
        #there is a race condition on disconnect - ignore any additional frames
        self.watchingFrames = False
        
        try:
            logger.debug('Disconnecting from frame source')
            self.frameSource.disconnect(self.OnFrame, dispatch_uid=self._spooler_uuid)
            logger.debug('Frame source should be disconnected')
        except:
            logger.exception('Error disconnecting frame source')


        try:
            eventLog.remove_event_handler(self._backend.event_logger)
        except ValueError:
            pass

        self.spoolOn = False
        self.on_stop.send(self)
        

    def on_frame(self, sender, frameData, **kwargs):
        """Callback which should be called on every frame"""
        if not self.watchingFrames:
            #we have allready disconnected - ignore any new frames
            return
        
        self._backend.store_frame(self.frame_num, frameData)

        t = time.time()
            
        self.frame_num += 1
        
        if (t > (self._last_gui_update +.1)):
            self._last_gui_update = t
            self.on_progress.send(self)
            
        try:
            import wx #FIXME - shouldn't do this here
            wx.CallAfter(self.protocol.OnFrame, self.frame_num)
            #FIXME - The GUI logic shouldn't be here (really needs to change at the level of the protocol and/or general structure of PYMEAcquire
        except (ImportError, AssertionError):  # handle if spooler doesn't have a GUI
            self.protocol.OnFrame(self.frame_num) #FIXME - This will most likely fail for anything but a NullProtocol

        if self.frame_num == 2 and sampleInformation and sampleInformation.currentSlide[0]: #have first frame and should thus have an imageID
            sampleInformation.createImage(self.md, sampleInformation.currentSlide[0])
            
        if self.frame_num >= self.maxFrames:
            self.stop()
            

    def _collect_start_metadata(self):
        """Record pertinant information to metadata at start of acquisition.
        
        Loops through all registered sources of start metadata and adds their entries.
        
        See Also
        --------
        PYME.IO.MetaDataHandler
        """
        dt = datetime.datetime.now()
        
        self.dtStart = dt
        
        #self.tStart = time.time()
        
        # create an in-memory metadata handler and populate this prior to copying data over to the spooler
        # metadata handler. This significantly improves performance if the spooler metadata handler has high latency
        # (as is the case for both the HDFMetaDataHandler and, especially, the QueueMetaDataHandler).
        mdt = MetaDataHandler.NestedClassMDHandler()
        mdt.setEntry('StartTime', self.tStart)

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(mdt)
            
        self.md.copyEntriesFrom(mdt)
       

    def _collect_stop_metadata(self):
        """Record information to metadata at end of acquisition"""
        self.md.setEntry('EndTime', time.time())
        
        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStopMetadata:
           mdgen(self.md)

    def _fake_time(self):
        """Generate a fake timestamp for use with the simulator where the camera
        cycle time does not match the actual time elapsed to generate the frame"""
        #return self.tStart + self.frame_num*self.scope.cam.GetIntegTime()
        return self.tStart + self.frame_num*self._fakeCamCycleTime
    
    @property
    def _time_fcn(self):
        if self._fakeCamCycleTime:
            return self._fake_time
        else:
            return time.time

    def FlushBuffer(self):
        pass
    
    def status(self):
        return {'spooling' : self.spoolOn,
                'frames_spooled' : self.frame_num,
                'spool_complete' : self.spool_complete,}
    
    def cleanup(self):
        """ over-ride to do any cleanup"""
        del self._backend
    
    def finished(self):
        """ over-ride in derived classes to indicate when buffers flushed"""
        
        # FIXME - this probably needs a bit more work.
        # FIXME - delegate to backend?
        return self._stopping
    
    def get_n_frames(self):
        return self.frame_num
        
    def __del__(self):
        if self.spoolOn:
            self.StopSpool()




