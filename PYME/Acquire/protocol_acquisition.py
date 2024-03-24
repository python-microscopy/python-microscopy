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

from PYME.contrib import dispatch

from PYME import config
from PYME.IO import MetaDataHandler
from PYME.IO import acquisition_backends
from PYME.IO.events import HDFEventLogger, MemoryEventLogger

from PYME.Acquire import eventLog
from PYME.Acquire import protocol as p


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


class ProtocolAcquision(object):
    """Spooler base class"""
    def __init__(self, filename, frameSource, protocol = p.NullProtocol, 
                 guiUpdateCallback=None, fakeCamCycleTime=None, maxFrames = p.maxint, backend='hdf', **kwargs):
        """Create a new spooler.
        
        Parameters
        ----------
        scope : PYME.Acquire.microscope.microscope object
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
        #self.scope = scope
        self.filename=filename
        self.frameSource = frameSource
        self.seriesName = getReducedFilename(filename)
        self.guiUpdateCallback = guiUpdateCallback
        self.protocol = protocol
        
        self.maxFrames = maxFrames
        
        stack_settings = kwargs.get('stack_settings', None)
        if stack_settings:
            # only record stack settings if provided (letting protocol fall through to global stack settings,
            # if not provided / None)
            self.stack_settings = stack_settings
        
        self.onSpoolStop = dispatch.Signal()

        self._last_gui_update = 0
        self.spoolOn = False
        self.imNum = 0
        
        self.spool_complete = False
        
        self._spooler_uuid = uuid.uuid4()
            
        self._fakeCamCycleTime = fakeCamCycleTime

        self._create_backend(backend_type=backend, **kwargs)


    def _create_backend(self, backend_type='hdf', **kwargs):
        logger.debug('Creating backend of type %s' % backend_type)
        if backend_type in  ['cluster', 'Cluster', acquisition_backends.ClusterBackend]:
            self._aggregate_h5 = kwargs.get('aggregate_h5', False)
            
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
                                                                compression_settings=kwargs.get('compressionSettings', {}),
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

        self.imNum = 0
        
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
        if not self.guiUpdateCallback is None:
            self.guiUpdateCallback()
        
        self.finalise()
        self.onSpoolStop.send(self)
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
        self.onSpoolStop.send(self)
        

    def on_frame(self, sender, frameData, **kwargs):
        """Callback which should be called on every frame"""
        if not self.watchingFrames:
            #we have allready disconnected - ignore any new frames
            return
        
        self._backend.store_frame(self.imNum, frameData)

        t = time.time()
            
        self.imNum += 1
        if not self.guiUpdateCallback is None:
            if (t > (self._last_gui_update +.1)):
                self._last_gui_update = t
                self.guiUpdateCallback()
            
        try:
            import wx #FIXME - shouldn't do this here
            wx.CallAfter(self.protocol.OnFrame, self.imNum)
            #FIXME - The GUI logic shouldn't be here (really needs to change at the level of the protocol and/or general structure of PYMEAcquire
        except (ImportError, AssertionError):  # handle if spooler doesn't have a GUI
            self.protocol.OnFrame(self.imNum) #FIXME - This will most likely fail for anything but a NullProtocol

        if self.imNum == 2 and sampleInformation and sampleInformation.currentSlide[0]: #have first frame and should thus have an imageID
            sampleInformation.createImage(self.md, sampleInformation.currentSlide[0])
            
        if self.imNum >= self.maxFrames:
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
        #return self.tStart + self.imNum*self.scope.cam.GetIntegTime()
        return self.tStart + self.imNum*self._fakeCamCycleTime
    
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
                'frames_spooled' : self.imNum}
    
    def cleanup(self):
        """ over-ride to do any cleanup"""
        del self._backend
    
    def finished(self):
        """ over-ride in derived classes to indicate when buffers flushed"""
        
        # FIXME - this probably needs a bit more work.
        # FIXME - delegate to backend?
        return self._stopping
    
    def get_n_frames(self):
        return self.imNum
        
    def __del__(self):
        if self.spoolOn:
            self.StopSpool()




