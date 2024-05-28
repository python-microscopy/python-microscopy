import numpy as np
import time
import datetime

from PYME.contrib import dispatch
from PYME.IO import MetaDataHandler
from PYME.IO.acquisition_backends import MemoryBackend
from PYME.Acquire.acquisition_base import AcquisitionBase
from PYME.Acquire import eventLog


class TimeSettings(object):
    '''
    Class to hold settings for time acquisition

    This class pricipally exists to document the interface of the time_settings parameter to XYZTCAcquisition.
    
    Parameters
    ----------
    
    num_timepoints : int
        Number of timepoints to acquire.

    time_interval : float (or None)
        Time interval between timepoints. If None, the acquisition will be continuous. NOTE: the logic for non-none values 
        is not yet implemented, and this parameter will be ignored.
    
    
    '''
    def __init__(self, num_timepoints=1, time_interval=None):
        self.num_timepoints = num_timepoints
        self.time_interval = time_interval

class XYZTCAcquisition(AcquisitionBase):
    def __init__(self, scope, dim_order='XYCZT', stack_settings=None, time_settings=None, channel_settings=None, backend=MemoryBackend, backend_kwargs={}):
        """
        Class to handle an XYZTC acquisition. This should serve as a base class for more specific acquisition classes, whilst also allowing 
        for simple 3D and time-series acquisitions.

        Parameters
        ----------

        scope : PYME.Acquire.microscope.Microscope instance
            The microscope instance to use for acquisition.

        dim_order : str
            A string specifying the order of dimensions in the acquisition. Currently only 'XYCZT' is supported. 

        stack_settings : PYME.Acquire.stackSettings.StackSettings instance
            The settings for the Z-stack acquisition. If None, the settings from scope.stackSettings will be used.

        time_settings : an object with a num_timepoints attribute
            The settings for the time acquisition. If None, only one timepoint will be acquired.

        channel_settings : an object with a num_channels attribute
            The settings for the channel acquisition. If None, only one channel will be acquired.  

        backend : class
            A class implementing the backend interface (see PYME.IO.acquisition_backends) to use for storing the acquired data.
            Used for storing the acquired data and metadata. If None, a MemoryBackend will be used.     

        """
        #if stack_settings is None:
        #    stack_settings = scope.stackSettings
        AcquisitionBase.__init__(self)

        assert(dim_order[:2] == 'XY') #first two dimensions must be XY (camera acquisition)
        # TODO more sanity checks on dim_order
        
        self.dim_order = dim_order
        self.scope = scope
        
        self.shape_x, self.shape_y = scope.frameWrangler.currentFrame.shape[:2]
        
        if stack_settings:
            if isinstance(stack_settings, dict):
                from PYME.Acquire.stackSettings import StackSettings
                stack_settings = StackSettings(scope, **stack_settings)

            self.shape_z = stack_settings.GetSeqLength()
        else:
            self.shape_z = 1

        #keep a reference to the stack settings so we can home the piezo appropriately
        self._stack_settings = stack_settings

        self.shape_t = getattr(time_settings, 'num_timepoints', 1)
        self.shape_c = getattr(channel_settings, 'num_channels', 1)
        
        # note shape_t can be negative if we want to run until explicitly stopped
        self.n_frames = self.shape_z*self.shape_c*self.shape_t
        self.frame_num = 0

        self._running = False
        
        self.storage = backend(size_x = self.shape_x, size_y=self.shape_y, n_frames=self.n_frames, dim_order=dim_order, shape=self.shape, **backend_kwargs)
        
        #do any precomputation
        self._init_z(stack_settings)
        self._init_t(time_settings)
        self._init_c(channel_settings)
    
    
    @classmethod
    def from_spool_settings(cls, scope, settings, backend, backend_kwargs={}, series_name=None, spool_controller=None):
        '''Create an XYZTCAcquisition object from a spool_controller settings object'''

        backend_kwargs['series_name'] = series_name

        return cls(scope=scope, 
                   #dim_order=settings.dim_order, 
                   stack_settings=settings.get('stack_settings', None), 
                   time_settings=settings.get('time_settings', None), 
                   channel_settings=settings.get('channel_settings', None), 
                   backend=backend, backend_kwargs=backend_kwargs)
    
    @property
    def shape(self):
        return self.shape_x, self.shape_y, self.shape_z, self.shape_t, self.shape_c
    
    @property
    def md(self):
        ''' for compatibility with spoolers'''
        return self.storage.mdh
    
        
    def _zct_indices(self, frame_no):
        if self.dim_order == 'XYCZT':
            c = frame_no % self.shape_c
            z = int(frame_no / self.shape_c) % self.shape_z
            t = int(frame_no / (self.shape_c*self.shape_z))
            
            return z, c, t
        else:
            raise NotImplementedError('Mode %s is not supported yet' % self.dim_order)
            # TODO - fix for other modes
        
        
    def on_frame(self, sender, frameData, **kwargs):
        self.storage.store_frame(self.frame_num, frameData)
        
        self.frame_num += 1
        
        if (self.frame_num >= self.n_frames) and (self.n_frames > 0):
            # if shape_t  == -1 (infinte loop), then self.n_frames is negative, don't stop.
            self.stop()
            return
        
        z_idx, c_idx, t_idx = self._zct_indices(self.frame_num)
        
        self.set_z(z_idx)
        self.set_c(c_idx)
        
        #probably don't need to set anything along the t axis, but provide anyway
        self.set_t(t_idx)
        
        self.on_progress.send(self)
        
    def _collect_metadata(self):
        self.storage.mdh['StartTime'] = time.time()
        self.storage.mdh['AcquisitionType'] = 'Stack'  # TODO - change acquisition type?

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(self.storage.mdh)
        
        
    def start(self):
        self.scope.stackSettings.SetPrevPos(self.scope.stackSettings._CurPos())

        with self.scope.frameWrangler.spooling_stopped():
            # avoid stopping both here and in the SpoolController
            #self.scope.frameWrangler.stop()
            if self._stack_settings:
                self._stack_settings.SetPrevPos(self._stack_settings._CurPos())
            
            self.frame_num = 0
            
            z_idx, c_idx, t_idx = self._zct_indices(self.frame_num)

            self.set_z(z_idx)
            self.set_c(c_idx)
            #probably don't need to set anything along the t axis, but provide anyway
            self.set_t(t_idx)
            
            self._collect_metadata()
            
            self.scope.frameWrangler.onFrame.connect(self.on_frame)

            self.storage.initialise()

            self._running = True

            self.dtStart = datetime.datetime.now() #for spooler compatibility - FIXME
            #self.scope.frameWrangler.start()
        eventLog.register_event_handler(self.storage.event_logger)

        
    def stop(self):
        self.scope.frameWrangler.stop()
        self.scope.frameWrangler.onFrame.disconnect(self.on_frame)
        
        if self._stack_settings:
            self._stack_settings.piezoGoHome()
        
        self.scope.frameWrangler.start()

        try:
            eventLog.remove_event_handler(self.storage.event_logger)
        except ValueError:
            pass

        self.storage.finalise()
        
        self.on_stop.send(self)
        self._running = False
        self.spool_complete = True

    def abort(self):
        self.stop()
        
    def _init_z(self, stack_settings):
        if stack_settings:
            self._z_poss = np.arange(stack_settings.GetStartPos(),
                               stack_settings.GetEndPos() + .95 * stack_settings.GetStepSize(),
                               stack_settings.GetStepSize() * stack_settings.GetDirection())

            self._z_chan = stack_settings.GetScanChannel()
        else:
            self._z_chan = 'z'
            self._z_poss = [self.scope.GetPos()[self._z_chan]]

        self._z_initial_pos = self.scope.GetPos()[self._z_chan]
        
    
    def set_z(self, z_idx):
        self.scope.SetPos(**{self._z_chan: self._z_poss[z_idx]})
        
    def _init_c(self, channel_settings):
        pass
    
    def set_c(self, c_idx):
        pass

    def _init_t(self, time_settings):
        pass

    def set_t(self, t_idx):
        pass

    def status(self):
        return {'spooling' : self._running,
                'frames_spooled' : self.frame_num,
                'spool_complete' : self.spool_complete,}
        
        
            
            
        
class ZStackAcquisition(XYZTCAcquisition):
    """
    Class for a simple Z-Stack acquisition.
    """
    @classmethod
    def from_spool_settings(cls, scope, settings, backend, backend_kwargs={}, series_name=None, spool_controller=None):
         '''Create an XYZTCAcquisition object from a spool_controller settings object'''
    
         backend_kwargs['series_name'] = series_name
    
         return cls(scope=scope, 
                    #dim_order=settings.dim_order, 
                    stack_settings=settings.get('stack_settings', scope.stackSettings), 
                    time_settings=settings.get('time_settings', None), 
                    channel_settings=settings.get('channel_settings', None), 
                    backend=backend, backend_kwargs=backend_kwargs)
    
    @classmethod
    def get_frozen_settings(cls, scope, spool_controller=None):
        return {'stack_settings' : scope.stackSettings.settings(),}