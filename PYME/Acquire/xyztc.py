import numpy as np
import time

from PYME.contrib import dispatch
from PYME.IO import MetaDataHandler
from PYME.IO.acquisition_backends import MemoryBackend


class XYZTCAcquisition(object):
    def __init__(self, scope, dim_order='XYCZT', stack_settings=None, time_settings=None, channel_settings=None, backend=MemoryBackend):
        if stack_settings is None:
            stack_settings = scope.stackSettings

        assert(dim_order[:2] == 'XY') #first two dimensions must be XY (camera acquisition)
        # TODO more sanity checks on dim_order
        
        self.dim_order = dim_order
        self.scope = scope
        
        self.shape_x, self.shape_y = scope.frameWrangler.currentFrame.shape[:2]
        self.shape_z = stack_settings.GetSeqLength()
        self.shape_t = getattr(time_settings, 'num_timepoints', 1)
        self.shape_c = getattr(channel_settings, 'num_channels', 1)
        
        # note shape_t can be negative if we want to run until explicitly stopped
        self.n_frames = self.shape_z*self.shape_c*self.shape_t
        self.frame_num = 0
        
        self.storage = backend(self.shape_x, self.shape_y, self.n_frames, dim_order=dim_order, shape=self.shape)
        
        #do any precomputation
        self._init_z(stack_settings)
        self._init_t(time_settings)
        self._init_c(channel_settings)

        self.on_single_frame = dispatch.Signal()  #dispatched when a frame is ready
        self.on_series_end = dispatch.Signal()  #dispatched when a sequence is complete
    
    @property
    def shape(self):
        return self.shape_x, self.shape_y, self.shape_z, self.shape_t, self.shape_c
        
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
            self.finish()
            return
        
        z_idx, c_idx, t_idx = self._zct_indices(self.frame_num)
        
        self.set_z(z_idx)
        self.set_c(c_idx)
        
        #probably don't need to set anything along the t axis, but provide anyway
        self.set_t(t_idx)
        
        self.on_single_frame.send(self)
        
    def _collect_metadata(self):
        self.storage.mdh['StartTime'] = time.time()
        self.storage.mdh['AcquisitionType'] = 'Stack'  # TODO - change acquisition type?

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(self.storage.mdh)
        
        
    def start(self):
        self.scope.frameWrangler.stop()
        self.frame_num = 0
        
        z_idx, c_idx, t_idx = self._zct_indices(self.frame_num)

        self.set_z(z_idx)
        self.set_c(c_idx)
        #probably don't need to set anything along the t axis, but provide anyway
        self.set_t(t_idx)
        
        self._collect_metadata()
        
        self.scope.frameWrangler.onFrame.connect(self.on_frame)
        self.scope.frameWrangler.start()
        
        
    def finish(self):
        self.scope.frameWrangler.stop()
        self.scope.frameWrangler.onFrame.disconnect(self.on_frame)
        self.scope.frameWrangler.start()
        
        self.on_series_end.send(self)
        
    def _init_z(self, stack_settings):
        self._z_poss = np.arange(stack_settings.GetStartPos(),
                               stack_settings.GetEndPos() + .95 * stack_settings.GetStepSize(),
                               stack_settings.GetStepSize() * stack_settings.GetDirection())

        self._z_chan = stack_settings.GetScanChannel()
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
        
        
            
            
        
        