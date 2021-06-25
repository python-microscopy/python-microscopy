import numpy as np
from PYME.contrib import dispatch
from PYME.IO import MetaDataHandler
from PYME.IO import image
from PYME.IO.DataSources.BaseDataSource import XYZTCWrapper
from PYME.IO.DataSources.ArrayDataSource import ArrayDataSource

import abc
import random
import time

class Backend(abc.ABC):
    def __init__(self):
        self.mdh = MetaDataHandler.DictMDHandler()
        self.mdh['imageID'] = self.sequence_id    

    @abc.abstractmethod
    def store_frame(self, n, frame_data):
        """ Store frame data (over-ride in derived classes)

        Parameters
        ----------

        n : int
            frame index / position in sequence
        frame_data : np.ndarray
            data for the frame as a numpy array
        """
        raise NotImplementedError('Must be over-ridden in derived class')
    
    def finalise(self):
        """ Called after acquisition is complete, may be overridden to e.g. flush buffers, close files, etc ...
        """
        pass

    @classmethod
    def gen_sequence_id(cls):
        """ Generate a unique sequence ID which belongs to this acquisition and can be used as, e.g. a database key 
        """
        return  int(time.time()) & random.randint(0, 2**31) << 31

    @property
    def sequence_id(self):
        if not hasattr(self, '_sequence_id'):
            self._sequence_id = self.gen_sequence_id()

        return self._sequence_id


class MemoryBackend(Backend):
    def __init__(self, size_x, size_y, n_frames, dtype='uint16', dim_order='XYCZT', shape=[-1, -1,1,1,1]):
        Backend.__init__(self)
        self.data = np.empty([size_x, size_y, n_frames], dtype=dtype)
        
        # once we have proper xyztc support in the image viewer
        ds = XYZTCWrapper(ArrayDataSource(self.data), dim_order, shape[2], shape[3], shape[4])
        self.image = image.ImageStack(data=ds, mdh=self.mdh)
        
    def store_frame(self, n, frame_data):
        self.data[:,:,n] = frame_data

def distfcn_oidic(n_servers, i=None):
            # example distfcn for OIDIC, assuming an XYCZT acquisition order

            if i is None:
                # distribute at random
                return random.randrange(n_servers)

            # TODO - make a bit smarter and/or implement in derived class
            return int(i/6) % n_servers


from PYME.IO import PZFFormat
class ClusterBackend(Backend):
    default_compression_settings = {
        'compression' : PZFFormat.DATA_COMP_HUFFCODE,
        'quantization' : PZFFormat.DATA_QUANT_NONE,
        'quantizationOffset' : -1e6, # set to an unreasonable value so that we raise an error if default offset is used
        'quantizationScale' : 1.0
    }


    def __init__(self, series_name, dim_order='XYCZT', shape=[-1, -1,1,1,1], distribution_fcn=None, compression_settings={}):
        from PYME.IO import cluster_streaming
        from PYME.IO import PZFFormat

        Backend.__init__(self)

        self.series_name = series_name

        def _pzfify(data):
            frame_data, im_num = data # packed together as a tuple
            return PZFFormat.dumps(frame_data, sequenceID=self.sequence_id, frameNum = im_num, **self._check_comp_settings(compression_settings))
        
        self._streamer = cluster_streaming.Streamer(filter=_pzfify, distribution_fcn=distribution_fcn)
        
    @classmethod
    def _check_comp_settings(cls, compression_settings):
        compSettings = {}
        compSettings.update(cls.default_compression_settings)
        compSettings.update(compression_settings)
        
        if not compSettings['quantization'] == PZFFormat.DATA_QUANT_NONE:
            # do some sanity checks on our quantization parameters
            # note that these conversions will throw a ValueError if the settings are not numeric
            offset = float(compSettings['quantizationOffset'])
            scale = float(compSettings['quantizationScale'])
            
            # these are potentially a bit too permissive, but should catch an offset which has been left at the
            # default value
            assert(offset >= 0)
            assert(scale >=.001)
            assert(scale <= 100)

        return compSettings
    
    def store_frame(self, n, frame_data):
        self.data[:,:,n] = frame_data
        fn = '/'.join([self.series_name, 'frame%05d.pzf' % self.n])

        self._streamer.put(fn, (frame_data, n), i=n)

    def finalise(self):
        # TODO - add an `initialize()` method and write metadata there so we can use backends for spoolers too?
        self._streamer.put(self.seriesName + '/metadata.json', self.md.to_JSON().encode(), i=0)
        # events are used as a signal in the ClusterPZFDataSource that a series is complete.
        #TODO - better event support?
        self._streamer.put(self.seriesName + '/events.json', '[]', i=0) 
        self._streamer.close()


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
        
        
            
            
        
        