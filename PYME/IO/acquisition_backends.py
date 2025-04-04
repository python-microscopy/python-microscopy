import abc
import random
import time
import numpy as np

from PYME.IO import MetaDataHandler, clusterIO
from PYME.IO import image
from PYME.IO.DataSources.BaseDataSource import XYZTCWrapper
from PYME.IO.DataSources.ArrayDataSource import ArrayDataSource
from PYME.IO import PZFFormat
from PYME.IO.events import HDFEventLogger, MemoryEventLogger

import logging
logger = logging.getLogger(__name__)
import warnings

class Backend(abc.ABC):
    """
    Base class for acquisition backends.

    """
    def __init__(self, dim_order='XYCZT', shape=[-1, -1,-1,1,1], spoof_timestamps=False, cycle_time=None, **kwargs):
        if not hasattr(self, 'mdh'):
            self.mdh = MetaDataHandler.DictMDHandler()
        self.mdh['imageID'] = self.sequence_id
        
        self._dim_order=dim_order
        self._shape=shape
        self._finished = False
        
        self.mdh['DimOrder'] = dim_order
        self.mdh['SizeC'] =  shape[4]
        self.mdh['SizeT'] =  shape[3]
        self.mdh['SizeZ'] =  shape[2]

        self.imNum = -1 # for event logger compatibility

        self._spoof_timestamps = spoof_timestamps
        self._fakeCamCycleTime = cycle_time
        self._t_start = time.time() # record start time for fake timestamps

        if not hasattr(self, 'event_logger'):
            # if we haven't already defined an event logger in a derived class, create a memory logger
            self.event_logger = MemoryEventLogger(spooler=self, time_fcn=self._timestamp)


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

    def initialise(self):
        """ Called before acquisition starts, may be overridden to e.g. save metadata.
        NOTE: We typically save metadata in advance of the data so that analysis can start while acquisition is underway.
        """
        
        # set imNum to 0 (now handling the first frame)
        # TODO - this is a hack to fix startAq event frame nums, and there has to be a better way of doing this.
        self.imNum = 0
        self._t_start = time.time() # record start time for fake timestamps
        
    
    def finalise(self, events=None):
        """ Called after acquisition is complete, may be overridden to e.g. flush buffers, close files, , write events etc ...
        """
        self._finished = True
        pass

    @classmethod
    def gen_sequence_id(cls):
        """ Generate a unique sequence ID which belongs to this acquisition and can be used as, e.g. a database key 
        """
        return  int(time.time()) & random.randint(0, 2**31) << 31

    @property
    def sequence_id(self):
        """ This sequences unique ID
        """
        if not hasattr(self, '_sequence_id'):
            self._sequence_id = self.gen_sequence_id()

        return self._sequence_id
    
    def getURL(self):
        '''Get URL for the series to pass to other processes so they can open it
        
        Implement in derived classes if appropriate.
        '''
        raise NotImplementedError('getURL() not implemented - this might not be a cluster-aware backend')

    def _timestamp(self):
        if self._spoof_timestamps:
            return self._fake_time()
        else:
            return time.time()
    
    def _fake_time(self):
        """Generate a fake timestamp for use with the simulator where the camera
        cycle time does not match the actual time elapsed to generate the frame"""
        #return self.tStart + self.frame_num*self.scope.cam.GetIntegTime()
        return self._t_start + max(self.imNum, 0)*self._fakeCamCycleTime
    
    @property
    def md(self):
        warnings.warn(DeprecationWarning('.md property is deprecated, use .mdh instead'))
        return self.mdh
    
    def get_n_frames(self):
        # FIXME?? 
        return self.imNum
    
    def finished(self):
        return self._finished


class MemoryBackend(Backend):
    def __init__(self, size_x, size_y, n_frames, dtype='uint16', series_name=None, dim_order='XYCZT', shape=[-1, -1,1,1,1], **kwargs):
        Backend.__init__(self, dim_order, shape, spoof_timestamps=kwargs.pop('spoof_timestamps', False), cycle_time=kwargs.pop('cycle_time', None))
        self.data = np.empty([size_x, size_y, n_frames], dtype=dtype)
        
        # once we have proper xyztc support in the image viewer
        ds = XYZTCWrapper(ArrayDataSource(self.data), dim_order, shape[2], shape[3], shape[4])
        self.image = image.ImageStack(data=ds, mdh=self.mdh)
        
    def store_frame(self, n, frame_data):
        self.data[:,:,n] = frame_data
        self.imNum = n+1

    def finalise(self):
        self.image.events = self.event_logger.events
        super().finalise()




def distfcn_oidic(n_servers, i=None):
    # example distfcn for OIDIC, assuming an XYCZT acquisition order

    if i is None:
        # distribute at random
        return random.randrange(n_servers)

    # TODO - make a bit smarter and/or implement in derived class
    return int(i/6) % n_servers

def distfcn_random(n_servers, i=None):
        # distribute at random
        import random
        return random.randrange(n_servers)

class ClusterBackend(Backend):
    default_compression_settings = {
        'compression' : PZFFormat.DATA_COMP_HUFFCODE,
        'quantization' : PZFFormat.DATA_QUANT_NONE,
        'quantizationOffset' : -1e6, # set to an unreasonable value so that we raise an error if default offset is used
        'quantizationScale' : 1.0
    }

    def __init__(self, series_name, dim_order='XYCZT', shape=[-1, -1,-1,1,1], distribution_fcn=None, compression_settings={}, 
                cluster_h5=False, serverfilter=clusterIO.local_serverfilter, **kwargs):
        from PYME.IO import cluster_streaming
        from PYME.IO import PZFFormat

        Backend.__init__(self, dim_order, shape, spoof_timestamps=kwargs.pop('spoof_timestamps', False), cycle_time=kwargs.pop('cycle_time', None))

        self.series_name = series_name
        self.serverfilter = serverfilter
        self._cluster_h5 = cluster_h5

        if cluster_h5:
            # we need to send all frames to the one server
            server_n = random.randrange(cluster_streaming.n_cluster_servers(serverfilter))
            def dist_fcn_1_server(n_servers, i=None):
                return server_n

            distribution_fcn = dist_fcn_1_server
        elif distribution_fcn is None:
            distribution_fcn = distfcn_random

        
        def _pzfify(data):
            if not isinstance(data, tuple):
                # data is already pre-formatted (normally metadata or similar) - usually a bytes string
                # image data is passed as (frame_data, im_num)
                # TODO - make this a bit cleaner
                return data

            frame_data, im_num = data # packed together as a tuple
            return PZFFormat.dumps(frame_data, sequenceID=self.sequence_id, frameNum = im_num, **self._check_comp_settings(compression_settings))

        self._streamer = cluster_streaming.Streamer(serverfilter=serverfilter, filter=_pzfify, distribution_fcn=distribution_fcn)
        
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

    @property
    def _series_location(self):
        if self._cluster_h5:
            return '__aggregate_h5/' + self.series_name
        else:
            return self.series_name
        
    def getURL(self):
        '''Get URL for the series to pass to other processes so they can open it'''
        return 'PYME-CLUSTER://%s/%s' % (self.serverfilter, self.series_name)
    
    def store_frame(self, n, frame_data):
        fn = '/'.join([self._series_location, 'frame%05d.pzf' % n])

        self._streamer.put(fn, (frame_data, n), i=n)
        self.imNum = n+1

    def initialise(self):
        super().initialise()
        self._streamer.put(self._series_location + '/metadata.json', self.mdh.to_JSON().encode())
    
    def finalise(self):
        #TODO - is this needed
        self._streamer.put(self._series_location + '/final_metadata.json', self.mdh.to_JSON().encode())

        # events are used as a signal in the ClusterPZFDataSource that a series is complete.
        # TODO - better events support - current assumption is that they are passed already formatted as json
        # TODO - use a binary format for saving events - they can be quite
        # numerous

        logger.debug('Putting events')
        self._streamer.put(self._series_location + '/events.json', self.event_logger.to_JSON().encode()) 
        self._streamer.close()
        super().finalise()


class HDFBackend(Backend):
    def __init__(self, series_name, dim_order='XYCZT', shape=[-1, -1,-1,1,1], complevel=6, complib='zlib', evt_time_fcn=time.time, **kwargs):
        import tables

        self.h5File = tables.open_file(series_name, 'w')
        self.mdh = MetaDataHandler.HDFMDHandler(self.h5File)
        self.event_logger = HDFEventLogger(self, self.h5File, time_fcn=self._timestamp)
        Backend.__init__(self, dim_order, shape, spoof_timestamps=kwargs.pop('spoof_timestamps', False), cycle_time=kwargs.pop('cycle_time', None))
           
        self._complevel = complevel
        self._complib = complib
        self.series_name = series_name
        

    def store_frame(self, n, frame_data):
        import tables
        
        if n == 0:
            fs = frame_data.squeeze().shape
            filt = tables.Filters(self._complevel, self._complib, shuffle=True)
            self.imageData = self.h5File.create_earray(self.h5File.root, 'ImageData', tables.UInt16Atom(), (0,fs[0],fs[1]), filters=filt)

        if frame_data.shape[0] == 1:
            self.imageData.append(frame_data)
        else:
            self.imageData.append(frame_data.reshape(1,frame_data.shape[0],frame_data.shape[1]))

        self.imNum = n+1


    def finalise(self, events=None):
        self.imageData.attrs.DimOrder = self._dim_order
        self.imageData.attrs.SizeC = self._shape[4]
        #self.imageData.attrs.SizeZ = self._shape[3]
        #self.imageData.attrs.SizeT = self._shape[2]
        self.imageData.attrs.SizeZ = self._shape[2]
        self.imageData.attrs.SizeT = self._shape[3]

        self.h5File.flush()
        self.h5File.close()

        super().finalise()

    def getURL(self):
        '''Get URL for the series to pass to other processes so they can open it'''
        return self.series_name
    

class TiffFolderBackend(Backend):
    def __init__(self, series_name, dim_order='XYCZT', shape=[-1, -1,-1,1,1],  **kwargs):
        Backend.__init__(self, dim_order, shape, spoof_timestamps=kwargs.pop('spoof_timestamps', False), cycle_time=kwargs.pop('cycle_time', None))

        self.series_name = series_name

    @property
    def _series_location(self):
        return self.series_name 
        
    def getURL(self):
        '''Get URL for the series to pass to other processes so they can open it'''
        return 'PYME-CLUSTER://%s/%s' % (self.serverfilter, self.series_name)
    
    def store_frame(self, n, frame_data):
        try:
            import tifffile
        except ImportError:
            from PYME.contrib.gohlke import tifffile

        fn = '/'.join([self._series_location, 'frame%05d.tif' % n])

        tifffile.imsave(fn, frame_data)
        
        self.imNum = n+1

    def initialise(self):
        super().initialise()
        import os
        os.makedirs(self._series_location, exist_ok=True)
        with open(self._series_location + '/metadata.json', 'w') as f:
            f.write(self.mdh.to_JSON())
    
    def finalise(self):
        #TODO - is this needeed
        with open(self._series_location + '/final_metadata.json', 'w') as f:
            f.write(self.mdh.to_JSON())

        # events are used as a signal in the ClusterPZFDataSource that a series is complete.
        # TODO - better events support - current assumption is that they are passed already formatted as json
        # TODO - use a binary format for saving events - they can be quite
        # numerous

        logger.debug('Putting events')
        with open(self._series_location + '/events.json', 'w') as f:
            f.write(self.event_logger.to_JSON())
        
        super().finalise()