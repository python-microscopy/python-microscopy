import abc
import random
import time
import numpy as np

from PYME.IO import MetaDataHandler, clusterIO
from PYME.IO import image
from PYME.IO.DataSources.BaseDataSource import XYZTCWrapper
from PYME.IO.DataSources.ArrayDataSource import ArrayDataSource
from PYME.IO import PZFFormat

class Backend(abc.ABC):
    def __init__(self, dim_order='XYCZT', shape=[-1, -1,-1,1,1]):
        if not hasattr(self, 'mdh'):
            self.mdh = MetaDataHandler.DictMDHandler()
        self.mdh['imageID'] = self.sequence_id
        
        self._dim_order=dim_order
        self._shape=shape
        
        self.mdh['DimOrder'] = dim_order
        self.mdh['SizeC'] =  shape[4]
        self.mdh['SizeT'] =  shape[3]
        self.mdh['SizeZ'] =  shape[2]

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
        pass
    
    def finalise(self, events=None):
        """ Called after acquisition is complete, may be overridden to e.g. flush buffers, close files, , write events etc ...
        """
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



class ClusterBackend(Backend):
    default_compression_settings = {
        'compression' : PZFFormat.DATA_COMP_HUFFCODE,
        'quantization' : PZFFormat.DATA_QUANT_NONE,
        'quantizationOffset' : -1e6, # set to an unreasonable value so that we raise an error if default offset is used
        'quantizationScale' : 1.0
    }

    def __init__(self, series_name, dim_order='XYCZT', shape=[-1, -1,-1,1,1], distribution_fcn=None, compression_settings={}, 
                cluster_h5=False, serverfilter=clusterIO.local_serverfilter):
        from PYME.IO import cluster_streaming
        from PYME.IO import PZFFormat

        Backend.__init__(self, dim_order, shape)

        self.series_name = series_name
        self._cluster_h5 = cluster_h5

        if cluster_h5:
            # we need to send all frames to the one server
            server_n = random.randrange(cluster_streaming.n_cluster_servers(serverfilter))
            def dist_fcn_1_server(n_servers, i=None):
                return server_n

            distribution_fcn = dist_fcn_1_server

        
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
    
    def store_frame(self, n, frame_data):
        fn = '/'.join([self._series_location, 'frame%05d.pzf' % n])

        self._streamer.put(fn, (frame_data, n), i=n)

    def initialise(self):
        self._streamer.put(self._series_location + '/metadata.json', self.mdh.to_JSON().encode())
    
    def finalise(self, events='[]'):
        #TODO - is this needed
        self._streamer.put(self._series_location + '/final_metadata.json', self.mdh.to_JSON().encode())

        # events are used as a signal in the ClusterPZFDataSource that a series is complete.
        # TODO - better events support - current assumption is that they are passed already formatted as json
        # TODO - use a binary format for saving events - they can be quite
        # numerous
        if events is not None:
            self._streamer.put(self._series_location + '/events.json', events) 
        
        self._streamer.close()


class HDFBackend(Backend):
    def __init__(self, series_name, dim_order='XYCZT', shape=[-1, -1,-1,1,1], complevel=6, complib='zlib'):
        import tables

        self.h5File = tables.open_file(series_name, 'w')
        self.mdh = MetaDataHandler.HDFMDHandler(self.h5File)
        Backend.__init__(self, dim_order, shape)
           
        self._complevel = complevel
        self._complib = complib
        

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


    def finalise(self, events=None):
        self.imageData.attrs.DimOrder = self._dim_order
        self.imageData.attrs.SizeC = self._shape[4]
        self.imageData.attrs.SizeZ = self._shape[3]
        self.imageData.attrs.SizeT = self._shape[2]

        self.h5File.flush()
        self.h5File.close()