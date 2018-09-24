import threading
from PYME.IO import PZFFormat
import ujson as json
from PYME import config
import numpy as np
import traceback

file_cache = {}

openLock = threading.Lock()

def openH5(filename, mode='r'):
    key = (filename, mode)
    
    with openLock:
        if key in file_cache and file_cache[key].is_alive:
            return file_cache[key]
        else:
            file_cache[key] = H5File(filename, mode)
            return file_cache[key]

from . import h5rFile

class H5File(h5rFile.H5RFile):
    PZFCompression = PZFFormat.DATA_COMP_HUFFCODE
    KEEP_ALIVE_TIMEOUT = 120
    
    @property
    def image_data(self):
        try:
            image_data = getattr(self._h5file.root, 'ImageData')
        except AttributeError:
            image_data = getattr(self._h5file.root, 'PZFImageData', None)
            
        return image_data
    
    @property
    def n_frames(self):
        nFrames = 0
        with h5rFile.tablesLock:
            if not self.image_data is None:
                nFrames = self.image_data.shape[0]
            
        return nFrames
            
        
    def get_listing(self):
        #spoof a directory based listing
        from PYME.IO import clusterListing as cl
        
        listing = {}
        listing['metadata.json'] = cl.FileInfo(cl.FILETYPE_NORMAL, 0)
        listing['events.json'] = cl.FileInfo(cl.FILETYPE_NORMAL, 0)
            
        if self.n_frames > 0:
            for i in range(self.n_frames):
                listing['frame%05d.pzf' % i] = cl.FileInfo(cl.FILETYPE_NORMAL, 0)
                
        return listing
    
    def get_frame(self, frame_num):
        if frame_num >= self.n_frames:
            raise IOError('Frame num %d out of range' % frame_num)
        
        with h5rFile.tablesLock:
            data = self.image_data[frame_num]
        
        if isinstance(data, np.ndarray):
            return PZFFormat.dumps((data.squeeze()), compression = self.PZFCompression)
        else: #already PZF compressed
            return data
    
    def get_file(self, filename):
        if filename == 'metadata.json':
            return self.mdh.to_JSON()
        elif filename == 'events.json':
            raise NotImplementedError('reading events not yet implemented')
        else:
            #names have the form "frameXXXXX.pzf, get frame num
            if not filename.startswith('frame'):
                raise IOError('Invalid component name')
            
            frame_num = int(filename[5:10])
            
            return self.get_frame(frame_num)
        
        
    def put_file(self, filename, data):
        if filename in ['metadata.json', 'Metadata']:
            self.updateMetadata(json.loads(data))
        elif filename == 'events.json':
            raise NotImplementedError('saving events not yet implemented')
        
        elif filename.startswith('frame'):
            #FIXME - this will not preserve ordering
            frame_num = int(filename[5:10])
            self.appendToTable('PZFImageData', data)
        
        
            
                
        
        
