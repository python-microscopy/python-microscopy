import threading
from PYME.IO import PZFFormat
import ujson as json
from PYME import config
import numpy as np
import traceback
from . import h5rFile

from PYME.IO import events
#file_cache = {}

#openLock = threading.Lock()

def openH5(filename, mode='r'):
    key = (filename, mode)
    if (mode == 'r'):
        #if we already have the file open in append mode, allow us to read from it
        key2 = (filename, 'a')
    else:
        key2 = None
    
    with h5rFile.openLock:
        if key in h5rFile.file_cache and h5rFile.file_cache[key].is_alive:
            return h5rFile.file_cache[key]
        elif key2 in h5rFile.file_cache and h5rFile.file_cache[key2].is_alive:
            return h5rFile.file_cache[key2]
        else:
            h5rFile.file_cache[key] = H5File(filename, mode)
            return h5rFile.file_cache[key]



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
    def pzf_index(self):
        if self._pzf_index is None:
            try:
                pi = getattr(self._h5file.root, 'PZFImageIndex')[:]
                self._pzf_index = np.sort(pi, order='FrameNum')
            except AttributeError:
                pass
            
        return self._pzf_index
    
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
        
        if not getattr(self._h5file.root, 'Events', None) is None:
            listing['events.json'] = cl.FileInfo(cl.FILETYPE_NORMAL, 0)
            
        if self.n_frames > 0:
            if not self.pzf_index is None:
                frame_nums = self.pzf_index['FrameNum']
            else:
                frame_nums = range(self.n_frames)
                
            for i in frame_nums:
                listing['frame%05d.pzf' % i] = cl.FileInfo(cl.FILETYPE_NORMAL, 0)
                
        return listing
    
    def get_frame(self, frame_num):
        if frame_num >= self.n_frames:
            raise IOError('Frame num %d out of range' % frame_num)
        
        with h5rFile.tablesLock:
            if not self.pzf_index is None:
                idx = self.pzf_index['Position'][np.searchsorted(self.pzf_index['FrameNum'], frame_num)]
            else:
                idx = frame_num
                
            data = self.image_data[idx]
        
        if isinstance(data, np.ndarray):
            return PZFFormat.dumps((data.squeeze()), compression = self.PZFCompression)
        else: #already PZF compressed
            return data
    
    def get_file(self, filename):
        if filename == 'metadata.json':
            return self.mdh.to_JSON()
        elif filename == 'events.json':
            try:
                events = self._h5file.root.Events[:]
                return json.dumps(list(zip(events['EventName'], events['EventDescr'], events['Time'])), reject_bytes=False)
            except AttributeError:
                raise IOError('File has no events')
            #raise NotImplementedError('reading events not yet implemented')
        else:
            #names have the form "frameXXXXX.pzf, get frame num
            if not filename.startswith('frame'):
                raise IOError('Invalid component name')
            
            frame_num = int(filename[5:10])
            
            return self.get_frame(frame_num)
        
        
    def put_file(self, filename, data):
        if filename in ['metadata.json', 'MetaData']:
            self.updateMetadata(json.loads(data))
        elif filename == 'events.json':
            evts = json.loads(data)
            events_array = events.EventLogger.list_to_array(evts)
                
            self.addEvents(events_array)
        
        elif filename.startswith('frame'):
            #FIXME - this will not preserve ordering
            frame_num = int(filename[5:10])
            self.appendToTable('PZFImageData', data)
        
        
            
                
        
        
