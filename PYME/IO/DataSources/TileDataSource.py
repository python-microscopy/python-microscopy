from PYME.IO.FileUtils.nameUtils import getFullExistingFilename
from PYME.IO import MetaDataHandler
#from PYME.IO.FileUtils import readTiff
#from PIL import Image
import glob
import os
import numpy as np
from .BaseDataSource import XYZTCDataSource
#from PYME.misc import TiffImagePlugin #monkey patch PIL with improved tiff support from Priithon

from urllib.parse import parse_qs
#import numpy as np

#from PYME.misc import tifffile

import logging
logger = logging.getLogger(__name__)

class TileDataSource(XYZTCDataSource):
    moduleName = 'TileDataSource'
    def __init__(self, pyramid, level=0):
        # NOTE: We cheat a bit here to allow this to be constructed from an existing pyramid - the cannonical module.DataSource(filename) instantiation 
        # is handled in the function below
        self.level = int(level)

        self.mdh = MetaDataHandler.NestedClassMDHandler()
        self.mdh.copyEntriesFrom(pyramid.mdh)
        
        voxelsize = self.mdh['Pyramid.PixelSize'] * (2 ** self.level)
        self.mdh['voxelsize.x'], self.mdh['voxelsize.y'] = (voxelsize, voxelsize)

        self._pyr = pyramid

        self._nx = pyramid.n_tiles_x/(2**level)
        self._ny = pyramid.n_tiles_y/(2**level)        
        self.tile_size = pyramid.tile_size

        self._levels = None

        XYZTCDataSource.__init__(self)
    

    @classmethod
    def from_filename(cls, filename):
        from PYME.Analysis.tile_pyramid import ImagePyramid
        p = ImagePyramid.load_existing(filename)
        
        return cls(p)
    
    @property
    def levels(self):
        if self._levels is None:
            self._levels = [self.__class__(self._pyr, level) for level in range(self._pyr.depth + 1)]
        
        return self._levels
        
    
    def getSlice(self, ind):
        assert(ind == 0)
        return self._pyr.layers[self.level][:,:].squeeze()
    
    def __getitem__(self, keys):
        logger.debug(f'TileDataSource.__getitem__ called with level: {self.level}, keys: {keys}')
        return self._pyr.layers[self.level].__getitem__(keys)
    
    @property
    def dtype(self):
        return self._pyr.layers[self.level].dtype
    

    def getSliceShape(self):
        return self._pyr.layers[self.level].shape
        #return self.im[0].shape[1:3]
        #return self.data.shape[:2]

    def getNumSlices(self):
        return 1

    def getEvents(self):
        return []

    def release(self):
        #self.im.close()
        pass

    def reloadData(self):
        pass
    
def DataSource(filename, taskQueue=None):
    # cannonical DataSource constructor from filename (needed in order to be able to use this datasource as a task input)
    return TileDataSource.from_filename(filename)