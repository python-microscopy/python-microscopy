#!/usr/bin/python

##################
# TiffDataSource.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

from PYME.IO.FileUtils.nameUtils import getFullExistingFilename
from PYME.IO import MetaDataHandler
#from PYME.IO.FileUtils import readTiff
#from PIL import Image
import glob
import os
import numpy as np
from .BaseDataSource import XYTCDataSource
#from PYME.misc import TiffImagePlugin #monkey patch PIL with improved tiff support from Priithon

from six.moves.urllib.parse import parse_qs
#import numpy as np

#from PYME.misc import tifffile

class SupertileDataSource(XYTCDataSource):
    moduleName = 'SupertileDataSource'
    def __init__(self, pyramid, level=0, stride=3, overlap=1):
        # NOTE: We cheat a bit here to allow this to be constructed from an existing pyramid - the cannonical module.DataSource(filename) instantiation 
        # is handled in the function below
        self.level = int(level)
        self.stride = int(stride)
        self.overlap = int(overlap)

        self.mdh = MetaDataHandler.NestedClassMDHandler()
        self.mdh.copyEntriesFrom(pyramid.mdh)
        
        voxelsize = self.mdh['Pyramid.PixelSize'] * (2 ** self.level)
        self.mdh['voxelsize.x'], self.mdh['voxelsize.y'] = (voxelsize, voxelsize)

        self._pyr = pyramid
        
        self.tile_size = self._pyr.tile_size*(self.stride + self.overlap)
    
    @staticmethod
    def from_raw_tile_series(filename):
        # TODO - do we really want/need this?? We should not be creating the pyramid from scratch in the datasource as this violates the assumption that
        # datasource loading is comparatively fast / lightweight. Delete me??
        import warnings
        warnings.warn('This function might dissappear')
        from PYME.Analysis.tile_pyramid import create_pyramid_from_dataset
        from tempfile import TemporaryDirectory
        
        tile_base, query = filename.split('?')
        qp = parse_qs(query)
        level = int(qp.get('level', [0])[0])
        stride = int(qp.get('stride', [3])[0])
        overlap = int(qp.get('overlap', [1])[0])
        tile_size = int(qp.get('tilesize', [256])[0])
        
        p = create_pyramid_from_dataset(tile_base, TemporaryDirectory(), tile_size)
        
        return SupertileDataSource(p, level, stride, overlap)
    
    @staticmethod
    def from_filename(filename):
        from PYME.Analysis.tile_pyramid import ImagePyramid
        
        tile_base, query = filename.split('?')
        qp = parse_qs(query)
        level = int(qp.get('level', [0])[0])
        stride = int(qp.get('stride', [3])[0])
        overlap = int(qp.get('overlap', [1])[0])        
        p = ImagePyramid.load_existing(tile_base)
        
        return SupertileDataSource(p, level, stride, overlap)

    @property
    def tile_coords(self):
        tc = np.array(self._pyr.get_layer_tile_coords(self.level))

        x0, y0 = tc.min(axis=0)
        xm, ym  = tc.max(axis=0)
        
        xvs = np.arange(x0, xm + self.stride, self.stride)
        yvs = np.arange(y0, ym + self.stride, self.stride)
        
        
        xc = (xvs[:,None]*np.ones_like(yvs)[None,:]).ravel()
        yc = (yvs[None, :] * np.ones_like(xvs)[:, None]).ravel()
        
        return np.vstack([xc, yc]).T
    
    @property
    def tile_coords_um(self):
        """
        Returns
        -------
        ndarray
            coordinates of each frame, in micrometers, referenced to the origin
            of the camera chip.
        """
        px_size = self.mdh['voxelsize.x']
        
        # Pyramid.x0 and Pyramid.y0 should be referenced to the camera origin
        p0 = np.array([self.mdh['Pyramid.x0'], self.mdh['Pyramid.y0']])
        
        return px_size*self._pyr.tile_size*self.tile_coords + p0[None,:]
        
    
    def getSlice(self, ind):
        x, y = self.tile_coords[ind, :]
        return self._pyr.get_oversize_tile(self.level, x, y, span=(self.stride+self.overlap))

    def getSliceShape(self):
        return (self.tile_size, self.tile_size)
        #return self.im[0].shape[1:3]
        #return self.data.shape[:2]

    def getNumSlices(self):
        return len(self.tile_coords)

    def getEvents(self):
        return []

    def release(self):
        #self.im.close()
        pass

    def reloadData(self):
        pass
    
def DataSource(filename, taskQueue=None):
    # cannonical DataSource constructor from filename (needed in order to be able to use this datasource as a task input)
    return SupertileDataSource.from_filename(filename)
