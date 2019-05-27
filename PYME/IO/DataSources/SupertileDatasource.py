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
from .BaseDataSource import BaseDataSource
#from PYME.misc import TiffImagePlugin #monkey patch PIL with improved tiff support from Priithon

from six.moves.urllib.parse import parse_qs
#import numpy as np

#from PYME.misc import tifffile

class DataSource(BaseDataSource):
    moduleName = 'SupertileDataSource'
    def __init__(self, filename, taskQueue=None):
        from PYME.Analysis.tile_pyramid import ImagePyramid
        self.tile_base, query = filename.split('?')
        qp = parse_qs(query)
        self.level = int(qp.get('level', [0])[0])
        self.stride = int(qp.get('stride', [3])[0])
        self.overlap = int(qp.get('overlap', [1])[0])

        self.mdh = MetaDataHandler.load_json(os.path.join(self.tile_base, 'metadata.json'))

        self._pyr = ImagePyramid(self.tile_base, pyramid_tile_size=self.mdh['Pyramid.TileSize'])
        
        self.tile_size = self._pyr.tile_size*(self.stride + self.overlap)


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
