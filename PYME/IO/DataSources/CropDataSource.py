#!/usr/bin/python
##################
# UnsplitDataSource.py
#
# Copyright David Baddeley, 2011
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
import numpy as np
from scipy import ndimage
from .BaseDataSource import BaseDataSource

class DataSource(BaseDataSource): 
    moduleName = 'CropDataSource'
    def __init__(self,dataSource, xrange=None, yrange=None, trange=None):
        #self.unmixer = unmixer
        self.dataSource = dataSource
        
        sizeC = self.dataSource.sizeC

        if xrange is None or tuple(xrange) == (0, -1):
            self.xrange = (0, self.dataSource.shape[0])
        else:
            # force to positive values we can handle simply in getSliceShape
            x0 = xrange[0] if xrange[0] > -1 else self.dataSource.shape[0] + xrange[0]
            x1 = xrange[1] if xrange[1] > -1 else self.dataSource.shape[0] + xrange[1]
            assert x1 - x0 > 0 and x1 - x0 <= self.dataSource.shape[0]
            assert x0 < self.dataSource.shape[0] and x0 > 0
            assert x1 < self.dataSource.shape[0] and x1 > 0
            self.xrange = (x0, x1)
            
        if yrange is None or tuple(yrange) == (0, -1):
            self.yrange = (0, self.dataSource.shape[1])
        else:
            # force to positive values we can handle simply in getSliceShape
            y0 = yrange[0] if yrange[0] > -1 else self.dataSource.shape[1] + yrange[0]
            y1 = yrange[1] if yrange[1] > -1 else self.dataSource.shape[1] + yrange[1]
            assert y1 - y0 > 0 and y1 - y0 <= self.dataSource.shape[1]
            assert y0 < self.dataSource.shape[1] and y0 > 0
            assert y1 < self.dataSource.shape[1] and y1 > 0
            self.yrange = (y0, y1)
            
        if trange is None or tuple(trange) == (0, -1):
            self.trange = (0, self.dataSource.getNumSlices())
        else:
            # force to positive values we can handle simply in getNumSlices
            n_slices = self.dataSource.getNumSlices()
            t0 = trange[0] if trange[0] > -1 else n_slices + trange[0]
            t1 = trange[1] if trange[1] > -1 else n_slices + trange[1]
            assert t1 - t0 > 0 and t1 - t0 <= n_slices
            assert t0 < n_slices and t0 > 0
            assert t1 < n_slices and t1 > 0
            self.trange = (t0, t1)
        
    
    def getSlice(self,ind):
        sl = self.dataSource.getSlice(ind + self.trange[0])
        
        return sl[self.xrange[0]:self.xrange[1], self.yrange[0]:self.yrange[1]]

    def getSliceShape(self):
        #return (self.im.size[1], self.im.size[0])
        return (self.xrange[1] - self.xrange[0], self.yrange[1] - self.yrange[0])
        #return self.data.shape[:2]

    def getNumSlices(self):
        return self.trange[1] - self.trange[0]

    def getEvents(self):
        return self.dataSource.getEvents()

    def release(self):
        return self.dataSource.release()
        

    def reloadData(self):
        return self.dataSource.reloadData()
        


