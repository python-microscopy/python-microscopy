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
from .BaseDataSource import XYTCDataSource

class DataSource(XYTCDataSource):
    moduleName = 'CropDataSource'
    def __init__(self,dataSource, xrange=None, yrange=None, trange=None):
        #self.unmixer = unmixer
        self.dataSource = dataSource
        
        if xrange is None:
            self.xrange = (0, self.dataSource.shape[0])
        else:
            self.xrange = xrange
            
        if yrange is None:
            self.yrange = (0, self.dataSource.shape[1])
        else:
            self.yrange = yrange
            
        if trange is None:
            self.trange = (0, self.dataSource.getNumSlices())
        else:
            self.trange = trange
        
    
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
        


