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
from __future__ import print_function
import numpy as np
from .BaseDataSource import XYTCDataSource

from PYME.localization import splitting

class DataSource(XYTCDataSource):
    moduleName = 'UnsplitDataSource'
    def __init__(self,dataSource, ROI, chan=0, flip=True, shiftfield=None, voxelsize=(70., 70., 200.), chanROIs=None):
        #self.unmixer = unmixer
        self.dataSource = dataSource
        self.sliceShape = list(self.dataSource.shape[:-1])
        
        self._raw_w, self._raw_h = self.dataSource.shape[:2]
        
        self.sliceShape[1] = int(self.sliceShape[1] / 2)
        
        if not chanROIs is None:
            x, y, w, h = chanROIs[0]
            #self.sliceShape = [min(self.dataSource.shape[0], w), min(self.dataSource.shape[1], h)]
            
            for x_, y_, w_, h_ in chanROIs:
                w = min(self._raw_w - x_, w)
                h = min(self._raw_h - y_, h)
                
            self.sliceShape = [w, h]
        
        self.ROI = ROI
        self.chan = chan

        #self.pixelsize = pixelsize
        self.flip = flip
        self.voxelsize = voxelsize
        if shiftfield and chan == 1:
            #fixme for nChans >= 2 
            self.SetShiftField(shiftfield)

        self.chanROIs = chanROIs

    def SetShiftField(self, shiftField):
       from PYME.Analysis import splitting
       self.shift_corr = splitting.ShiftCorrector(shiftField)

    def getSlice(self,ind):
        from PYME.Analysis import splitting
        sl = self.dataSource.getSlice(ind)
        dsa = sl.squeeze()

        flip = False
        if self.chan > 0:
            if self.flip: #FIXME - change the flip parameter to the data source to be consistent with splitting.get_channel
                flip='up_down'

        c = splitting.get_channel(dsa, self.ROI, flip=flip, chanROIs=self.chanROIs, chan=self.chan)

        if hasattr(self, 'shift_corr'):
            # do shift correction
            return self.shift_corr.correct(c, self.voxelsize, origin_nm=self.voxelsize*np.array(self.ROI[:2]))

        return c

        

    def getSliceShape(self):
        #return (self.im.size[1], self.im.size[0])
        return tuple(self.sliceShape)
        #return self.data.shape[:2]

    def getNumSlices(self):
        return self.dataSource.getNumSlices()

    def getEvents(self):
        return self.dataSource.getEvents()

    def release(self):
        return self.dataSource.release()
        

    def reloadData(self):
        return self.dataSource.reloadData()
        


