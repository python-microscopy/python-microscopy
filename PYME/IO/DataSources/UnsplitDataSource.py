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

from PYME.Analysis import splitting

#from PYME.localization import splitting

class DataSource(XYTCDataSource):
    moduleName = 'UnsplitDataSource'
    def __init__(self,dataSource, splitting_info : splitting.SplittingInfo, chan=0, shiftfield=None, voxelsize=(70., 70., 200.)):
        #self.unmixer = unmixer
        self.dataSource = dataSource
        self.splitting_info = splitting_info

        self.sliceShape = list(self.dataSource.shape[:-1])
        
        self._raw_w, self._raw_h = self.dataSource.shape[:2]

        self.sliceShape[:2] = self.splitting_info.channel_shape
    
        self.chan = chan

        self.voxelsize = voxelsize
        if shiftfield and chan == 1:
            # FIXME for nChans >= 2 
            self.SetShiftField(shiftfield)


    def SetShiftField(self, shiftField):
       self.shift_corr = splitting.ShiftCorrector(shiftField)

    def getSlice(self,ind):
        sl = self.dataSource.getSlice(ind)
        dsa = sl.squeeze()

        flip = False
        if self.chan > 0:
            if self.flip: #FIXME - change the flip parameter to the data source to be consistent with splitting.get_channel
                flip='up_down'

        c = splitting.get_channel(dsa, self.splitting_info, chan=self.chan)

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
        


