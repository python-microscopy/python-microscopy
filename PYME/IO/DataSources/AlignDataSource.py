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
    moduleName = 'AlignDataSource'
    def __init__(self,dataSource, shifts=[0,0,0], voxelsize=(70., 70., 200.)):
        #self.unmixer = unmixer
        self.dataSource = dataSource
        self.buffer = {}
        
        self.voxelsize = voxelsize
        self.SetShifts(shifts)
        
        

        

    def SetShifts(self, shifts):
        self.shifts = shifts
        self.pixelShifts = [s/v for s, v, in zip(shifts, self.voxelsize)] 
        self.buffer.clear()
        
        #ds = np.concatenate([self.dataSource.getSlice()])
        
    def _getXYShiftedSlice(self, ind):
        sl = self.dataSource.getSlice(ind)
        #print sl.shape
        return ndimage.shift(sl, self.pixelShifts[:2], cval=sl.min())
    
    def getSlice(self,ind):
        if ind in self.buffer.keys():
            return self.buffer[ind]
        else:
            zs = self.pixelShifts[2]
            if zs == 0: #no z shift 
                ret = self._getXYShiftedSlice(ind)
            else: #linearly interpolate between slices
                zs = ind - zs            
                zf = int(np.floor(zs))
                zp = zs - zf
                
                zf = min(max(zf, 0), self.getNumSlices()-1)
                zf1 = min(max(zf, 0), self.getNumSlices()-1)
                #print zp
                ret = (1-zp)*self._getXYShiftedSlice(zf) + zp*self._getXYShiftedSlice(zf1)   
                
            self.buffer[ind] = ret
            return ret

    def getSliceShape(self):
        #return (self.im.size[1], self.im.size[0])
        return self.dataSource.getSliceShape()
        #return self.data.shape[:2]

    def getNumSlices(self):
        return self.dataSource.getNumSlices()

    def getEvents(self):
        return self.dataSource.getEvents()

    def release(self):
        return self.dataSource.release()
        

    def reloadData(self):
        return self.dataSource.reloadData()
        


