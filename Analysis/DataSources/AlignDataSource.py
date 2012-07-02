#!/usr/bin/python
##################
# UnsplitDataSource.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy as np
from scipy import ndimage
from BaseDataSource import BaseDataSource

class DataSource(BaseDataSource): 
    moduleName = 'AlignDataSource'
    def __init__(self,dataSource, shifts=[0,0,0], voxelsize=(70., 70., 200.)):
        #self.unmixer = unmixer
        self.dataSource = dataSource
        
        self.voxelsize = voxelsize
        self.SetShifts(shifts)

        

    def SetShifts(self, shifts):
        self.shifts = shifts
        self.pixelShifts = [s/v for s, v, in zip(shifts, self.voxelsize)] 
        
        #ds = np.concatenate([self.dataSource.getSlice()])
        
    def _getXYShiftedSlice(self, ind):
        sl = self.dataSource.getSlice(ind)
        #print sl.shape
        return ndimage.shift(sl, self.pixelShifts[:2], cval=sl.min())
    
    def getSlice(self,ind):
        zs = self.pixelShifts[2]
        if zs == 0: #no z shift 
            return self._getXYShiftedSlice(ind)
        else: #linearly interpolate between slices
            zs = ind - zs            
            zf = int(np.floor(zs))
            zp = zs - zf
            
            zf = min(max(zf, 0), self.getNumSlices()-1)
            zf1 = min(max(zf, 0), self.getNumSlices()-1)
            #print zp
            return (1-zp)*self._getXYShiftedSlice(zf) + zp*self._getXYShiftedSlice(zf1)   

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
        


