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
import numpy
from BaseDataSource import BaseDataSource

class DataSource(BaseDataSource): 
    moduleName = 'UnsplitDataSource'
    def __init__(self,dataSource, ROI, chan=0, flip=True, shiftfield=None, voxelsize=(70., 70., 200.)):
        #self.unmixer = unmixer
        self.dataSource = dataSource
        self.sliceShape = list(self.dataSource.shape[:-1])
        self.sliceShape[1]/=2
        
        self.ROI = ROI
        self.chan = chan

        #self.pixelsize = pixelsize
        self.flip = flip
        self.voxelsize = voxelsize
        if shiftfield and chan == 1:
            self.SetShiftField(shiftfield)

        

    def SetShiftField(self, shiftField):
        #self.shiftField = shiftField
        #self.shiftFieldName = sfname
        X, Y = numpy.ogrid[:512, :256]

        self.X2 = numpy.round(X - shiftField[0](X*self.voxelsize[0], Y*self.voxelsize[1])/self.voxelsize[0]).astype('i')
        self.Y2 = numpy.round(Y - shiftField[1](X*self.voxelsize[0], Y*self.voxelsize[1])/self.voxelsize[1]).astype('i')

        x1, y1, x2, y2 = self.ROI
        x1 = x1 - 1
        #x2 = self.scope.cam.GetROIX2()
        y1 = y1 - 1

        #print self.X2.shape

        Xn = self.X2[x1:x2, y1:(y1 + self.sliceShape[1])] - x1
        Yn = self.Y2[x1:x2, y1:(y1 + self.sliceShape[1])] - y1

        #print Xn.shape

        self.Xn = numpy.maximum(numpy.minimum(Xn, self.sliceShape[0]-1), 0)
        self.Yn = numpy.maximum(numpy.minimum(Yn, self.sliceShape[1]-1), 0)

    def getSlice(self,ind):
        sl = self.dataSource.getSlice(ind)
        
        #sl = self.unmixer.Unmix(sl, self.mixmatrix, self.offset, self.ROI)

        #print self.chan, sl.shape

        #print sl.shape

        dsa = sl.squeeze()

        if self.chan == 0:
            return dsa[:, :(dsa.shape[1]/2)]
        else: #chan = 1
            r_ = dsa[:, (dsa.shape[1]/2):]
            if self.flip:
                r_ = numpy.fliplr(r_)

            if 'X2' in dir(self):
                return r_[self.Xn, self.Yn]
            else:
                return r_

        

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
        


