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
from .BaseDataSource import XYTCDataSource, XYZTCDataSource, BaseDataSource

class _DataSource(XYTCDataSource):
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

class DataSource(XYZTCDataSource):
    moduleName = 'CropDataSource'

    @classmethod
    def _sliceify(cls, range, nmax):
        if range is None:
            return slice(0, nmax, 1)
        else:
            if not isinstance(range, slice):
                range = slice(*range)
            
            return slice(*range.indices(nmax))

    def __init__(self,dataSource, xrange=None, yrange=None, zrange=None, trange=None):
        #self.unmixer = unmixer
        self.dataSource = dataSource #type: XYZTCDataSource

        # check than input datasource is xyztc (duck-typed by checking dimension number so that 5D ListWraps are also OK) 
        assert(isinstance(dataSource, BaseDataSource))
        assert(dataSource.ndim == 5)

        self.xslice = self._sliceify(xrange, self.dataSource.shape[0])
        self.yslice = self._sliceify(yrange, self.dataSource.shape[1])
        self.zslice = self._sliceify(zrange, self.dataSource.shape[2])
        self.tslice = self._sliceify(trange, self.dataSource.shape[3])
        
        
        szs = [int(np.floor((r.stop-r.start)/r.step)) for r in [self.xslice, self.yslice, self.zslice, self.tslice]] + [self.dataSource.shape[4],]

        self.set_dim_order_and_size(self.dataSource._input_order, szs[2], szs[3], szs[4])
        
        self._shape = tuple(szs)
        self._dtype = dataSource.dtype
        
        self._i_z_stride = self.dataSource._z_stride
        self._i_t_stride = self.dataSource._t_stride
        self._i_c_stride = self.dataSource._c_stride


    
    def getSlice(self,ind):
        o_strides = np.array((self._z_stride, self._t_stride, self._c_stride))
        i_strides = np.array((self._i_z_stride*self.zslice.step, self._i_t_stride*self.tslice.step, self._i_c_stride)).astype('i')
        i_offsets = np.array((self.zslice.start, self.tslice.start, 0))

        stride_order = np.argsort(o_strides)
        #print(stride_order, o_strides, i_strides)
        so_strides = o_strides[stride_order]

        i = int(np.floor(ind/so_strides[-1]))
        j = int(np.floor((ind-i*so_strides[-1])/so_strides[-2]))
        k = ind - i*so_strides[-1] -j*so_strides[-2]

        ijk = int(((np.array([k,j,i]) + i_offsets[stride_order])*i_strides[stride_order]).sum())

        #print(ind, ijk, (k, j, i))

        sl = self.dataSource.getSlice(ijk)

        #print(sl.shape)
        
        sl = sl[self.xslice, self.yslice]

        #print(sl.shape)
        return sl

    def getSliceShape(self):
        #return (self.im.size[1], self.im.size[0])
        #return (self.xrange[1] - self.xrange[0], self.yrange[1] - self.yrange[0])
        return self._shape[:2]

    def getNumSlices(self):
        return np.prod(self.shape[2:])

    def getEvents(self):
        return self.dataSource.getEvents()

    def release(self):
        return self.dataSource.release()  

    def reloadData(self):
        return self.dataSource.reloadData()
        


def crop_image(image, xrange=None, yrange=None, zrange=None, trange=None):
    from PYME.IO.image import ImageStack

    vx, vy, vz = image.voxelsize
    ox, oy, oz = image.origin

    def _offset(r):     
        if r is None:
            return 0
        elif isinstance(r, slice):
            return r.start
        else:
            return r[0]

    cropped = DataSource(image.data_xyztc, xrange=xrange, yrange=yrange, zrange=zrange, trange=trange)
    
    im = ImageStack(cropped, titleStub='Cropped Image')
    im.mdh.copyEntriesFrom(image.mdh)
    im.mdh['Parent'] = image.filename
    #im.mdh['Processing.CropROI'] = roi
    
    im.mdh['Origin.x'] = ox + vx*_offset(xrange)
    im.mdh['Origin.y'] = oy + vy*_offset(yrange)
    im.mdh['Origin.z'] = oz + vz*_offset(zrange)
    return im

def roi_crop_image(image, roi, z=True, t=False):
    if z:
        zrange = roi[2]
    else:
        zrange = None

    if t:
        trange = roi[3]
    else:
        trange = None

    xrange = roi[0]
    yrange = roi[1]

    return crop_image(image, xrange, yrange, zrange, trange)