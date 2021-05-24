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
#from PYME.IO.FileUtils import readTiff
#import Image
#from PYME.misc import TiffImagePlugin #monkey patch PIL with improved tiff support from Priithon

#import numpy as np

try:
    import tifffile
    local_tifffile = False
except ImportError:
    local_tifffile = True
    from PYME.contrib.gohlke import tifffile
    
from .BaseDataSource import XYZTCDataSource

import logging
logger = logging.getLogger(__name__)

class DataSource(XYZTCDataSource):
    moduleName = 'TiffDataSource'
    def __init__(self, filename, taskQueue=None, chanNum = 0, series=0):
        self.filename = getFullExistingFilename(filename)#convert relative path to full path
        self.chanNum = chanNum
        self.RGB = False
        
        #self.data = readTiff.read3DTiff(self.filename)

        #self.im = Image.open(filename)

        #self.im.seek(0)

        #PIL's endedness support is subtly broken - try to fix it
        #NB this is untested for floating point tiffs
        #self.endedness = 'LE'
        #if self.im.ifd.prefix =='MM':
        #    self.endedness = 'BE'

        #to find the number of images we have to loop over them all
        #this is obviously not ideal as PIL loads the image data into memory for each
        #slice and this is going to represent a huge performance penalty for large stacks
        #should still let them be opened without having all images in memory at once though
        #self.numSlices = self.im.tell()
        
        #try:
        #    while True:
        #        self.numSlices += 1
        #        self.im.seek(self.numSlices)
                
        #except EOFError:
        #    pass

        print((self.filename))
        
        if local_tifffile:
            logger.info('Using PYMEs built-in, old version of tifffile, better support for ImageJ tiffs can be had with the more recent pip version (`pip install tifffile`)')
            tf = tifffile.TIFFfile(self.filename)
        else:
            tf = tifffile.TiffFile(self.filename)
        
        self.tf = tf # keep a reference for debugging
        
        print(tf.series[series].shape)
        self.im = tf.series[series].pages
        
        axisOrder = 'XYZTC'
        size_z = len(self.im)
        size_c, size_t = 1,1
        
        if tf.is_ome or ((not local_tifffile)):
            #print('Detected OME TIFF')
            sh = {'Z':1, 'T': 1,'C':1}
            _axes = tf.series[series].axes
            if 'I' in _axes:
                logger.info('Tiff file does not fully specify axes (axes=%s)' % (_axes.replace('I', '?')[::-1]))
                if 'Z' not in _axes:
                    logger.info('Assuming unknown axis is Z') # TODO - explain how to change axes later
                    _axes = _axes.replace('I', 'Z')
                elif 'C' not in _axes:
                    logger.info('Assuming unknown axis is C')
                    _axes = _axes.replace('I', 'C')
                elif 'T' not in _axes:
                    logger.info('Assuming unkown axis is T')
                    _axes = _axes.replace('I', 'T')
                else:
                    logger.warning('Unknown axis with all standard axes defined - data might not read correctly')
                
            sh.update(dict(zip(_axes, tf.series[0].shape)))
            logger.debug('sh = %s' % sh)
            size_c = sh['C']
            size_z  = sh['Z']
            size_t = sh['T']
            
            axisOrder = _axes[::-1]
            
            axisOrder = axisOrder + ''.join([a for a in ['Z', 'T', 'C'] if not a in axisOrder])
            
            logger.debug('raw TIFF axisOrder = %s' %axisOrder)
            
            #self.additionalDims = ''.join([a for a in axisOrder[2:] if sh[a] > 1])
        elif tf.is_rgb:
            print('WARNING: Detected RGB TIFF - data not likely to be suitable for quantitative analysis')
            size_c = 3
            self.RGB = True
            if len(self.im) > 1:
                # we can have multi-page RGB TIFF - why?????
                print('WARNING: Multi-page RGB TIFF detected - where did this come from???')
                
            axisOrder = 'XYCZT'
            size_z = len(self.im)
            size_t = 1
            
        XYZTCDataSource.__init__(self, input_order=axisOrder, size_z=size_z, size_t=size_t, size_c=size_c)
        
        sl0 = self.getSlice(0)
        self._dtype = sl0.dtype
        self._shape = [sl0.shape[0], sl0.shape[1], size_z, size_t, size_c]
    
    def getSlice(self, ind):
        #self.im.seek(ind)
        #ima = np.array(im.getdata()).newbyteorder(self.endedness)
        #return ima.reshape((self.im.size[1], self.im.size[0]))
        #return self.data[:,:,ind]
        if self.RGB:
            # special case for RGB TIFF
            ind_0 = ind%len(self.im)
            ind_1 = int(ind/len(self.im))
            return self.im[ind_0].asarray(squeeze=False)[0, 0, :,:,ind_1].squeeze()
        
        if local_tifffile:
            res =  self.im[ind].asarray(squeeze=False, colormapped=False)
            res = res[0, self.chanNum, :, :].squeeze()
        else:
            res = self.im[ind].asarray()
        #if res.ndim == 3:
        #print res.shape
        #print self.chanNum
        
        #print res.shape
        return res
    
    def getSliceShape(self):
        #return (self.im.size[1], self.im.size[0])
        if len(self.im[0].shape) == 2:
            return self.im[0].shape
        elif self.RGB:
            return self.im[0].shape[:2]
        else:
            return self.im[0].shape[1:3] #FIXME - when is this used?
        #return self.data.shape[:2]

    def getNumSlices(self):
        if self.RGB:
            return len(self.im)*3
        
        return len(self.im)

    def getEvents(self):
        return []

    def release(self):
        #self.im.close()
        pass

    def reloadData(self):
        pass
