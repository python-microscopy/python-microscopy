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

from PYME.contrib.gohlke import tifffile
from .BaseDataSource import BaseDataSource

class DataSource(BaseDataSource):
    moduleName = 'TiffDataSource'
    def __init__(self, filename, taskQueue=None, chanNum = 0):
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
        
        tf = tifffile.TIFFfile(self.filename)
        
        self.tf = tf # keep a reference for debugging
        
        print(tf.series[0].shape)

        self.im = tf.series[0].pages
        if tf.is_ome:
            print('Detected OME TIFF')
            sh = dict(zip(tf.series[0].axes, tf.series[0].shape))
            print('sh = %s' % sh)
            self.sizeC = sh['C']
            
            axisOrder = tf.series[0].axes[::-1]
            
            self.additionalDims = ''.join([a for a in axisOrder[2:] if sh[a] > 1])
        elif tf.is_rgb:
            print('WARNING: Detected RGB TIFF - data not likely to be suitable for quantitative analysis')
            self.sizeC = 3
            self.RGB = True
            if len(self.im) > 1:
                # we can have multi-page RGB TIFF - why?????
                print('WARNING: Multi-page RGB TIFF detected - where did this come from???')
                self.additionalDims='TC'
            else:
                self.additionalDims = 'C'
            
            
                


    def getSlice(self, ind):
        #self.im.seek(ind)
        #ima = np.array(im.getdata()).newbyteorder(self.endedness)
        #return ima.reshape((self.im.size[1], self.im.size[0]))
        #return self.data[:,:,ind]
        if self.RGB:
            # special case for RGB TIFF
            ind_0 = ind%len(self.im)
            ind_1 = int(ind/len(self.im))
            return self.im[ind_0].asarray(False, False)[0, 0, :,:,ind_1].squeeze()
        
        res =  self.im[ind].asarray(False, False)
        #if res.ndim == 3:
        #print res.shape
        #print self.chanNum
        
        
        res = res[0,self.chanNum, :,:].squeeze()
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
