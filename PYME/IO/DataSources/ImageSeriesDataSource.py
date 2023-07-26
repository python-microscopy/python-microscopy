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
from PYME.IO import MetaDataHandler
#from PYME.IO.FileUtils import readTiff
from PIL import Image
import glob
import os
import numpy as np
from .BaseDataSource import XYTCDataSource
#from PYME.misc import TiffImagePlugin #monkey patch PIL with improved tiff support from Priithon

#import numpy as np

#from PYME.misc import tifffile
try:
    import tifffile
except ImportError:
    from PYME.contrib.gohlke import tifffile

class DataSource(XYTCDataSource):
    moduleName = 'ImageSeriesDataSource'
    def __init__(self, filename='', metadata=None, taskQueue=None):
        if filename is not '':
            self.filename = getFullExistingFilename(filename)#convert relative path to full path
            #use metadata for glob
            md = MetaDataHandler.SimpleMDHandler(self.filename)
        else:
            md = metadata
            self.filename = ''

        pattern = md.getEntry('SeriesPattern')

        self.files = glob.glob(os.path.join(os.path.split(self.filename)[0], pattern))

        self.files.sort()

        f0 = self.files[0]
        if f0.endswith('.tif'):
            self.im0 = tifffile.imread(f0).squeeze()
            self._slice_shape = self.im0.shape[:2]
        else:
            self.im0 = Image.open(self.files[0])
            self._slice_shape = self.im0.size[1], self.im0.size[0]

            #self.im.seek(0)
    
            #PIL's endedness support is subtly broken - try to fix it
            #NB this is untested for floating point tiffs
            self.endedness = 'LE'
            if self.im0.ifd.prefix =='MM':
                self.endedness = 'BE'
                
            print((self.im0.ifd.prefix))
            print((self.endedness))

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

        #self.im = tifffile.TIFFfile(self.filename)


    def getSlice(self, ind):
        #self.im.seek(ind)
        fn = self.files[ind]
        if fn.endswith('.tif'):
            return tifffile.imread(fn).squeeze()
        else:
            im = Image.open(fn)
            ima = np.array(im.getdata())#.newbyteorder(self.endedness)
            return ima.reshape(self._slice_shape)
        #return self.data[:,:,ind]
        #return self.im[ind].asarray()

    def getSliceShape(self):
        return self._slice_shape
        #return self.im[0].shape[1:3]
        #return self.data.shape[:2]

    def getNumSlices(self):
        return len(self.files)

    def getEvents(self):
        return []

    def release(self):
        #self.im.close()
        pass

    def reloadData(self):
        pass
