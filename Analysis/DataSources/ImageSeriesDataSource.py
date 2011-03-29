#!/usr/bin/python

##################
# TiffDataSource.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.ParallelTasks.relativeFiles import getFullFilename
from PYME.Acquire import MetaDataHandler
#from PYME.FileUtils import readTiff
import Image
import glob
import os
import numpy as np
#from PYME.misc import TiffImagePlugin #monkey patch PIL with improved tiff support from Priithon

#import numpy as np

#from PYME.misc import tifffile

class DataSource:
    moduleName = 'ImageSeriesDataSource'
    def __init__(self, filename, taskQueue=None):
        self.filename = getFullFilename(filename)#convert relative path to full path
        #self.data = readTiff.read3DTiff(self.filename)

        #use metadata for glob
        md = MetaDataHandler.SimpleMDHandler(self.filename)

        pattern = md.getEntry('SeriesPattern')

        self.files = glob.glob(os.path.join(os.path.split(self.filename)[0], pattern))

        self.files.sort()

        self.im0 = Image.open(self.files[0])

        #self.im.seek(0)

        #PIL's endedness support is subtly broken - try to fix it
        #NB this is untested for floating point tiffs
        self.endedness = 'LE'
        if self.im0.ifd.prefix =='MM':
            self.endedness = 'BE'

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
        im = Image.open(self.files[ind])
        ima = np.array(im.getdata()).newbyteorder(self.endedness)
        return ima.reshape((im.size[1], im.size[0]))
        #return self.data[:,:,ind]
        #return self.im[ind].asarray()

    def getSliceShape(self):
        return (self.im0.size[1], self.im0.size[0])
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
