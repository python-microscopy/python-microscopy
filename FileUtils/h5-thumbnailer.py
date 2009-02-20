#!/usr/bin/python

import sys
import gnomevfs

import tables

#from PYME.Analysis.LMVis import inpFilt
from scipy import minimum, maximum
import Image

from PYME.Analysis import MetaData
from PYME.Acquire import MetaDataHandler
from PYME.Analysis.DataSources import HDFDataSource

#import logging
#LOG_FILENAME = '/tmp/h5r-thumbnailer.log'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)


inputFile = gnomevfs.get_local_path_from_uri(sys.argv[1])
outputFile = sys.argv[2]
thumbSize = int(sys.argv[3])

#logging.debug('Input File: %s\n' % inputFile)
#logging.debug('Ouput File: %s\n' % outputFile)
#logging.debug('Thumb Size: %s\n' % thumbSize)

h5f = tables.openFile(inputFile)

dataSource = HDFDataSource.DataSource(inputFile, None)

md = MetaData.genMetaDataFromSourceAndMDH(dataSource, MetaDataHandler.HDFMDHandler(h5f))


xsize = h5f.root.ImageData.shape[1]
ysize = h5f.root.ImageData.shape[2]

if xsize > ysize:
    zoom = float(thumbSize)/xsize
else:
    zoom = float(thumbSize)/ysize

size = (int(xsize*zoom), int(ysize*zoom))

im = h5f.root.ImageData[min(md.EstimatedLaserOnFrameNo+10,(h5f.root.ImageData.shape[0]-1)) , :,:].astype('f')

im = im.T - min(md.Camera.ADOffset, im.min())

h5f.close()

im = maximum(minimum(1*(255*im)/im.max(), 255), 0)


Image.fromarray(im.astype('uint8')).resize(size).save(outputFile, 'PNG')
