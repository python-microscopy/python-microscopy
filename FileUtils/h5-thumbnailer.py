#!/usr/bin/python

##################
# h5-thumbnailer.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/python

import sys
#import gnomevfs

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

size = (200,200)


def generateThumbnail(inputFile, thumbSize):
    global size
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

    return im.astype('uint8')

if __name__ == '__main__':
    import gnomevfs
    inputFile = gnomevfs.get_local_path_from_uri(sys.argv[1])
    outputFile = sys.argv[2]
    thumbSize = int(sys.argv[3])

    im = generateThumbnail(inputFile, thumbSize)

    Image.fromarray(im).resize(size).save(outputFile, 'PNG')
