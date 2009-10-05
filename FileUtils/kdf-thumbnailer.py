#!/usr/bin/python

##################
# kdf-thumbnailer.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/python

import sys
import gnomevfs

#import tables

#from PYME.Analysis.LMVis import inpFilt
from scipy import minimum, maximum
import Image

from PYME import cSMI

#from PYME.Analysis import MetaData
#from PYME.Acquire import MetaDataHandler
#from PYME.Analysis.DataSources import HDFDataSource

#import logging
#LOG_FILENAME = '/tmp/h5r-thumbnailer.log'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)

#print sys.argv
inputFile = gnomevfs.get_local_path_from_uri(sys.argv[1])
outputFile = sys.argv[2]
thumbSize = int(sys.argv[3])

#logging.debug('Input File: %s\n' % inputFile)
#logging.debug('Ouput File: %s\n' % outputFile)
#logging.debug('Thumb Size: %s\n' % thumbSize)

im = cSMI.CDataStack_AsArray(cSMI.CDataStack(inputFile.encode()), 0).mean(2).squeeze()


xsize = im.shape[0]
ysize = im.shape[1]

if xsize > ysize:
    zoom = float(thumbSize)/xsize
else:
    zoom = float(thumbSize)/ysize

size = (int(xsize*zoom), int(ysize*zoom))

im = im - im.min()

im = maximum(minimum(1*(255*im)/im.max(), 255), 0)


Image.fromarray(im.astype('uint8')).resize(size).save(outputFile, 'PNG')
