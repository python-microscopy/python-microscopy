#!/usr/bin/python

##################
# sf-thumbnailer.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/python

#import logging
#LOG_FILENAME = '/tmp/sf-thumbnailer.log'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)

import sys
import gnomevfs

import cPickle

import matplotlib
matplotlib.use('Agg')

from pylab import *

dpi = 100.


inputFile = gnomevfs.get_local_path_from_uri(sys.argv[1])
outputFile = sys.argv[2]
thumbSize = int(sys.argv[3])

#logging.debug('Input File: %s\n' % inputFile)
#logging.debug('Ouput File: %s\n' % outputFile)
#logging.debug('Thumb Size: %s\n' % thumbSize)

#def generateThumbnail(inputFile, thumbsize):
fid = open(inputFile)
spx, spy = cPickle.load(fid)
fid.close()

f = figure(figsize=(thumbSize/dpi, 0.5*thumbSize/dpi))

axes([0, 0, 1, 1])
xin, yin = meshgrid(arange(0, 512*70, 4000), arange(0, 256*70, 4000))
xin = xin.ravel()
yin = yin.ravel()
quiver(xin, yin, spx.ev(xin, yin), spy.ev(xin, yin), scale=2e3)
xticks([])
yticks([])
axis('image')

f.savefig(outputFile, dpi=dpi, format='png')

