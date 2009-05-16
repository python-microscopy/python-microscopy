#!/usr/bin/python

import logging
LOG_FILENAME = '/tmp/h5r-thumbnailer.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)

import sys
import gnomevfs

from PYME.Analysis.LMVis import inpFilt
from scipy import histogram2d, arange, minimum
import Image




inputFile = gnomevfs.get_local_path_from_uri(sys.argv[1])
outputFile = sys.argv[2]
thumbSize = int(sys.argv[3])

logging.debug('Input File: %s\n' % inputFile)
logging.debug('Ouput File: %s\n' % outputFile)
logging.debug('Thumb Size: %s\n' % thumbSize)

f1 = inpFilt.h5rSource(inputFile)
f2 = inpFilt.resultsFilter(f1, error_x=[0,30], A=[5, 1e5], sig=[100/2.35, 350/2.35])


xmax = f2['x'].max()
ymax = f2['y'].max()

if xmax > ymax:
    step = xmax/thumbSize
else:
    step = ymax/thumbSize

im, edx, edy = histogram2d(f2['x'], f2['y'], [arange(0, xmax, step), arange(0, ymax, step)]) 

f1.close()

im = minimum(2*(255*im)/im.max(), 255)


Image.fromarray(im.astype('uint8')).save(outputFile, 'PNG')
