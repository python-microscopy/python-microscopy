#!/usr/bin/python

##################
# IJLut.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#! /usr/bin/python
import pylab
import numpy
import sys
import os

if __name__ == "__main__":

    if not len(sys.argv) == 2:
        raise 'expected a directory to save the luts to'

    outDir = sys.argv[1]

    cmapnames = pylab.cm.cmapnames

    for cmn in cmapnames:
        c = (255*pylab.cm.__dict__[cmn](numpy.arange(256)))[:,:3].astype('uint8')

        f = open(os.path.join(outDir, '%s.lut' % cmn), 'wb')
        c.T.tofile(f)
        f.close()
