#!/usr/bin/python

##################
# h5ToTiffSeries.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import tables
import Image
import sys
import os

if not (len(sys.argv) == 3):
    raise 'Usage: h5ToTiffSeries infile outdir'

inFile = sys.argv[1]
outDir = sys.argv[2]

h5f = tables.openFile(inFile)

nSlices = h5f.root.ImageData.shape[0]

if os.path.exists(outDir):
    raise 'Destination already exists'

os.makedirs(outDir)


for i in range(nSlices):
    Image.fromarray(h5f.root.ImageData[i, :,:].squeeze(), 'I;16').save(os.path.join(outDir, 'frame_%03d.tif'%i))    



