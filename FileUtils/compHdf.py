#!/usr/bin/python

##################
# compHdf.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import tables

if not (len(sys.argv) in [2,3]):
    raise RuntimeError('Usage: kdfToImage infile [outfile]')

inFile = sys.argv[1]

if len(sys.argv) == 2:
    outFile = inFile.split('.')[0] + '_c.h5'
else:
    outFile = sys.argv[2]

inF = tables.openFile(inFile)
outF = tables.openFile(outFile, 'w')
