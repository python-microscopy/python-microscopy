#!/usr/bin/python

##################
# h5rToTxt.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.Analysis.LMVis import inpFilt
import os
import sys


def convertFile(inFile, outFile):
    ds = inpFilt.h5rSource(inFile)

    nRecords = len(ds[ds.keys()[0]])

    of = open(outFile, 'w')

    of.write('#' + '\t'.join(['%s' % k for k in ds._keys]) + '\n')

    for row in zip(*[ds[k] for k in ds._keys]):
        of.write('\t'.join(['%e' % c for c in row]) + '\n')

    of.close()



if __name__ == '__main__':

    if (len(sys.argv) == 3):
        inFile = sys.argv[1]
        outFile = sys.argv[2]
    elif (len(sys.argv) == 2):
        inFile = sys.argv[1]
    else:
        raise 'Usage: h5rToTxt.py inDir [outFile]'

    if (len(sys.argv) == 2): #generate an output file name
        outFile = os.path.splitext(inFile)[0] + '.txt'

    if os.path.exists(outFile):
        print 'Output file already exists - please remove'
    else:
        convertFile(inFile, outFile)