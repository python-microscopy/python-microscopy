#!/usr/bin/python

##################
# kdfToFloat32Tiff.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import Image
import read_kdf

if not (len(sys.argv) == 3):
    raise RuntimeError('Usage: procStack inDir resDir')

inDir = sys.argv[1]
outDir = sys.argv[2]
