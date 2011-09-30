#!/usr/bin/python

##################
# __init__.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from lut import *

def applyLUT(seg, gain, offset, lut, ima):
    if seg.dtype == 'uint8':
        applyLUTu8(seg, gain, offset, lut, ima)
    elif seg.dtype == 'uint16':
        applyLUTu16(seg, gain, offset, lut, ima)
    else:
        applyLUTf((1.0*seg).astype('f'), gain, offset, lut, ima)
