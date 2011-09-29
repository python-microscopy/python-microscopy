#!/usr/bin/python
##################
# test_lut.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from pylab import *

import lut

lut1 = (255*cm.gray(linspace(0,1,256))[:,:3].T).astype('uint8')
print lut1.shape

d = (100*rand(5,5)).astype('uint16')
o = np.zeros((5,5,3), 'uint8')

lut.applyLUT(d, .01, 0,lut1, o)

print o
print lut1.T[((d + 0)*.01*256).astype('i')]

