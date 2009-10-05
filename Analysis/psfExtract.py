#!/usr/bin/python

##################
# psfExtract.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import scipy

def getCentre(x, y, z, corrPic):
    X, Y, Z = scipy.mgrid[x - 5: x + 5,y - 5: y + 5,z - 15: z + 15]

    imc = corrPic[x - 5: x + 5,y - 5: y + 5,z - 15: z + 15].real.copy()
    imc -= imc.min()
    imcs = imc.sum()

    xc = (imc*X).sum()/imcs
    yc = (imc*Y).sum()/imcs
    zc = (imc*Z).sum()/imcs

    print zc

    pz = corrPic[x,y,z - 15: z + 15].real
    zc2 = (scipy.mgrid[z - 15: z + 15]*pz).sum()/pz.sum()
    print zc2

    return (xc,yc,zc2)


def getAlignedSlice(x, y, z, corrPic, data):
    xc,yc,zc = getCentre(x, y, z, corrPic)

    dx = xc - x
    dy = yc - y
    dz = zc - z

    print dx
    print dy
    print dz

    dc = data[x - 17: x + 17,y - 17: y + 17,z - 32: z + 32]

    dcs = scipy.ndimage.shift(dc, [dx,dy,dz], mode='nearest')

    return dcs[2:-2, 2:-2, 2:-2]
