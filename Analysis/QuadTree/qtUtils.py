#!/usr/bin/python

##################
# qtUtils.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import pointQT
import MetaData

def compareOriginalWithRec(ofd, qt, radius = 250, md = MetaData.TIRFDefault):
    I_origs = []
    for p in ofd:
        I_origs.append(ofd.filteredData[p.x,p.y])

    N_points = []
    for p in ofd:
    	Ns.append(len(pointQT.getInRadius(qt, 1e3*p.x*md.voxelsize.x, 1e3*p.y*md.voxelsize.y, radius)))

    return (I_origs, N_points)
