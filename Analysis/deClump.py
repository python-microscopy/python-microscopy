#!/usr/bin/python

##################
# deClump.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import tables
import type
import numpy as np

assigned = []

def findConnected(i, fr, f_ind):
    global assigned
    r = fr[i]

    neigh = fr[(i + 1):f_ind(r['tIndex'] + 5)]
    neigh = neigh[assigned[(i + 1):f_ind(r['tIndex'] + 5)] == 0]
    dis = (neigh['fitResults']['x0'] - r['fitResults']['x0'])**2 + (neigh['fitResults']['y0'] - r['fitResults']['y0'])**2

    sig_n = i + 1 + np.where((dis < (2*r['fitError']['x0'])**2) * (neigh['fitError']['x0'] > 0))
    assigned[sig_n] = 1

    neigh_neigh = []

    for n in sig_n:
        neigh_neigh += findConnected(n, fr, f_ind)

    return list(sig_n) + neigh_neigh



def deClump(h5fFile):
    global assigned
    if type(h5fFile) == tables.file.File:
        h5f = h5fFile
    else:
        h5f = tables.openFile(h5fFile)

    if not 'FitResults' in dir(h5f.root):
        raise 'Was expecting to find a "FitResults" table'

    fr = h5f.root.FitResults[:]

    #make sure the results are sorted in frame order
    I = fr['tIndex'].argsort()
    fr = fr[I]

    #generate a lookup table fro frame numbers
    f_ind = (len(fr['tIndex']) + 1)*ones(fr['tIndex'].max() + 1)

    for t, i in zip(fr['tIndex'], range(len(fr['tIndex']))):
        f_ind[:(t+1)] = minimum(f_ind[:(t+1)], i)

    filtered_res = []

    assigned = zeros(len(I))

    for i in range(len(I)):
        if not assigned[i]:
            assigned[i] = 1
            #r = fr[i]
            conn = findConnected(i, fr, f_ind)




