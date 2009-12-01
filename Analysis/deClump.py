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
#import type
import numpy as np

assigned = []

def findConnected(i, fr, f_ind):
    global assigned
    #print i
    r = fr[i]

    #print r['tIndex']

    #print (i + 1), f_ind[int(r['tIndex'] + 1)]

    neigh = fr[(i + 1):f_ind[int(r['tIndex'] + 5)]]
    neigh = neigh[assigned[(i + 1):f_ind[int(r['tIndex'] + 5)]] == 0]

    #print len(neigh)

    dis = (neigh['fitResults']['x0'] - r['fitResults']['x0'])**2 + (neigh['fitResults']['y0'] - r['fitResults']['y0'])**2

    sig_n = i + 1 + np.where((dis < (2*r['fitError']['x0'])**2) * (neigh['fitError']['x0'] > 0))[0]
    assigned[sig_n] = 1

    neigh_neigh = []

    #print len(sig_n)

    for n in sig_n:
        #print n
        neigh_neigh += findConnected(n, fr, f_ind)

    #print len(neigh_neigh)

    return list(sig_n) + neigh_neigh

def deClumpedDType(arr):
    dt = arr.dtype.descr

    dt = [it for it in dt if not it[0] in ['slicesUsed', 'resultCode']]
    dt = dt + [('nFrames', '<i4'), ('ATotal', '<f4')]

    return dt

def weightedAverage(vals, errs):
    res = vals[0]
    eres = errs[0]

    for k in vals.dtype.names:
        erk2 = errs[k]**2
        vark = 1./(1./erk2).sum()
        res[k] = (vals[k]/erk2).sum()*vark
        eres[k] = np.sqrt(vark)

    return res, eres

def deClumpf(h5fFile):
    if type(h5fFile) == tables.file.File:
        h5f = h5fFile
    else:
        h5f = tables.openFile(h5fFile)

    if not 'FitResults' in dir(h5f.root):
        raise 'Was expecting to find a "FitResults" table'

    fr = h5f.root.FitResults[:]
    
    return deClump(fr)

def deClump(fr):
    global assigned
    #make sure the results are sorted in frame order
    I = fr['tIndex'].argsort()
    fr = fr[I]

    #generate a lookup table fro frame numbers
    f_ind = (len(fr['tIndex']) + 2)*np.ones(fr['tIndex'].max() + 10)

    for t, i in zip(fr['tIndex'], range(len(fr['tIndex']))):
        f_ind[:(t+1)] = np.minimum(f_ind[:(t+1)], i)

    filtered_res = []

    assigned = np.zeros(len(I))

    dt = deClumpedDType(fr)
    #print dt

    for i in range(len(I)):
        if not assigned[i]:
            assigned[i] = 1
            #r = fr[i]
            conn = findConnected(i, fr, f_ind)

            vals = fr[[i] + conn]

            vn = np.zeros(1,dtype=dt)

            #print vn
            #print vals
            
            vn['tIndex'] = vals['tIndex'].min()

            vn['fitResults'], vn['fitError'] = weightedAverage(vals['fitResults'], vals['fitError'])

            vn['nFrames'] = len(vals)
            vn['ATotal'] = vals['fitResults']['A'].sum()

            filtered_res.append(vn)


    return np.hstack(filtered_res)


            




