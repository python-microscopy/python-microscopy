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
import numpy as np


def findConnected(i, t,x,y,delta_x, frameIndices, assigned, clumpNum, nFrames=5):
    #get the indices of all the points in the next n frames
    neighbour_inds = np.arange((i+1), min(frameIndices[int(t[i] + nFrames)], len(t)), dtype='int32')
    #print neighbour_inds
    
    #keep only those which haven't already been asigned to a clump
    neighbour_inds = neighbour_inds[assigned[neighbour_inds] == 0]

    #get the actual neighbour information
    #neighbours = fitResults[neighbour_inds]
    
    #calculate the square distances to the neighbours
    dis = (x[neighbour_inds] - x[i])**2 + (y[neighbour_inds] - y[i])**2

    #find the indices of those neighbours which are within twice the localisation precision
    #note that we're relying on the fact that the x and y localisaiton precisions are
    #typically practically the same
    sig_n = neighbour_inds[dis < (2*delta_x[i])**2]

    #add them to the clump
    assigned[sig_n] = clumpNum

    #and add their neighbours to the clump
    for n in sig_n:
        findConnected(n, t,x,y,delta_x, frameIndices, assigned, clumpNum, nFrames)


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
        raise RuntimeError('Was expecting to find a "FitResults" table')

    fr = h5f.root.FitResults[:]
    
    return deClump(fr)

def findClumps(t, x, y, delta_x):
    '''Finds clumps (or single particle trajectories) of data points in a series.
    fitRsults MUST be sorted in increasing time order.
    '''

    nRes = len(t)

    #there may be a different number of points in each frame; generate a lookup
    #table for frame numbers so we can index into our list of results to get
    #all the points within a certain range of frames
    frameIndices = (nRes + 2)*np.ones(t.max() + 10, 'int32')

    for t_i, i in zip(t, range(nRes)):
        frameIndices[:(t_i+1)] = np.minimum(frameIndices[:(t_i+1)], i)

    #print frameIndices
    
    #record whether a point has already been asigned to a clump
    assigned = np.zeros(nRes, 'int32')

    clumpNum = 1

    for i in range(nRes): #loop over all the points
        if assigned[i] == 0:
            #if a point hasn't already been assigned to a clump, start a new
            #clump from that point.
            assigned[i] = clumpNum
        
            #find all the points which are connected to this one
            findConnected(i, t,x,y,delta_x, frameIndices, assigned, clumpNum)

            #next pass will be a new clump
            clumpNum +=1

    return assigned


def coalesceClumps(fitResults, assigned):
    '''Agregates clumps to a single event'''
    NClumps = int(assigned.max())

    #work out what the data type for our declumped data should be
    dt = deClumpedDType(fitResults)

    filtered_res = []

    for i in range(1, NClumps+1):
            #coalesce the connected ponts into one
            vals = fitResults[assigned == i]

            vn = np.zeros(1,dtype=dt)
                        
            vn['tIndex'] = vals['tIndex'].min()

            vn['fitResults'], vn['fitError'] = weightedAverage(vals['fitResults'], vals['fitError'])

            vn['nFrames'] = len(vals)
            vn['ATotal'] = vals['fitResults']['A'].sum()

            filtered_res.append(vn)


    return np.hstack(filtered_res)


def deClump(fitResults):
    #select those points which fitted and have a reasonable fit error
    fitResults = fitResults[(fitResults['fitError']['x0'] > 0)*(fitResults['fitError']['x0'] < 60)]

    #make sure the results are sorted in frame order
    I = fitResults['tIndex'].argsort()
    fitResults = fitResults[I]

    t = fitResults['tIndex']
    x = fitResults['fitResults']['x0']
    y = fitResults['fitResults']['y0']
    delta_x = fitResults['fitError']['x0']

    assigned = findClumps(t, x, y, delta_x)

    return coalesceClumps(fitResults, assigned)

            




