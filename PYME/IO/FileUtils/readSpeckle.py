# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:46:29 2015

@author: david
"""
import numpy as np
from PYME.IO import unifiedIO

def readSpeckles(filename):
    with unifiedIO.openFile(filename, 'r') as f:
        
        speckles = []
        currentSpeckle = None    
        
        for l in f.readlines():
            if l.startswith('#%start speckle'):
                currentSpeckle = []
                speckles.append(currentSpeckle)
            elif l.startswith('#'):
                #comment
                pass
            else:
                currentSpeckle.append([float(val) for val in l.split('\t')])
                
        return [np.array(s) for s in speckles]


def gen_traces_from_speckles(speckles, leadFrames=10, followFrames=50, seriesLength=1000000, clipRegion=[0,0,512,512]):
    """ Generate pseudo particle trajectories from speckle data.

     These trajectories look like they were generated using the normal particle tracking methods, but are centered
     at the detected speckle position. They extend before the start of the speckle trace (to establish a baseline), and
     after the end of the speckle trace (to look at the diffusion of the membrane label away from the point of fusion.

    Parameters
    ----------
    speckles : list
        list of [x,y,t]xN arrays containing the trace for each speckle
    leadFrames : int
        Number of frames to extract before the start of the speckle for establishing the baseline
    followFrames : int
        Number of frames to extract after the end of the speckle to follow the diffusion of the fused dyes.
    seriesLength : int
        the length of the image series - traces will not extend beyond the end of the series.

    Returns
    -------
    a numpy recarray containing x, y, t, and clumpIndex columns
    """

    point_dtype = np.dtype([('x_pixels', 'f4'), ('y_pixels', 'f4'), ('t', 'i4'), ('clumpIndex', 'i4')])

    extended_speckles = []
    
    x0, y0, x1, y1 = clipRegion

    for i, speck in enumerate(speckles):
        y_, x_, t_ = speck.T
        trace_start = int(max(t_[0] - leadFrames, 0))
        trace_end = int(min(t_[-1] + followFrames, seriesLength))

        xm, ym = x_.mean(), y_.mean()
        
        if (xm >= x0) and (ym >= y0) and (xm < x1) and (ym < y1):
            for j in range(trace_start, trace_end):
                s_ = np.zeros(1, dtype=point_dtype)
                s_['x_pixels'] = xm
                s_['y_pixels'] = ym
                s_['t'] = j
                s_['clumpIndex'] = i
                extended_speckles.append(s_)

    sp = np.hstack(extended_speckles)

    I = np.argsort(sp['t'])
    return sp[I]

