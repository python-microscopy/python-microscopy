import numpy as np
import scipy as sp
from scipy import ndimage
#from matplotlib import pyplot
import inspect
import os
from collections import OrderedDict


def makeFilter(filtFunc):
    '''wrapper function for different filters'''
    
    def ffcn(t, data, scale):
        out = {}
        for k, v in data.items():
            r_v = v[~np.isnan(v)]
            r_t = t[~np.isnan(v)]
            out[k] = filtFunc(np.interp(t, r_t, r_v), scale)
        return out
    
    return ffcn


FILTER_FUNCS = {
    'Gaussian': makeFilter(ndimage.gaussian_filter),
    'Uniform': makeFilter(ndimage.uniform_filter),
    'Median': makeFilter(ndimage.median_filter)
}


def extractAverageTrajectory(pipeline, clumpRadiusVar='error_x', clumpRadiusMultiplier=5.0,
                              timeWindow=25, filter='Gaussian', filterScale=10.0):
    #import PYME.Analysis.trackUtils as trackUtils
    import PYME.Analysis.points.DeClump as deClump
    from scipy.optimize import fmin
    #track beads through frames
    if clumpRadiusVar == '1.0':
        delta_x = 0 * pipeline['x'] + clumpRadiusMultiplier
    else:
        delta_x = clumpRadiusMultiplier * pipeline[clumpRadiusVar]
    
    t = pipeline['t'].astype('i')
    x = pipeline['x'].astype('f4')
    y = pipeline['y'].astype('f4')
    try:
        z = pipeline['z'].astype('f4')
    except:
        z = np.zeros_like(x)
        
    delta_x = delta_x.astype('f4')
    
    I = np.argsort(t)
    
    clumpIndex = np.zeros(len(x), dtype='i')
    clumpIndex[I] = deClump.findClumps(t[I], x[I], y[I], delta_x[I], timeWindow)
    #trackUtils.findTracks(pipeline, clumpRadiusVar,clumpRadiusMultiplier, timeWindow)
    
    #longTracks = pipeline['clumpSize'] > 50
    
    #x = x[longTracks].copy()
    #y = pipeline['y_raw'][longTracks].copy()
    #t = pipeline['t'][longTracks].copy() #.astype('i')
    #clumpIndex = pipeline['clumpIndex'][longTracks].copy()
    
    tMax = t.max()
    
    clumpIndices = list(set(clumpIndex))
    
    x_f = []
    y_f = []
    z_f = []
    clump_sizes = []
    
    t_f = np.arange(0, tMax + 1, dtype='i')
    
    #loop over all our clumps and extract trajectories
    for ci in clumpIndices:
        if ci > 0:
            clump_mask = (clumpIndex == ci)
            x_i = x[clump_mask]
            clump_size = len(x_i)
            
            if clump_size > 50:
                y_i = y[clump_mask]
                z_i = z[clump_mask]
                t_i = t[clump_mask].astype('i')
                
                x_i_f = np.NaN * np.ones_like(t_f)
                x_i_f[t_i] = x_i - x_i.mean()
                
                y_i_f = np.NaN * np.ones_like(t_f)
                y_i_f[t_i] = y_i - y_i.mean()

                z_i_f = np.NaN * np.ones_like(t_f)
                z_i_f[t_i] = z_i - z_i.mean()
                
                #clumps.append((x_i_f, y_i_f))
                x_f.append(x_i_f)
                y_f.append(y_i_f)
                z_f.append(z_i_f)
                clump_sizes.append(len(x_i))
    
    #re-order to start with the largest clump
    clumpOrder = np.argsort(clump_sizes)[::-1]
    x_f = np.array(x_f)[clumpOrder, :]
    y_f = np.array(y_f)[clumpOrder, :]
    z_f = np.array(z_f)[clumpOrder, :]
    
    def _mf(p, meas):
        '''calculate the offset between trajectories'''
        m_adj = meas + np.hstack([[0], p])[:, None]
        
        return np.nansum(np.nanvar(m_adj, axis=0))
    
    def _align(meas, tol=.1):
        n_iters = 0
        
        dm_old = 5e12
        dm = 4e12
        
        mm = np.nanmean(meas, 0)
        
        while ((dm_old - dm) > tol) and (n_iters < 50):
            dm_old = dm
            mm = np.nanmean(meas, 0)
            d = np.nanmean(meas - mm, 1)
            dm = sum(d ** 2)
            meas = meas - d[:, None]
            n_iters += 1
            print('%s, %s' % (n_iters, dm))
        
        mm = np.nanmean(meas, 0)
        print('Finished: %d, %f' %( n_iters, dm))
        return mm
    
    x_corr = _align(x_f)
    y_corr = _align(y_f)
    z_corr = _align(z_f)
    
    filtered_corr = FILTER_FUNCS[filter](t_f, {'x': x_corr, 'y': y_corr, 'z': z_corr}, filterScale)
    
    return t_f, filtered_corr, clumpIndex
