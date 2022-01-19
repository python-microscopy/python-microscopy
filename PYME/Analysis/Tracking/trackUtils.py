#!/usr/bin/python

###############
# trackUtils.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################
import numpy as np
import matplotlib.pyplot as plt

import warnings

try:
    import mpld3
    if warnings.filters[0] == ('always', None, DeprecationWarning, None, 0):
        #mpld3 has messed with warnings - undo
        warnings.filters.pop(0)
except ImportError:
    warnings.warn('Could not import mpld3, track plotting will not work')

import pandas as pd
import os

class FeaturePlot(object):
    def __init__(self, clump):
        self.clump = clump
    
    def __getitem__(self, key):
        if not self.clump.dtypes[key] in ['float32', 'float64']:
            return ''
            
        data = self.clump[key]

        if len(data.shape) >= 2:
            # FIXME - handle this better. Potentially a kymograph for ndim==2?
            # FIXME - warn???
            return ''
            
        plt.ioff()
        f = plt.figure(figsize=(6,2))

        if 't' in self.clump.keys():        
            plt.plot(self.clump['t'], data)
        else:
            plt.plot(data)
        
        plt.tight_layout()
        
        plt.ion()

        ret = mpld3.fig_to_html(f)
        plt.close(f)
        return ret
        
class FeatureMean(object):
    def __init__(self, clump):
        self.clump = clump
    
    def __getitem__(self, key):
        if self.clump.dtypes[key] == 'object':
            return 'N/A'
        else:
            #print key
            data = self.clump[key]
            return data.mean()
            
class FeatureStd(object):
    def __init__(self, clump):
        self.clump = clump
    
    def __getitem__(self, key):
        if self.clump.dtypes[key] == 'object':
            return 'N/A'
        else:
            data = self.clump[key]
            return data.std()

class Clump(object):
    def __init__(self, pipeline, clumpID):
        self.pipeline = pipeline
        self.clumpID = clumpID
        
        self.index = np.array(pipeline['clumpIndex'] == clumpID, dtype=np.bool)
        self.nEvents = self.index.sum()
        self.enabled = True
        self.cache = {}
        
        self.image = None
        
        self.featureplot = FeaturePlot(self)
        self.featuremean = FeatureMean(self)
        self.featurestd = FeatureStd(self)
            
    
    def keys(self):
        return self.pipeline.keys()
        
    @property
    def varnames(self):
        return self.keys()
    
    @property
    def dtypes(self):
        if not '_dtypes' in dir(self):
            if 'dtypes' in dir(self.pipeline):
                self._dtypes = self.pipeline.dtypes
            else:
                self._dtypes = {k:self.pipeline[k].dtype for k in self.pipeline.keys()}
        return self._dtypes

    def __getitem__(self, key):
        if not key in self.keys():
            raise KeyError('Key not defined: % s' % key)
        
        if not key in self.cache.keys():
            self.cache[key] = np.array(self.pipeline[key][self.index])
            
        return self.cache[key]
        
    def save(self, filename, keys = None):
        d = {}        
        if keys is None:
            d.update(self)
        else:
            for k in keys:
                if k in self.keys():
                    d[k] = self[k]
        
        df = pd.DataFrame(d)

        ext = os.path.splitext(filename)[-1]        
        
        if ext == '.csv': 
            df.to_csv(filename)
        elif ext in ['.xls', '.xlsx']:
            df.to_excel(filename)
        elif ext == '.hdf':
            df.to_hdf(filename, 'Results')
        elif ext == '.pik':
            df.to_pickle(filename)
        else:
            raise RuntimeError('Unkonwn extension: %s' %ext)
    

def powerMod2D(p,t):
    D, alpha = p
    return 4*D*t**alpha #factor 4 for 2D (6 for 3D)

def powerMod3D(p,t):
    D, alpha = p
    return 6*D*t**alpha #factor 4 for 2D (6 for 3D)


        
class Track(Clump):
    @property
    def distance(self):
        if not '_distance' in dir(self):
            #cache result
            self._distance = np.sqrt((self['x'][-1] - self['x'][0])**2 + (self['y'][-1] - self['y'][0])**2)
        
        return self._distance
        
    @property
    def trajectory(self):
        plt.ioff()
        f = plt.figure(figsize=(4,3))
        
        x = self['x']
        y = self['y']
        plt.plot(x - x.mean(), y-y.mean())
        
        plt.xlabel('x [nm]')
        plt.ylabel('y [nm]')
        plt.title('Particle Trajectory')

        plt.axis('equal')
        plt.tight_layout(pad=2)
        
        plt.ion()
        
        return mpld3.fig_to_html(f)
        
    @property
    def msdinfo(self):
        if not '_msdinfo' in dir(self):
            from PYME.Analysis.points.DistHist import msdHistogram
            from PYME.Analysis._fithelpers import FitModel
            
            x = self['x']
            y = self['y']
            t = self['t']
            
            dt = self.pipeline.mdh.getOrDefault('Camera.CycleTime', 1.0)
            
            nT = int((t.max() - t.min())/2)
            
            h = msdHistogram(x, y, t, nT)
            t_ = dt*np.arange(len(h))
            
            res = FitModel(powerMod2D, [h[-1]/t_[-1], 1.], h[1:], t_[1:])
            D, alpha = res[0]
            
            
            #calculate the diffusion coefficient for normal (not anomolous diffusion)
            #restrict to the first 100 bins to try and minimise effects of drift
            #fit an offset to take localization precision into account
            D_ = float(np.linalg.lstsq(np.vstack([t_[1:100], np.ones_like(t_[1:100])]).T, h[1:100])[0][0]/4)
            
            self._msdinfo = {'t':t_, 'msd':h, 'D':D, 'alpha':alpha, 'Dnormal' : D_}
        
        return self._msdinfo
        
    @property
    def msdplot(self):
        plt.ioff()
        f = plt.figure(figsize=(4,3))
        
        msdi = self.msdinfo
        t = msdi['t']
        plt.plot(t[1:], msdi['msd'][1:])
        plt.plot(t, powerMod2D([msdi['D'], msdi['alpha']], t))
        
        plt.xlabel('delta t')
        plt.ylabel('MSD [nm^2]')
        plt.title(r'MSD - D = %(D)3.2f, alpha = %(alpha)3.1f' % msdi)
        
        plt.tight_layout(pad=2)
        
        plt.ion()

        ret = mpld3.fig_to_html(f)
        plt.close(f)
        return ret
        
    @property
    def movieplot(self):
        plt.ioff()
        f = plt.figure(figsize=(12,3))
        
        #msdi = self.msdinfo
        #t = msdi['t']
        #plt.plot(t[1:], msdi['msd'][1:])
        #plt.plot(t, powerMod2D([msdi['D'], msdi['alpha']], t))
        
        xp, yp = self['centroid'][0]
        
        xp = int(np.round(xp))
        yp = int(np.round(yp))
        
        if not self.image is None:
            for i in range(self.nEvents):
                plt.subplot(1, self.nEvents, i+1)
                img = self.image[(xp - 10):(xp + 10), (yp - 10):(yp + 10), self['t'][i]]
                plt.imshow(img, interpolation ='nearest', cmap=plt.cm.gray)
                plt.xticks([])
                plt.yticks([])    
        else:
            for i in range(self.nEvents):
                plt.subplot(1, self.nEvents, i+1)
                plt.imshow(self['intensity_image'][i], interpolation ='nearest', cmap=plt.cm.gray)
                plt.xticks([])
                plt.yticks([])
        
        #plt.xlabel('delta t')
        #plt.ylabel('MSD [nm^2]')
        #plt.title(r'MSD - D = %(D)3.2f, alpha = %(alpha)3.1f' % msdi)
        
        plt.tight_layout(pad=1)
        
        plt.ion()

        ret = mpld3.fig_to_html(f)
        plt.close(f)
        return ret
        
    #@property
    
            
        
        
        
        
class ClumpManager(object):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def __getitem__(self, key):
        return Track(self.pipeline, key)
    
    @property    
    def all(self):
        ids = list(set(self.pipeline['clumpIndex']))
        ids.sort()
        
        return [self.__getitem__(id) for id in ids]
            

class TrackManager(object):
    def __init__(self, trackList):
        self.trackList = trackList
        
    #def __getitem__(self, key):
    #    return Track(self.pipeline, key)
    
    @property    
    def all(self):
        return self.trackList 
        
    @property
    def filtered(self):
        return [t for t in self.trackList if t.enabled]
        
    
    

def findTracks(pipeline, rad_var='error_x', multiplier='2.0', nFrames=20):
    import PYME.Analysis.points.DeClump as deClump
    import warnings
    
    warnings.warn('deprecated, use findTracks2 instead', DeprecationWarning)
    
    if rad_var == '1.0':
        delta_x = 0*pipeline['x'] + multiplier
    else:
        delta_x = multiplier*pipeline[rad_var]
        
    t = pipeline['t'].astype('i')
    x = pipeline['x'].astype('f4')
    y = pipeline['y'].astype('f4')
    delta_x = delta_x.astype('f4')
    
    I = np.argsort(t)

    clumpIndices = np.zeros(len(x), dtype='i')
    clumpIndices[I] = deClump.findClumps(t[I], x[I], y[I], delta_x[I], nFrames)
    
    numPerClump, b = np.histogram(clumpIndices, np.arange(clumpIndices.max() + 1.5) + .5)

    trackVelocities = 0*x
    trackVelocities[I] = calcTrackVelocity(x[I], y[I], clumpIndices[I], t.astype('f')[I])
    #print b

    pipeline.addColumn('clumpIndex', clumpIndices, -1)
    pipeline.addColumn('clumpSize', numPerClump[clumpIndices - 1])
    pipeline.addColumn('trackVelocity', trackVelocities)
    
    pipeline.clumps = ClumpManager(pipeline)


def findTracks2(datasource, rad_var='error_x', multiplier='2.0', nFrames=20, minClumpSize=0):
    import PYME.Analysis.points.DeClump as deClump
    from PYME.IO import tabular
    
    with_clumps = tabular.MappingFilter(datasource)
    
    if rad_var == '1.0':
        delta_x = 0 * datasource['x'] + multiplier
    else:
        delta_x = multiplier * datasource[rad_var]
    
    t = datasource['t'].astype('i')
    x = datasource['x'].astype('f4')
    y = datasource['y'].astype('f4')
    delta_x = delta_x.astype('f4')
    
    I = np.argsort(t)
    
    clumpIndices = np.zeros(len(x), dtype='i')
    clumpIndices[I] = deClump.findClumps(t[I], x[I], y[I], delta_x[I], nFrames)
    
    numPerClump, b = np.histogram(clumpIndices, np.arange(clumpIndices.max() + 1.5) + .5)
    
    v, edges = calcTrackVelocity(x[I], y[I], clumpIndices[I], t.astype('f')[I])
    trackVelocities = 0 * x
    trackVelocities[I] = v

    clumpEdges = 0 * x # or should we inforce int?
    clumpEdges[I] = edges
    #print b
    
    with_clumps.addColumn('clumpIndex', clumpIndices)
    with_clumps.addColumn('clumpSize', numPerClump[clumpIndices - 1])
    with_clumps.addColumn('trackVelocity', trackVelocities)
    with_clumps.addColumn('clumpEdge', clumpEdges)
    
    if minClumpSize > 0:
        filt = tabular.ResultsFilter(with_clumps, clumpSize=[minClumpSize, 1e6])
    else:
        filt = with_clumps

    try:
        filt.mdh = datasource.mdh
    except AttributeError:
        pass
    
    return with_clumps, ClumpManager(filt)


def calcTrackVelocity(x, y, ci, t):
    #if not self.init:
    #    self.InitGL()
    #    self.init = 1
    
    #first sort by time
    #I = t.argsort()
    #x = x[I]
    #y = y[I]
    #ci = ci[I]

    #sort by both time and clump

    ind = ci + .5*t/t.max()

    #I = ci.argsort()
    I = ind.argsort()

    v = np.zeros(x.shape) #velocities
    w = np.zeros(x.shape) #weights
    edges = np.zeros(x.shape) # edge mask
    
    x = x[I]
    y = y[I]
    ci = ci[I]
    t = t[I]

    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    #now calculate a mask so that we only include distances from within the trace
    mask = 1.0*(np.diff(ci) < .5)

    #a points velocity is the average of the steps in each direction
    v[1:] += dists*mask
    v[:-1] += dists*mask

    #calulate weights to divide velocities by (end points only have one contributing step)
    w[1:] += mask
    w[:-1] += mask

    #leave velocities for length 1 traces as zero (rather than 0/0
    v[w > 0] = v[w > 0]/w[w > 0]

    #reorder
    v[I] = (1.0*v.copy())

    #edges
    edges[1:] += (1.0-mask) # here we are looking for positive steps (i.e. np.diff(ci) >= .5)
    edges[:-1] += (1.0-mask)
    edges[0] = 1.0 # manually add very first and very last as obvious edges
    edges[-1] = 1.0

    #reorder
    edges[I] = (1.0*(edges > 0.5)) # clumps of size 1 will have 'edgeness' values of 2 from the procedure above
    
    return (v, edges)

def jumpDistProb(r, D, t):
    return (1./(4*np.pi*D*t))*np.exp(-r**2/(4*D*t))*2*np.pi*r


def jumpDistModel(p, r, N, t, dx):
    fT = 0
    res = 0

    for i in range(len(p)/2):
      D, f = p[(2*i):(2*i +2)]

      res += f*jumpDistProb(r, D, t)
      fT += f

    res += (1 - fT)*jumpDistProb(r, p[-1], t)

    return N*res*dx

def FitJumpSizeDist(velocities, startParams, dT):
    from PYME.Analysis._fithelpers import FitModel
    
    N = len(velocities)

    plt.figure()

    h, b, _ = plt.hist(velocities, 200)
    
    x = b[1:]
    dx = x[1] - x[0]

    r = FitModel(jumpDistModel, startParams, h, x, N, dT, dx)

    plt.plot(x, jumpDistModel(r[0], x, N, dT, dx), lw=2)

    fT = 0

    for i in range(len(startParams)/2):
      D, f = r[0][(2*i):(2*i +2)]

      plt.plot(x, N*f*jumpDistProb(x, D, dT)*dx, '--', lw=2, label='D = %3.2g, f=%3.2f' % (D, f))
      fT += f

    D = r[0][-1]
    plt.plot(x, N*(1 - fT)*jumpDistProb(x, D, dT)*dx, '--', lw=2, label='D = %3.2g, f=%3.2f' % (D, (1-fT)))

    plt.xlabel('Jump Size [nm]')
    plt.ylabel('Frequency')
    plt.legend()

    return r







