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
import mpld3
import pandas as pd
import os

class FeaturePlot(object):
    def __init__(self, clump):
        self.clump = clump
    
    def __getitem__(self, key):
        if not self.clump.dtypes[key] in ['float32', 'float64']:
            return ''
            
        data = self.clump[key]
            
        plt.ioff()
        f = plt.figure(figsize=(6,1))        
        
        plt.plot(data)
        
        plt.tight_layout()
        
        plt.ion()
        
        return mpld3.fig_to_html(f)
        
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
        
        self.index = pipeline['clumpIndex'] == clumpID
        self.nEvents = self.index.sum()
        self.enabled = True
        self.cache = {}
        
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
            self._dtypes = self.pipeline.dtypes
        return self._dtypes

    def __getitem__(self, key):
        if not key in self.keys():
            raise RuntimeError('Key not defined')
        
        if not key in self.cache.keys():
            self.cache[key] = self.pipeline[key][self.index]
            
        return self.cache[key]
        
    def save(self, filename):
        d = {}
        d.update(self)
        
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
            from PYME.Analysis.DistHist import msdHistogram
            from PYME.Analysis._fithelpers import FitModel
            
            x = self['x']
            y = self['y']
            t = self['t']
            
            dt = self.pipeline.mdh.getOrDefault('Camera.CycleTime', 1.0)
            
            nT = (t.max() - t.min())/2
            
            h = msdHistogram(x, y, t, nT)
            t_ = dt*np.arange(len(h))
            
            res = FitModel(powerMod2D, [h[-1]/t_[-1], 1.], h[1:], t_[1:])
            D, alpha = res[0]
            
            self._msdinfo = {'t':t_, 'msd':h, 'D':D, 'alpha':alpha}
        
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
        
        return mpld3.fig_to_html(f)
        
    #@property
    
            
        
        
        
        
class ClumpManager(object):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def __getitem__(self, key):
        return Track(self.pipeline, key)
    
    @property    
    def all(self):
        ids = list(set(self.pipeline['clumpIndex']))
        
        return [self.__getitem__(id) for id in ids]
            
        
    

def findTracks(pipeline, rad_var='error_x', multiplier='2.0', nFrames=20):
    import PYME.Analysis.DeClump.deClump as deClump
    
    if rad_var == '1.0':
        delta_x = 0*pipeline.mapping['x'] + multiplier
    else:
        delta_x = multiplier*pipeline.mapping[rad_var]

    clumpIndices = deClump.findClumps(pipeline.mapping['t'].astype('i'), pipeline.mapping['x'].astype('f4'), pipeline.mapping['y'].astype('f4'), delta_x.astype('f4'), nFrames)
    numPerClump, b = np.histogram(clumpIndices, np.arange(clumpIndices.max() + 1.5) + .5)

    trackVelocities = calcTrackVelocity(pipeline.mapping['x'], pipeline.mapping['y'], clumpIndices)
    #print b

    pipeline.selectedDataSource.clumpIndices = -1*np.ones(len(pipeline.selectedDataSource['x']))
    pipeline.selectedDataSource.clumpIndices[pipeline.filter.Index] = clumpIndices

    pipeline.selectedDataSource.clumpSizes = np.zeros(pipeline.selectedDataSource.clumpIndices.shape)
    pipeline.selectedDataSource.clumpSizes[pipeline.filter.Index] = numPerClump[clumpIndices - 1]

    pipeline.selectedDataSource.trackVelocities = np.zeros(pipeline.selectedDataSource.clumpIndices.shape)
    pipeline.selectedDataSource.trackVelocities[pipeline.filter.Index] = trackVelocities

    pipeline.selectedDataSource.setMapping('clumpIndex', 'clumpIndices')
    pipeline.selectedDataSource.setMapping('clumpSize', 'clumpSizes')
    pipeline.selectedDataSource.setMapping('trackVelocity', 'trackVelocities')
    
    pipeline.clumps = ClumpManager(pipeline)


def calcTrackVelocity(x, y, ci):
    #if not self.init:
    #    self.InitGL()
    #    self.init = 1

    I = ci.argsort()

    v = np.zeros(x.shape) #velocities
    w = np.zeros(x.shape) #weights

    x = x[I]
    y = y[I]
    ci = ci[I]

    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    #now calculate a mask so that we only include distances from within the trace
    mask = np.diff(ci) < 1

    #a points velocity is the average of the steps in each direction
    v[1:] += dists*mask
    v[:-1] += dists*mask

    #calulate weights to divide velocities by (end points only have one contributing step)
    w[1:] += mask
    w[:-1] += mask

    #leave velocities for length 1 traces as zero (rather than 0/0
    v[w > 0] = v[w > 0]/w[w > 0]

    #reorder
    v[I] = v

    return v

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
    from pylab import *
    from PYME.Analysis._fithelpers import *
    
    N = len(velocities)

    figure()

    h, b, p = hist(velocities, 200)
    
    x = b[1:]
    dx = x[1] - x[0]

    r = FitModel(jumpDistModel, startParams, h, x, N, dT, dx)

    plot(x, jumpDistModel(r[0], x, N, dT, dx), lw=2)

    fT = 0

    for i in range(len(startParams)/2):
      D, f = r[0][(2*i):(2*i +2)]

      plot(x, N*f*jumpDistProb(x, D, dT)*dx, '--', lw=2, label='D = %3.2g, f=%3.2f' % (D, f))
      fT += f

    D = r[0][-1]
    plot(x, N*(1 - fT)*jumpDistProb(x, D, dT)*dx, '--', lw=2, label='D = %3.2g, f=%3.2f' % (D, (1-fT)))

    xlabel('Jump Size [nm]')
    ylabel('Frequency')
    legend()

    return r







