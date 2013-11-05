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







