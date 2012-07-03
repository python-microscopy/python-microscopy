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
import numpy

def calcTrackVelocity(x, y, ci):
    #if not self.init:
    #    self.InitGL()
    #    self.init = 1

    I = ci.argsort()

    v = numpy.zeros(x.shape) #velocities
    w = numpy.zeros(x.shape) #weights

    x = x[I]
    y = y[I]
    ci = ci[I]

    dists = numpy.sqrt(numpy.diff(x)**2 + numpy.diff(y)**2)

    #now calculate a mask so that we only include distances from within the trace
    mask = numpy.diff(ci) < 1

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
    return (1./(4*numpy.pi*D*t))*numpy.exp(-r**2/(4*D*t))*2*numpy.pi*r


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







