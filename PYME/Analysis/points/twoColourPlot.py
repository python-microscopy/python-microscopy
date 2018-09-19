#!/usr/bin/python

##################
# twoColour.py
#
# Copyright David Baddeley, 2009
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
##################

#from pylab import *
import matplotlib.pyplot as plt
import numpy as np

def PlotShiftField(dx, dy, spx, spy):
    plt.figure()
    #subplot(221)
    plt.axes([.05, .7, .4, .25])
    plt.imshow(dx.T)
    plt.axis('image')
    plt.colorbar()
    plt.axes([.55, .7, .4, .25])
    plt.imshow(dy.T)
    plt.axis('image')
    plt.colorbar()

    plt.axes([.05, .05, .9, .6])
    xin, yin = np.meshgrid(np.arange(0, 512*70, 500), np.arange(0, 256*70, 1000))
    xin = xin.ravel()
    yin = yin.ravel()
    plt.quiver(xin, yin, spx.ev(xin, yin), spy.ev(xin, yin), scale=1e4)
    plt.axis('image')

def PlotShiftResiduals(x, y, dx, dy, spx, spy, z=0):
    plt.figure()
    if 'ZDEPSHIFT' in dir(spx):
        dx1 = spx.ev(x,y, z)
        dy1 = spy.ev(x,y, z)
    else:
        dx1 = spx.ev(x,y)
        dy1 = spy.ev(x,y)
    
    print((dx1.shape, dx.shape))
    
    dist = np.sqrt((dx1 - dx)**2 + (dy1 - dy)**2)
    plt.quiver(x, y, dx1 - dx, dy1 - dy, dist, scale=2e2, clim=(0, dist.mean()*2))
    plt.colorbar()
    plt.axis('image')
    plt.title('Residuals')

def PlotShiftResidualsS(x, y, dx, dy, spx, spy):
    #figure()
    dx1 = spx.ev(x,y)
    dy1 = spy.ev(x,y)
    
    print((dx1.shape, dx.shape))
    
    dist = np.sqrt((dx1 - dx)**2 + (dy1 - dy)**2)
    plt.quiver(x, y, dx1 - dx, dy1 - dy, dist, scale=2e2, clim=(0, dist.mean()*2))
    plt.colorbar()
    plt.axis('image')
    plt.title('Residuals')

def PlotShiftField2(spx, spy, shape=[512, 256], voxelsize=(70., 70., 200.)):
    xi, yi = np.meshgrid(np.arange(0, shape[0]*voxelsize[0], 100), np.arange(0, shape[1]*voxelsize[1], 100))
    xin = xi.ravel();yin = yi.ravel()
    dx = spx.ev(xin[:], yin[:]).reshape(xi.shape)
    dy = spy.ev(xin[:], yin[:]).reshape(xi.shape)
    plt.figure()
    plt.subplot(131)
    #axes([.05, .7, .4, .25])
    plt.imshow(dx[::-1, :], extent=[0,shape[0], 0, shape[1]])
    plt.axis('image')
    plt.colorbar()
    #axes([.55, .7, .4, .25])
    plt.subplot(132)
    im = plt.imshow(dy[::-1, :], extent=[0,shape[0], 0, shape[1]])
    plt.axis('image')
    plt.colorbar()

    #axes([.05, .05, .9, .6])
    xi, yi = np.meshgrid(np.arange(0, shape[0]*voxelsize[0], 2200), np.arange(0, shape[1]*voxelsize[1], 2200))
    xin = xi.ravel();yin = yi.ravel()
    dx = spx.ev(xin[:], yin[:]).reshape(xi.shape)
    dy = spy.ev(xin[:], yin[:]).reshape(xi.shape)
    plt.subplot(133)

    plt.quiver(xin/voxelsize[0], yin/voxelsize[0], spx.ev(xin, yin), spy.ev(xin, yin), scale=1e4)

    plt.axis('image')
    plt.xlim(0, shape[0])
    plt.ylim(0, shape[1])

    #colorbar(im)
