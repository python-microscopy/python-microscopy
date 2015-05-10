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

from pylab import *

def PlotShiftField(dx, dy, spx, spy):
    figure()
    #subplot(221)
    axes([.05, .7, .4, .25])
    imshow(dx.T)
    axis('image')
    colorbar()
    axes([.55, .7, .4, .25])
    imshow(dy.T)
    axis('image')
    colorbar()

    axes([.05, .05, .9, .6])
    xin, yin = meshgrid(arange(0, 512*70, 500), arange(0, 256*70, 1000))
    xin = xin.ravel()
    yin = yin.ravel()
    quiver(xin, yin, spx.ev(xin, yin), spy.ev(xin, yin), scale=1e4)
    axis('image')

def PlotShiftResiduals(x, y, dx, dy, spx, spy, z=0):
    figure()
    if 'ZDEPSHIFT' in dir(spx):
        dx1 = spx.ev(x,y, z)
        dy1 = spy.ev(x,y, z)
    else:
        dx1 = spx.ev(x,y)
        dy1 = spy.ev(x,y)
    
    print((dx1.shape, dx.shape))
    
    dist = sqrt((dx1 - dx)**2 + (dy1 - dy)**2)
    quiver(x, y, dx1 - dx, dy1 - dy, dist, scale=2e2, clim=(0, dist.mean()*2))
    colorbar()
    axis('image')
    title('Residuals')

def PlotShiftResidualsS(x, y, dx, dy, spx, spy):
    #figure()
    dx1 = spx.ev(x,y)
    dy1 = spy.ev(x,y)
    
    print((dx1.shape, dx.shape))
    
    dist = sqrt((dx1 - dx)**2 + (dy1 - dy)**2)
    quiver(x, y, dx1 - dx, dy1 - dy, dist, scale=2e2, clim=(0, dist.mean()*2))
    colorbar()
    axis('image')
    title('Residuals')

def PlotShiftField2(spx, spy, shape=[512, 256], voxelsize=(70., 70., 200.)):
    xi, yi = meshgrid(arange(0, shape[0]*voxelsize[0], 100), arange(0, shape[1]*voxelsize[1], 100));xin = xi.ravel();yin = yi.ravel()
    dx = spx.ev(xin[:], yin[:]).reshape(xi.shape)
    dy = spy.ev(xin[:], yin[:]).reshape(xi.shape)
    figure()
    subplot(131)
    #axes([.05, .7, .4, .25])
    imshow(dx[::-1, :], extent=[0,shape[0], 0, shape[1]])
    axis('image')
    colorbar()
    #axes([.55, .7, .4, .25])
    subplot(132)
    im = imshow(dy[::-1, :], extent=[0,shape[0], 0, shape[1]])
    axis('image')
    colorbar()

    #axes([.05, .05, .9, .6])
    xi, yi = meshgrid(arange(0, shape[0]*voxelsize[0], 2200), arange(0, shape[1]*voxelsize[1], 2200));xin = xi.ravel();yin = yi.ravel()
    dx = spx.ev(xin[:], yin[:]).reshape(xi.shape)
    dy = spy.ev(xin[:], yin[:]).reshape(xi.shape)
    subplot(133)
    
    quiver(xin/voxelsize[0], yin/voxelsize[0], spx.ev(xin, yin), spy.ev(xin, yin), scale=1e4)
    
    axis('image')
    xlim(0, shape[0])
    ylim(0, shape[1])

    #colorbar(im)