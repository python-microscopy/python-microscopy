#!/usr/bin/python
##################
# colocScaleDep.py
#
# Copyright David Baddeley, 2011
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
import numpy as np
from scipy import ndimage
from . import correlationCoeffs
from pylab import *
from . import edtColoc
from PIL import Image

DIRNAME='/home/david/Desktop/coloc/'

def rendP(imR, imG, x, y, w):
    imR[(x-w):x, (y-w/2):(y+w/2)] +=1
    imG[x:(x+w), (y-w/2):(y+w/2)] +=1

def rendS(imR, imG, x, y, w):
    if rand() > .5:
        imR[(x-w):x, (y-w/2):(y+w/2)] +=1
    else:
        imG[x:(x+w), (y-w/2):(y+w/2)] +=1


def corrScale(s, imR, imG):
    bR = ndimage.gaussian_filter(imR, s)
    bG = ndimage.gaussian_filter(imG, s)
    figure(2)
    clf()
    imshow(concatenate([bR[:,:,None]/bR.max(), bG[:,:,None]/bG.max(), zeros(imR.shape)[:,:,None]], 2))
    savefig(DIRNAME + 'blurred_%dnm.pdf' % (2.35*s))
    Image.fromarray((concatenate([bR[:,:,None]/bR.max(), bG[:,:,None]/bG.max(), zeros(imR.shape)[:,:,None]], 2)*255).astype('uint8')).save(DIRNAME + 'blurred_%dnm.tif' % (2.35*s))

    ps = correlationCoeffs.pearson(bR, bG)
    md1, md2 = correlationCoeffs.thresholdedManders(bR, bG, bR.max()/2, bG.max()/2)

    bn, bm, bins = edtColoc.imageDensityAtDistance(bR, bG > bG.max()/2, bins=np.arange(-200, 200, 5) + .01)
    figure(3)
    clf()
    bar(bins[:-1], bm, 5)
    ylim(0, .25)
    xlim(-50, 200)
    xlabel('Distance from edge of A [nm]')
    ylabel('Density of B')
    savefig(DIRNAME + 'dist_dense_%dnm.pdf' % (2.35*s))

    return ps, md1, md2, bm, bins

def genIm(size=500, w=30):
    imR = np.zeros((size, size))
    imG = np.zeros((size, size))

    rendP(imR, imG, size/2, size/2, w)

    return imR, imG

def genImR(size=500, w=30):
    imR = np.zeros((size, size))
    imG = np.zeros((size, size))

    for i in range(10):
        rendP(imR, imG, size*rand(), size*rand(), w)

    return imR, imG

def genImR2(size=500, w=30):
    imR = np.zeros((size, size))
    imG = np.zeros((size, size))

    for i in range(20):
        rendS(imR, imG, size*rand(), size*rand(), w)

    return imR, imG

def genPlots():
    imR, imG = genIm()

    close('all')

    figure()
    imshow(concatenate([imR[:,:,None], imG[:,:,None], zeros(imR.shape)[:,:,None]], 2))
    xlabel('Position [nm]')
    ylabel('Position [nm]')
    savefig(DIRNAME + 'object.pdf')

    figure()

    sca = arange(0, 100, 5)
    cfs = [corrScale(s, imR, imG) for s in sca]
    cfa = array([c[:-2] for c in cfs])

    figure()
    plot(2.35*sca, cfa[:,0], lw=2)
    ylim(0, 1)
    xlabel('Resolution (FWHM) [nm]')
    ylabel('Pearson Correlation Coefficient')
    savefig(DIRNAME + 'pearsons.pdf')






