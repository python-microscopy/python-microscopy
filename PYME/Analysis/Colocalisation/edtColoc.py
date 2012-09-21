#!/usr/bin/python

##################
# edtColoc.py
#
# Copyright David Baddeley, 2010
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

from scipy import ndimage
import numpy
from PYME.Analysis import binAvg

def imageDensityAtDistance(A, mask, voxelsize = None, bins=100):
    '''Calculates the distribution of a label at varying distances from a mask.
    Negative distances are on the inside of the mask.

    Parameters:
    A - intensity image
    mask - binary mask
    voxelsize - size of the pixels/voxels - should be either a constant, or an iterable
                with a length equal to the number of dimensions in the data
    bins - either a number of bins, or an array of bin edges


    Returns:
    bn - integrated intensity in distance bin
    bm - mean intensity in distance bin
    bins - the bin edges
    '''
    
    if voxelsize == None:
        voxelsize = numpy.ones(len(A.shape))

    dt = -ndimage.distance_transform_edt(mask, sampling=voxelsize)

    dt = dt + ndimage.distance_transform_edt(1- ndimage.binary_dilation(mask), sampling=voxelsize)

    if numpy.isscalar(bins):
        bins = numpy.linspace(dt.min(), dt.max(), bins+1)
        
    #print bins

    bn, bm, bs = binAvg.binAvg(dt, A, bins)

    return bn, bm, bins


def pointDensityAtDistance(points, mask, voxelsize, maskOffset, bins=100):
    '''Calculates the distribution of a label at varying distances from a mask.
    Negative distances are on the inside of the mask.

    Parameters:
    points - array containing point coordinates
    mask - binary mask
    voxelsize - size of the pixels/voxels in mask - should be an iterable
                with a length equal to the number of dimensions in the data
    maskOffset - iterable with lengh equal to number of dims giving coordinates (in point space)
                 or the 0th pixel in the mask
    bins - either a number of bins, or an array of bin edges


    Returns:
    bn - integrated intensity in distance bin
    bm - mean intensity in distance bin
    bins - the bin edges
    '''

    voxelsize = numpy.array(voxelsize)

    dt = -ndimage.distance_transform_edt(mask, sampling=voxelsize)

    dt = dt + ndimage.distance_transform_edt(ndimage.binary_dilation(1-mask), sampling=voxelsize)

    pixelCoords = numpy.round((points - maskOffset[None, :])/[voxelsize[None, :]]).astype('i')

    dists = dt[pixelCoords]


    if numpy.isscalar(bins):
        bins = numpy.linspace(dt.min(), dt.max(), bins+1)

    n_events, b = numpy.histogram(dists, bins)
    n_pixels, b = numpy.histogram(dt.flat, bins)

    ev_density = n_events/n_pixels
    ev_density[n_pixels ==0] = 0

    return n_events, ev_density, bins



