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
import numpy as np
from PYME.Analysis import binAvg

def imageDensityAtDistance(A, mask, voxelsize = None, bins=100, roi_mask=None):
    """Calculates the distribution of a label at varying distances from a mask.
    Negative distances are on the inside of the mask.

    Parameters
    -----------
        
    A - intensity image
    mask - binary mask
    voxelsize - size of the pixels/voxels - should be either a constant, or an iterable
                with a length equal to the number of dimensions in the data
    bins - either a number of bins, or an array of bin edges


    Returns
    --------
        
    bn - number of pixels in distance bin
    bm - mean intensity in distance bin
    bins - the bin edges
    """
    
    if voxelsize is None:
        voxelsize = numpy.ones(len(A.shape))

    dt = -ndimage.distance_transform_edt(mask, sampling=voxelsize)

    dt = dt + ndimage.distance_transform_edt(1- ndimage.binary_dilation(mask), sampling=voxelsize)

    if numpy.isscalar(bins):
        bins = numpy.linspace(dt.min(), dt.max(), bins+1)
        
    #print bins
    if not roi_mask is None:
        bn, bm, bs = binAvg.binAvg(dt[roi_mask], A[roi_mask], bins)
    else:
        bn, bm, bs = binAvg.binAvg(dt, A, bins)

    return bn, bm, bins

def image_enrichment_and_fraction_at_distance(A, mask, voxelsize = None, bins=100, roi_mask=None):
    """
    returns the relative enrichment of label of a label at a given distance from a mask, along with the total signal
    enclosed within that distance.
    
    -ve distances correspond to points in the interior of the mask.
    """
    
    bnA, bmA, binsA = imageDensityAtDistance(A, mask, voxelsize, bins, roi_mask=roi_mask)

    enrichment = bmA / bmA[bnA > 1].mean()

    total = bmA * bnA
    enclosed_signal = np.cumsum(total / total.sum())
    
    enclosed_area = np.cumsum(bnA.astype('f')/bnA.sum())
    
    return binsA, enrichment, enclosed_signal, enclosed_area


def plot_image_dist_coloc_figure(bins, enrichment_BA, enrichment_AA, enclosed_BA, enclosed_AA, enclosed_area, pearson=None, MA=None,
                                 MB=None, nameA='A', nameB = 'B'):
    import matplotlib.pyplot as plt
    import scipy.interpolate
    
    #find the distance at which 50% of the labelling is included
    d_50 = float(scipy.interpolate.interp1d(enclosed_BA, bins[1:])(.5))
    
    f = plt.figure()
    if not pearson is None:
        plt.figtext(.1, .95, 'Pearson: %2.2f   M1: %2.2f M2: %2.2f' % (pearson, MA, MB))
    
    plt.subplot(211)
    plt.plot(bins[1:], enrichment_BA, lw=2, drawstyle='steps')
    if not enrichment_AA is None:
        plt.plot(bins[1:], enrichment_AA, 'k--', drawstyle='steps')#, binsA[1] - binsA[0])
        
    plt.xlabel('Distance from edge of %s [nm]' % nameA)
    plt.ylabel('Relative enrichment')# % nameB)
    
    
    
    plt.plot([bins[0], bins[-1]], [1, 1], 'k:')
    plt.grid()
    plt.xlim([bins[0], bins[-1]])

    plt.legend([nameB, nameA + ' (control)', 'uniform'], fontsize='medium', frameon=False)
    
    plt.subplot(212)
    plt.plot(bins[1:], enclosed_BA, lw=2)
    if not enclosed_AA is None:
        plt.plot(bins[1:], enclosed_AA, 'k--')

    plt.plot(bins[1:], enclosed_area, 'k:')
    
    plt.plot([bins[0], d_50], [.5, .5], 'r:')
    plt.plot([d_50, d_50], [0, .5], 'r:')
    
    plt.text(d_50 + 150, .45, '50%% of %s is within %d nm' % (nameB, d_50))
    
    plt.xlabel('Distance from edge of %s [nm]' % nameA)
    plt.ylabel('Fraction of %s enclosed' % nameB)
    plt.grid()
    plt.xlim([bins[0], bins[-1]])
    
    return f


def pointDensityAtDistance(points, mask, voxelsize, maskOffset, bins=100):
    """Calculates the distribution of a label at varying distances from a mask.
    Negative distances are on the inside of the mask.

    Parameters
    ----------
    points : np.ndarray
        array containing point coordinates
    mask : np.ndarray 
        binary mask
    voxelsize : iterable 
        size of the pixels/voxels in mask - should be an iterable
        with a length equal to the number of dimensions in the data
    maskOffset : iterable
        iterable with lengh equal to number of dims giving coordinates (in point space)
        or the 0th pixel in the mask
    bins : int or ndarray, default=100 
        either a number of bins, or an array of bin edges


    Returns
    -------
    bn : ndarray 
        integrated intensity in distance bin
    bm : ndarray 
        mean intensity in distance bin
    bins : ndarray
        the bin edges

    """

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



