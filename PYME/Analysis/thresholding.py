#!/usr/bin/python

##################
# thresholding.py
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

import numpy
import numpy as np
import warnings

def _histogram(img, nbins=255, bin_spacing='linear'):
    im_mean = img.mean()
    if bin_spacing == 'log':
        bins = np.logspace(np.log10(img.min() + im_mean), np.log10(img.max() + im_mean), nbins) - im_mean
    elif (bin_spacing == 'adaptive'):
        imr = img.ravel()
        imr = imr[imr > 0]
        imr.sort()
        bins = imr[::int(len(imr) / nbins)]
        #print bins
        #nbins = len(bins)
    else:
        im_max = img.max()
        
        if im_max > 50 * im_mean:
            warnings.warn(RuntimeWarning('''Maximum value in image > 50*mean.
                All data will be concentrated in the lowest few bins and thresholding will be unreliable.
                Try running with bin_spacing='log' or bin_spacing='adaptive' '''))
        
        bins = np.linspace(img.min(), img.max(), nbins)
    
    return np.histogram(img, bins)

def isodata_f(img, nbins=255, bin_spacing='linear', tol=1e-5):
    """calculate isodata threshold - does iterations on histogrammed data rather than
    raw data to improve speed
    
    img - the image data
    nbins - the number of bins used for the histogram
    """

    N, bins = _histogram(img, nbins, bin_spacing)
    nbins = len(bins)
    
#    #calculate bin centres
    bin_mids = 0.5*(bins[:-1] + bins[1:])
    
    #precalculate bin weightings
    bw = N*bin_mids#numpy.arange(len(N))

    #start off with the largest possible delta
    delta = bin_mids[-1] #nbins

    #start off with threshold at middle of range
    t = bin_mids[int(nbins / 2)] #delta/2.

    while delta > tol:
        #new threshold = mean of the two segment means
        t_i = numpy.searchsorted(bin_mids, t)
        tn = (bw[:t_i].sum()/N[:t_i].sum() + bw[t_i:].sum()/N[t_i:].sum())/2

        delta = abs(tn - t)
        t = tn

    return t#bins[t]


def isodata(img, tol = 1e-3):
    """correct isodata implementation - slower than above
    tolerance indicates fraction of data range when delta is deemed not to have changed
    """

    imin = img.min()
    imax = img.max()

    #initial delta is data range
    delta = imax - imin

    tol = tol*delta

    #initial threshold is range midpoint
    t = imin + delta/2.

    while delta > tol:
        tn = (img[img <= t].mean() + img[img > t].mean())/2.
        delta = abs(t - tn)
        tn = t

    return t

def otsu(image, nbins=256, bin_spacing = 'linear'):
    """Return threshold value based on Otsu's method.
    
    Adapted from the skimage implmentation to allow non-linear bins
    
    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    ValueError
         If `image` only contains a single grayscale value.
    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    Notes
    -----
    The input image must be grayscale.
    """
    if len(image.shape) > 2 and image.shape[-1] in (3, 4):
        msg = "threshold_otsu is expected to work correctly only for " \
              "grayscale images; image shape {0} looks like an RGB image"
        warnings.warn(RuntimeWarning(msg.format(image.shape)))

    # Check if the image is multi-colored or not
    if image.min() == image.max():
        raise ValueError("threshold_otsu is expected to work with images "
                         "having more than one color. The input image seems "
                         "to have just one color {0}.".format(image.min()))

    hist, bins = _histogram(image.ravel(), nbins, bin_spacing)
    # calculate bin centres
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def signalFraction(img, frac, nbins = 5000):
    """ threshold to include a certain fraction of the total signal"""

    N, bins = numpy.histogram(img, nbins)

    #calculate bin centres
    bin_mids = (bins[:-1] )

    cN = numpy.cumsum(N*bin_mids)

    #print cN

    i = numpy.argmin(abs(cN - cN[-1]*(1-frac)))

    #print abs(cN - cN[-1]*frac)
    #print i, frac

    return bins[i]
