#!/usr/bin/python

##################
# correlationCoeffs.py
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
import numpy as np

def pearson(X, Y, roi_mask=None):
    if not roi_mask is None:
        #print X.shape, roi_mask.shape
        X = X[roi_mask]
        Y = Y[roi_mask]
        
    X = X - X.mean()
    Y = Y-Y.mean()
    return (X*Y).sum()/np.sqrt((X*X).sum()*(Y*Y).sum())

def overlap(X, Y, roi_mask=None):
    if not roi_mask is None:
        X = X[roi_mask]
        Y = Y[roi_mask]
    
    return (X*Y).sum()/np.sqrt((X*X).sum()*(Y*Y).sum())


def thresholdedManders(A, B, tA, tB, roi_mask=None):
    """Manders, as practically used with threshold determined masks"""
    A = A.astype('f')
    B = B.astype('f')
    if not roi_mask is None:
        A = A[roi_mask]
        B = B[roi_mask]
        
        #print A.shape, B.shape, tA, tB, A.sum(), (B>tB).sum()

    MA = ((B > tB)*A).sum()/A.sum()
    MB = ((A > tA)*B).sum()/B.sum()

    return MA, MB


def maskManders(A, B, mA, mB, roi_mask=None):
    """Manders, as practically used with threshold determined masks
    
    like thresholdedManders, but we pass already thresholded masks in to start with
    """
    A = A.astype('f')
    B = B.astype('f')
    if not roi_mask is None:
        A = A[roi_mask]
        B = B[roi_mask]
        mA = mA[roi_mask]
        mB = mB[roi_mask]
        
        
        #print A.shape, B.shape, tA, tB, A.sum(), (B>tB).sum()
    
    MA = (mB * A).sum() / A.sum()
    MB = (mA * B).sum() / B.sum()
    
    return MA, MB

def maskFractions(A, B, tA, tB):
    FA = (A > tA).mean()
    FB = (B > tB).mean()

    return FA, FB

def mutual_information(X, Y, roi_mask=None, nbins=256, bits=False):
    if not roi_mask is None:
        X = X[roi_mask]
        Y = Y[roi_mask]

    h = np.histogram2d(X.ravel(), Y.ravel(), bins=nbins)[0]
    
    pxy = h/float(h.sum())
    
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    
    px_py = px[:,None]*py[None,:]
    
    # pxy == 0 doesn't contribute to the sum, avoid problems with log by masking these out
    m = pxy > 0
    
    if bits:
        # return as 'bits' of entropy (i.e. calculate log base 2)
        return np.sum(pxy[m] * np.log2(pxy[m] / px_py[m]))
    else:
        return np.sum(pxy[m]*np.log(pxy[m]/px_py[m]))
    
    
def fourier_ring_correlation(imA, imB, voxelsize=[1.0, 1.0], window=False):
    import matplotlib.pyplot as plt
    from PYME.Analysis import binAvg
    
    imA = imA.squeeze()
    imB = imB.squeeze()
    
    X, Y = np.mgrid[0:float(imA.shape[0]), 0:float(imA.shape[1])]
    X = X / X.shape[0]
    Y = Y / X.shape[1]
    X = (X - .5)
    Y = Y - .5
    R = np.sqrt(X ** 2 + Y ** 2)
    
    if window:
        W =np.hanning(X.shape[0])[:,None]*np.hanning(X.shape[1])[None,:]
        imA = imA*W
        imB = imB*W
    
    H1 = np.fft.fftn(imA)
    H2 = np.fft.fftn(imB)
    
    #rB = np.linspace(0,R.max())
    rB = np.linspace(0, 0.5, 100)
    
    bn, bm, bs = binAvg.binAvg(R, np.fft.fftshift(H1 * H2.conjugate()).real, rB)
    
    bn1, bm1, bs1 = binAvg.binAvg(R, np.fft.fftshift((H1 * H1.conjugate()).real), rB)
    bn2, bm2, bs2 = binAvg.binAvg(R, np.fft.fftshift((H2 * H2.conjugate()).real), rB)
    
    plt.figure()

    fig = plt.gcf()
    try:
        fig.canvas.set_window_title('Fourier Ring Correlation')
    except AttributeError:
        # more recent version seem to have pushed this method into a manager sub-object
        # see also: https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.4.0.html#backend-deprecations
        fig.canvas.manager.set_window_title('Fourier Ring Correlation')

    ax = plt.gca()
    
    #FRC
    FRC = bm / np.sqrt(bm1 * bm2)
    ax.plot(rB[:-1], FRC)
    
    #noise envelope???????????
    ax.plot(rB[:-1], 2. / np.sqrt(bn / 2), ':')
    
    dfrc = np.diff(FRC)
    #print FRC
    monotone = np.where(dfrc > 0)[0][0] + 1
    
    #print FRC[:monotone], FRC[:(monotone+1)]
    
    intercept_m = np.interp(1.0 - 1 / 7.0, 1 - FRC[:monotone], rB[:monotone])
    
    print('Intercept_m= %3.2f (%3.2f nm)' % (intercept_m, voxelsize[0] / intercept_m))
    
    from scipy import ndimage
    f_s = np.sign(FRC - 1. / 7.)
    
    fss = ndimage.gaussian_filter(f_s, 10, mode='nearest')
    
    intercept = np.interp(0.0, - fss, rB[:-1])
    
    print('Intercept= %3.2f (%3.2f nm)' % (intercept, voxelsize[0] / intercept))
    
    xt = np.array([10., 15, 20, 30, 40, 50, 70, 90, 120, 150, 200, 300, 500])
    rt = voxelsize[0] / xt
    
    plt.xticks(rt[::-1], ['%d' % xi for xi in xt[::-1]], rotation='vertical')
    
    ax.plot([0, rt[0]], np.ones(2) / 7.0)
    
    plt.grid()
    plt.xlabel('Resolution [nm]')
    
    plt.ylabel('FRC')
    plt.ylim(0, 1.1)
    
    plt.plot([intercept, intercept], [0, 1], '--')
    
    plt.figtext(0.5, 0.5, 'FRC intercept at %3.1f nm' % (voxelsize[0] / intercept))
    
    plt.figure()
    plt.plot(rB[:-1], f_s)
    plt.plot(rB[:-1], fss)
    plt.show()
