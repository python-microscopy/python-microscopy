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
    