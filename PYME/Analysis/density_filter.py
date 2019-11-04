"""
Estimate density in a photon-starved image

"""

import numpy as np


class DensityFilter(object):
    def __init__(self, N=10, half_size=3):
        x_, y_ = np.mgrid[-half_size:(half_size + 1.0), -half_size:(half_size + 1.0)]
        r = 0.5 + np.sqrt(x_ * x_ + y_ * y_)
        
        rr = r.ravel()
        self.idx = np.argsort(rr)
        self.ri = rr[self.idx]
        
        self.u, indices = np.unique(self.ri, return_index=True)
        
        self.ind1 = np.hstack((indices[1:], len(self.ri))) - 1
        self.kernel_shape = r.shape
        
        self.N = N
    
    def __call__(self, roi):
        """ Kernel to be used with scipy.ndimage.generic_filter"""
        roi_r = roi.ravel()[self.idx]
        
        n_ = np.cumsum(roi_r)
        id2 = min(np.searchsorted(n_, self.N), len(self.ri) - 1)
        
        id3 = self.ind1[:(np.searchsorted(self.u, self.ri[id2]) + 1)]
        
        rv = self.ri[id3]
        n = n_[id3]
        
        A = np.atleast_2d(rv * rv).T
        
        return np.linalg.lstsq(A, n, rcond=None)[0][0]
    
    def filter(self, image):
        from scipy import ndimage
        
        return ndimage.generic_filter(image, self, self.kernel_shape)
    