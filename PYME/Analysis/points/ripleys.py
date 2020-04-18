import numpy as np

from PYME.Analysis.points import DistHist

def ripleys_k(x, y, xu, yu, n_bins, bin_size, area, z=None, zu=None, threaded=False):
    """
    Ripley's K-function for examining clustering and dispersion of points within a
    region R, where R is defined by a mask (2D or 3D) of the data.

    Parameters
    ----------
        x : np.array
            x-position of raw data
        y : np.array
            y-position of raw data
        xu : np.array
            x-position of simulated uniform random data over region R
        yu : np.array
            y-position of simulated uniform random data over region R
        n_bins : int
            Number of spatial bins
        bin_size: float
            Width of spatial bins (nm)
        area : float
            Area of region R (sum of the mask)
        z : np.array
            z-position of raw data
        zu : np.array
            z-position of simulated uniform random data over region R
        threaded : bool
            Calculate pairwise distances using multithreading (faster)
    """
    from PYME.Analysis.points import DistHist

    bb = np.arange(0, n_bins*bin_size, bin_size)  # bins
    hist = np.zeros(n_bins)  # counts
    lx = len(x)

    if (z is None) or (np.count_nonzero(z) == 0):
        # 2D case
        if threaded:
            dh_func = DistHist.distanceHistogramThreaded
        else:
            dh_func = DistHist.distanceHistogram
        w = np.pi*((bb+bin_size)**2-bb**2)  # we have an annulus, not a circumference

        # Calculate weighted pairwise distance histogram
        for _i in range(lx):
            dw = dh_func(x[_i], y[_i], xu, yu, n_bins, bin_size)
            d = dh_func(x[_i], y[_i], x, y, n_bins, bin_size)
            dww = w*d/dw
            dww[dw==0] = 0  # d[w==0]
            hist += dww
    else:
        # 3D case
        if threaded:
            dh_func = DistHist.distanceHistogram3DThreaded
        else:
            dh_func = DistHist.distanceHistogram3D
        w = (4.0/3.0)*np.pi*((bb+bin_size)**3-bb**3)
        
        # Calculate weighted pairwise distance histogram
        for _i in range(lx):
            dw = dh_func(x[_i], y[_i], z[_i], xu, yu, zu, n_bins, bin_size)
            d = dh_func(x[_i], y[_i], z[_i], x, y, z, n_bins, bin_size)
            dww = w*d/dw
            dww[dw==0] = 0  # d[w==0]
            hist += dww

    K = (area/(lx**2))*np.cumsum(hist)  # Ripley's K-function

    # return bins, K-function
    return bb, K

def ripleys_l(bb, K, d=2):
    """
    Normalizes Ripley's K-function to an L-function such that L > 0 indicates
    clustering and L < 0 indicates dispersion.

    Parameters
    ----------
        bb : np.array
            Histogram bins associated with K.
        K : np.array
            Ripley's K-function, calculated from 
            PYME.Analysis.points.spatial_descriptive.ripleys_k
        d : int
            Dimension of the input data to calculate K (2 or 3).
    """
    bin_size = np.diff(bb)[0]
    if d == 2:
        # Normalize 2D
        L = np.sqrt(K/np.pi)
    elif d == 3:
        # Normalize 3D
        L = ((3.0/(4.0*np.pi))*K)**(1.0/3.0)
    else:
        raise ValueError('Please enter a valid dimension.')

    L -= bb+bin_size

    return bb, L
