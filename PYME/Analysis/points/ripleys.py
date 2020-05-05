import numpy as np

from PYME.Analysis.points import DistHist

def ripleys_k_from_mask_points(x, y, xu, yu, n_bins, bin_size, mask_area, z=None, zu=None, threaded=False):
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
        mask_area : float
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
            dww = w*d.astype('f')/dw
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
            dww = w*d.astype('f')/dw
            dww[dw==0] = 0  # d[w==0]
            hist += dww

    K = (float(mask_area) / (lx ** 2)) * np.cumsum(hist)  # Ripley's K-function

    # return bins, K-function
    return bb, K

def points_from_mask(mask, sampling, three_d = True, coord_origin=(0,0,0)):
    vx, vy, vz = mask.voxelsize
    x0_m, y0_m, z0_m = mask.origin
    x0_p, y0_p, z0_p = coord_origin
    stride_x, stride_y, stride_z = [max(1, int(sampling / v)) for v in [vx, vy, vz]]
    
    if three_d:
        #convert mask to boolean image
        bool_mask = np.atleast_3d(mask.data[:, :, :, 0].squeeze()) > 0.5
        
        # generate uniformly sampled coordinates on mask
        xu, yu, zu = np.mgrid[0:bool_mask.shape[0]:stride_x, 0:bool_mask.shape[1]:stride_y,
                     0:bool_mask.shape[2]:stride_z]
        xu, yu, zu = vx * xu[bool_mask] + x0_m - x0_p, vy * yu[bool_mask] + y0_m - y0_p, vz * zu[
            bool_mask] + z0_m - z0_p
        
        mask_area = bool_mask.sum() * vx * vy * vz
    else:
        #convert mask to boolean image
        bool_mask = mask.data[:, :, :, 0].squeeze() > 0.5
        if bool_mask.ndim > 2:
            raise RuntimeError('Trying to calculate 2D Ripleys with 3D mask')
        
        zu = None
        xu, yu = np.mgrid[0:bool_mask.shape[0]:stride_x, 0:bool_mask.shape[1]:stride_y]
        xu, yu = vx * xu[bool_mask] + x0_m - x0_p, vy * yu[bool_mask] + y0_m - y0_p
        mask_area = bool_mask.sum() * vx * vy * vz
        
    return xu, yu, zu, mask_area
    

def ripleys_k(x, y, n_bins, bin_size, mask=None, bbox=None, z=None, threaded=False, sampling=5.0, coord_origin=(0,0,0)):
    """
    Ripley's K-function for examining clustering and dispersion of points within a
    region R, where R is defined by a mask (2D or 3D) of the data.
    
    x : np.array
            x-position of raw data
        y : np.array
            y-position of raw data
        n_bins : int
            Number of spatial bins
        bin_size: float
            Width of spatial bins (nm)
        mask : PYME.IO.image.ImageStack
            a mask of allowing the Ripleys to be computed within a given area
        bbox : optional, a tuple (x0, y0, x1, y1), or (x0, y0, z0, x1, y1, z1)
            bounding box of the region to consider if no mask provided (defaults to min and max of supplied data)
        z : optional, np.array
            z-position of raw data
        threaded : bool
            Calculate pairwise distances using multithreading (faster)
        sampling : float
            spacing (in nm) of samples from mask / region.
        coord_origin : 3-tuple
            Offset in nm of the x, y, and z coordinates w.r.t. the camera origin (used to make sure mask aligns)

    """
    three_d = z is not None
    
    if mask:
        xu, yu, zu, mask_area = points_from_mask(mask, sampling, three_d, coord_origin)
    else:
        if three_d:
            if not bbox:
                bbox = np.array([x.min(), y.min(), z.min(), x.max(), y.max(), z.max()])
                
            xu, yu, zu = np.mgrid[bbox[0]:bbox[3]:sampling, bbox[1]:bbox[4]:sampling, bbox[2]:bbox[5]:sampling]
            mask_area = np.prod((bbox[3:] - bbox[:3]))
        else:
            if not bbox:
                bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    
            xu, yu = np.mgrid[bbox[0]:bbox[2]:sampling, bbox[1]:bbox[3]:sampling]
            zu = None
            mask_area = np.prod((bbox[2:] - bbox[:2]))

    xu = xu.ravel()
    yu = yu.ravel()
    if zu is not None:
        zu = zu.ravel()
            
    return ripleys_k_from_mask_points(x=x, y=y, z=z,
                                      xu=xu, yu=yu, zu=zu,
                                      n_bins=n_bins, bin_size=bin_size, mask_area=mask_area, threaded=threaded)
        
        

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
