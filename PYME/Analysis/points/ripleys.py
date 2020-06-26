import numpy as np

from PYME.Analysis.points import DistHist

def ripleys_k_from_mask_points(x, y, xu, yu, n_bins, bin_size, mask_area, area_per_mask_point, z=None, zu=None, threaded=False):
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
        area_per_mask_point : float
            Scaling factor for weight calculations
        z : np.array
            z-position of raw data
        zu : np.array
            z-position of simulated uniform random data over region R
        threaded : bool
            Calculate pairwise distances using multithreading (faster)
    """
    from PYME.Analysis.points import DistHist

    bb = float(bin_size)*np.arange(1, n_bins+1)  # bins
    hist = np.zeros(n_bins)  # counts
    lx = len(x)

    if (z is None) or (np.count_nonzero(z) == 0):
        # 2D case
        if threaded:
            dh_func = DistHist.distanceHistogramThreaded
        else:
            dh_func = DistHist.distanceHistogram
        # Count the number of points we expect in a filled annulus
        # for comparison to the actual count.
        w = np.pi*(bb**2-(bb-bin_size)**2)/area_per_mask_point

        # Calculate weighted pairwise distance histogram
        for _i in range(lx):
            # Count the number of points within the mask at a distance r
            # from point _i
            dw = dh_func(x[_i], y[_i], xu, yu, n_bins, bin_size)
            # Count the number of points in the original data set at a 
            # distance r from point _i
            d = dh_func(x[_i], y[_i], x, y, n_bins, bin_size)
            # This is used a rough approximation of the Ripley's 
            # correction: 1/the fraction of the circumference
            # in the mask divided by the total circumference
            # 1/(dw/w)
            dww = w*d.astype('f')/dw
            # If there are no points in the mask, at this radius,
            # don't include these distances
            dww[dw==0] = 0
            # The histogram is a weighted count of d
            hist += dww
    else:
        # 3D case
        if threaded:
            dh_func = DistHist.distanceHistogram3DThreaded
        else:
            dh_func = DistHist.distanceHistogram3D
        w = (4.0/3.0)*np.pi*(bb**3-(bb-bin_size)**3)/area_per_mask_point
        
        # Calculate weighted pairwise distance histogram
        for _i in range(lx):
            dw = dh_func(x[_i], y[_i], z[_i], xu, yu, zu, n_bins, bin_size)
            d = dh_func(x[_i], y[_i], z[_i], x, y, z, n_bins, bin_size)
            dww = w*d.astype('f')/dw
            dww[dw==0] = 0 
            hist += dww

    K = (float(mask_area) / (lx ** 2)) * np.cumsum(hist)  # Ripley's K-function

    # return bins, K-function
    return bb, K

def points_from_mask(mask, sampling, three_d=True, coord_origin=(0,0,0)):
    """
    Calculate coordinate positions in nm from a regularly-sampled mask.

    Parameters
    ----------
        mask : np.array
            Mask image (array of 0s and 1s)
        sampling : float
            Sampling rate of mask image in nm.
        three_d : bool
            Return 2D or 3D coordinates
        coord_origin : 3-tuple
            Offset in nm of the x, y, and z coordinates w.r.t. the camera 
            origin (used to make sure mask aligns)
    """
    vx, vy, vz = mask.voxelsize_nm
    x0_m, y0_m, z0_m = mask.origin
    x0_p, y0_p, z0_p = coord_origin

    if (vz < 1e-12) and not three_d:
        vz = 1 #dummy value to prevent div by zero when calculating strides we don't use

    assert ((vx>0) and (vy>0) and (vz > 0))
    stride_x, stride_y, stride_z = [max(1, int(sampling/v)) for v in [vx, vy, vz]]
    
    if three_d:
        #convert mask to boolean image
        bool_mask = np.atleast_3d(mask.data[:, :, :, 0].squeeze()) > 0.5
        
        # generate uniformly sampled coordinates on mask
        xu, yu, zu = np.mgrid[0:bool_mask.shape[0]:stride_x, 0:bool_mask.shape[1]:stride_y,0:bool_mask.shape[2]:stride_z]
        xu, yu, zu = xu.ravel(), yu.ravel(), zu.ravel()
        
        #TODO - use ndimage.map_coordinates instead?
        mask_v = bool_mask[xu, yu, zu]
        xu, yu, zu = vx * xu[mask_v] + x0_m - x0_p, vy * yu[mask_v] + y0_m - y0_p, vz * zu[mask_v] + z0_m - z0_p
        
        mask_area = bool_mask.sum() * vx * vy * vz
        area_per_point = vx * vy * vz * stride_x * stride_y * stride_z
    else:
        #convert mask to boolean image
        bool_mask = mask.data[:, :, :, 0].squeeze() > 0.5
        if bool_mask.ndim > 2:
            raise RuntimeError('Trying to calculate 2D Ripleys with 3D mask')
        
        zu = None
        xu, yu = np.mgrid[0:bool_mask.shape[0]:stride_x, 0:bool_mask.shape[1]:stride_y]
        xu, yu = xu.ravel(), yu.ravel()

        #TODO - use ndimage.map_coordinates instead?
        mask_v = bool_mask[xu, yu]
        xu, yu = vx * xu[mask_v] + x0_m - x0_p, vy * yu[mask_v] + y0_m - y0_p
        
        mask_area = bool_mask.sum() * vx * vy
        area_per_point = vx * vy * stride_x * stride_y
        
    return xu, yu, zu, mask_area, area_per_point

def mc_points_from_mask(mask, n_points, three_d=True, coord_origin=(0,0,0)):
    """
    Calculate coordinate positions in nm from a Monte-Carlo-sampled mask.

    Parameters
    ----------
        mask : np.array
            Mask image (array of 0s and 1s)
        n_points : int
            Number of points to return.
        three_d : bool
            Return 2D or 3D coordinates
        coord_origin : 3-tuple
            Offset in nm of the x, y, and z coordinates w.r.t. the camera 
            origin (used to make sure mask aligns)
    """
    vx, vy, vz = mask.voxelsize_nm
    x0_m, y0_m, z0_m = mask.origin
    x0_p, y0_p, z0_p = coord_origin

    eps = 0.2  # scaling fudge factor

    if three_d:
        #convert mask to boolean image
        bool_mask = np.atleast_3d(mask.data[:, :, :, 0].squeeze()) > 0.5
        mask_area = bool_mask.sum()

        # Scale up number of points to simulate so we still have n_points left 
        # after the Monte-Carlo rejection step
        n_sim = int((np.prod(bool_mask.shape)/mask_area + eps)*n_points)
        
        # generate randomly sampled coordinates on a 3D space
        xu, yu, zu = (np.random.rand(n_sim,3)*[bool_mask.shape[0]-1,bool_mask.shape[1]-1,bool_mask.shape[2]-1]).T
        # Find corresponding (xu, yu, zu) in the mask
        point_mask = bool_mask[np.round(xu).astype(int), np.round(yu).astype(int), np.round(zu).astype(int)]
        # Monte-Carlo reject points outside of the mask and shift points inside to coordinate position
        xu, yu, zu = vx * xu[point_mask] + x0_m - x0_p, vy * yu[
            point_mask] + y0_m - y0_p, vz * zu[point_mask] + z0_m - z0_p
        
    else:
        bool_mask = mask.data[:, :, :, 0].squeeze() > 0.5
        if bool_mask.ndim > 2:
            raise RuntimeError('Trying to calculate 2D Ripleys with 3D mask')
        mask_area = bool_mask.sum()

        n_sim = int((np.prod(bool_mask.shape)/mask_area + eps)*(n_points + 3*np.sqrt(n_points+1) + 3))
        
        zu = None
        xu, yu = (np.random.rand(n_sim,2)*[bool_mask.shape[0]-1,bool_mask.shape[1]-1]).T
        point_mask = bool_mask[np.round(xu).astype(int), np.round(yu).astype(int)]
        xu, yu = vx * xu[point_mask] + x0_m - x0_p, vy * yu[point_mask] + y0_m - y0_p

    if (len(xu) < n_points) or (len(yu) < n_points) or ((zu is not None) and (len(zu) < n_points)):
        print('n_points, n_sim, len(xu):', n_points, n_sim, len(xu)) #debug
        # This one's for the developers
        raise RuntimeError('Not enough points were generated in the Monte-Carlo simulations. Revisit calculation of n_sim.')
    # Truncate
    xu, yu, zu = xu[:n_points], yu[:n_points], zu[:n_points] if zu is not None else None

    return xu, yu, zu

def mc_sampling_statistics(K, n_points, n_bins, bin_size, mask, three_d, 
                            significance=0.05, n_sim=20, threaded=False, sampling=5.0, 
                            coord_origin=(0,0,0)):
    """
    Calculates simulation envelope and significance of clustering on a mask
    by simulating random uniform distributions on the mask using Monte-Carlo 
    sampling.

    Parameters
    ----------
        K : np.array
            Ripley's K-values for real points to be compared to simulations.
            Expects output from ripleys_k.
        n_points : int
            Number of real points. Defines size of simulations.
        n_bins : int
            Number of spatial bins
        bin_size: float
            Width of spatial bins (nm)
        mask : PYME.IO.image.ImageStack
            a mask of allowing the Ripleys to be computed within a given area
        three_d : bool
            Indicates if the real points are in 2D or 3D space
        n_sim : int
            Number of Monte-Carlo simulations to run. More simulations = 
            more statistical power.
        threaded : bool
            Calculate pairwise distances using multithreading (faster)
        sampling : float
            spacing (in nm) of samples from mask / region.
        coord_origin : 3-tuple
            Offset in nm of the x, y, and z coordinates w.r.t. the camera 
            origin (used to make sure mask aligns)
    """
    xu, yu, zu, mask_area, area_per_mask_point = points_from_mask(mask, sampling, three_d, coord_origin)
    xu = xu.ravel()
    yu = yu.ravel()
    if zu is not None:
        zu = zu.ravel()

    print('mask_area:', mask_area) #debug
    
    # Monte-Carlo simulations on the mask
    K_arr = np.zeros((n_sim,len(K)))
    for _i in range(n_sim):
        print('Simulation {} ...'.format(_i))
        xm, ym, zm = mc_points_from_mask(mask, n_points, three_d, coord_origin)
        _, K_arr[_i,:] = ripleys_k_from_mask_points(x=xm, y=ym, z=zm,
                                    xu=xu, yu=yu, zu=zu,
                                    n_bins=n_bins, bin_size=bin_size, mask_area=mask_area, 
                                    area_per_mask_point=area_per_mask_point, threaded=threaded)
    # Envelope
    K_min = np.percentile(K_arr, significance*100, axis=0)
    K_max = np.percentile(K_arr, 100*(1.0-significance), axis=0)
    # Probability our original data is clustered (< 1/(n_sim+1) 
    # implies significance)
    # 
    # Definition of one-tailed p-value: we check the probability that our 
    # simulation K-values are greater than or equal to the real data 
    # K-values given that we expect no the simulated and real values
    # to be drawn from the same distributions (null hypothesis)
    p_clustered = ((K_arr>=K).sum(0) + 1)/(n_sim + 1)
    p_dispersed = ((K_arr<=K).sum(0) + 1)/(n_sim + 1)
    return K_min, K_max, p_clustered, p_dispersed

def ripleys_k(x, y, n_bins, bin_size, mask=None, bbox=None, z=None, 
                threaded=False, sampling=5.0, coord_origin=(0,0,0)):
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
            bounding box of the region to consider if no mask provided 
            (defaults to min and max of supplied data)
        z : optional, np.array
            z-position of raw data
        threaded : bool
            Calculate pairwise distances using multithreading (faster)
        sampling : float
            spacing (in nm) of samples from mask / region.
        coord_origin : 3-tuple
            Offset in nm of the x, y, and z coordinates w.r.t. the camera 
            origin (used to make sure mask aligns)

    """
    three_d = z is not None
    
    if mask:
        xu, yu, zu, mask_area, area_per_mask_point = points_from_mask(mask, sampling, three_d, coord_origin)
    else:
        if three_d:
            if bbox is None:
                bbox = np.array([x.min(), y.min(), z.min(), x.max(), y.max(), z.max()])
                
            xu, yu, zu = np.mgrid[bbox[0]:bbox[3]:sampling, bbox[1]:bbox[4]:sampling, bbox[2]:bbox[5]:sampling]
            mask_area = np.prod((bbox[3:] - bbox[:3]))
            area_per_mask_point = float(sampling)**3
        else:
            if bbox is None:
                bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    
            xu, yu = np.mgrid[bbox[0]:bbox[2]:sampling, bbox[1]:bbox[3]:sampling]
            zu = None
            mask_area = np.prod((bbox[2:] - bbox[:2]))
            area_per_mask_point = float(sampling) ** 2

    xu = xu.ravel()
    yu = yu.ravel()
    if zu is not None:
        zu = zu.ravel()

    return ripleys_k_from_mask_points(x=x, y=y, z=z,
                                      xu=xu, yu=yu, zu=zu,
                                      n_bins=n_bins, bin_size=bin_size, mask_area=mask_area, 
                                      area_per_mask_point=area_per_mask_point, threaded=threaded)
        
        

def ripleys_l(bb, K, d=2):
    """
    Normalizes Ripley's K-function to an L-function.

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
    if d == 2:
        # Normalize 2D
        L = np.sqrt(K/np.pi)
    elif d == 3:
        # Normalize 3D
        L = ((3.0/(4.0*np.pi))*K)**(1.0/3.0)
    else:
        raise ValueError('Please enter a valid dimension.')

    return bb, L

def ripleys_h(bb, K, d=2):
    """
    Normalizes Ripley's K-function to an H-function such that  > 0 indicates
    clustering and H < 0 indicates dispersion.

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
    bb, L = ripleys_l(bb, K, d)

    H = L - bb

    return bb, H

def ripleys_dl(bb, K, d=2):
    """
    Derivative of the H-function. 

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
    bb, L = ripleys_l(bb, K, d)
    bin_size = np.diff(bb)[0]

    # central difference
    dL = (L[2:]-L[:-2])/(2*bin_size)

    return bb[1:-1], dL


def ripleys_dh(bb, K, d=2):
    """
    Derivative of the H-function. 

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
    bb, H = ripleys_h(bb, K, d)
    bin_size = np.diff(bb)[0]

    # central difference
    dH = (H[2:]-H[:-2])/(2*bin_size)

    return bb[1:-1], dH
