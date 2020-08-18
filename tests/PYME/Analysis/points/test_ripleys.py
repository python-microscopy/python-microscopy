import numpy as np

## Two quick tests to assert that we detect a uniform distribution when
## there is a perfectly uniform distribution. These tests are by no means
## exhaustive, but the Ripley's functions are definitely broken if they 
## don't pass
def test_ripleys_k_2d():
    from PYME.Analysis.points.ripleys import ripleys_k_from_mask_points
    
    v = np.linspace(0,100,100)
    xr,yr = np.meshgrid(v,v)
    xu = xr.ravel()
    yu = yr.ravel()

    NBINS = 10
    BIN_SIZE = 5

    A = 100**2
        
    bb, K = ripleys_k_from_mask_points(xu, yu, xu, yu, NBINS, BIN_SIZE, A,
                      z=np.zeros_like(xu), area_per_mask_point=1, zu=None, threaded=False)

    assert (np.sum((np.pi*bb**2-K)**2) < 1e-4)

def test_ripleys_k_3d():
    from PYME.Analysis.points.ripleys import ripleys_k_from_mask_points
    
    v = np.linspace(0,10,10)
    xr,yr,zr = np.meshgrid(v,v,v)
    xu = xr.ravel()
    yu = yr.ravel()
    zu = zr.ravel()

    NBINS = 5
    BIN_SIZE = 1

    A = 10**3
        
    bb, K = ripleys_k_from_mask_points(xu, yu, xu, yu, NBINS, BIN_SIZE, A,
                      area_per_mask_point=1, z=zu, zu=zu, threaded=False)

    assert (np.sum(((4.0/3.0)*np.pi*bb**3-K)**2) < 1e-4)
