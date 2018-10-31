import os
import numpy as np
from PYME.localization import Test
from PYME.Analysis.MetaData import TIRFDefault
#from PYME.localization.FitFactories.InterpFitR import f_Interp3d

def f_Interp3d(p, interpolator, X, Y, Z, safeRegion, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    if len(p) == 5:
        A, x0, y0, z0, b = p
    else:
        A, x0, y0, z0 = p
        b = 0

    #currently just come to a hard stop when the optimiser tries to leave the safe region
    #prob. not ideal, for a number of reasons
    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(np.nanmax([z0, safeRegion[2][0]]), safeRegion[2][1])

    return interpolator.interp(X - x0 + 1, Y - y0 + 1, Z - z0 + 1)*A + b

def test_CSInterpolator():
    from PYME.localization.FitFactories.Interpolators.CSInterpolator import interpolator
    interpolator.setModelFromFile(os.path.join(os.path.dirname(Test.__file__), 'astig_theory.tif'))
    
    roiHalfSize = 5
    x, y = 10.,10.

    X, Y, Z, safeRegion = interpolator.getCoords(TIRFDefault, slice(x - roiHalfSize, x + roiHalfSize + 1),
                                                 slice(y - roiHalfSize, y + roiHalfSize + 1), slice(0, 1))

    m = f_Interp3d([1,x,y,0], interpolator, X, Y, Z, safeRegion)
    
    assert(np.all(np.isfinite(m)))
    assert (m.shape == (11, 11, 1))


def test_CSInterpolator_out_of_bounds():
    """Tests if the interpolator segfaults on out of range data"""
    from PYME.localization.FitFactories.Interpolators.CSInterpolator import interpolator
    interpolator.setModelFromFile(os.path.join(os.path.dirname(Test.__file__), 'astig_theory.tif'))
    
    roiHalfSize = 5
    x, y = 10., 10.
    
    X, Y, Z, safeRegion = interpolator.getCoords(TIRFDefault, slice(x - roiHalfSize, x + roiHalfSize + 1),
                                                 slice(y - roiHalfSize, y + roiHalfSize + 1), slice(0, 1))
    
    for i in range(100):
        try:
            m = f_Interp3d([1, x, y+1e3*np.random.normal(), 1e3*np.random.normal()], interpolator, X+ 1e3*np.random.normal(), Y+ 1e3*np.random.normal() , Z + 1e3*np.random.normal(), safeRegion)
            assert (np.all(np.isfinite(m)))
            assert (m.shape == (11, 11, 1))
        except RuntimeError:
            pass


def test_LinearInterpolator():
    from PYME.localization.FitFactories.Interpolators.LinearInterpolator import interpolator
    
    interpolator.setModelFromFile(os.path.join(os.path.dirname(Test.__file__), 'astig_theory.tif'))

    roiHalfSize = 5
    x, y = 0., 0.
    
    X, Y, Z, safeRegion = interpolator.getCoords(TIRFDefault, slice(x - roiHalfSize, x + roiHalfSize + 1),
                                                 slice(y - roiHalfSize, y + roiHalfSize + 1), slice(0, 1))
    
    m = f_Interp3d([1, x, y, 0], interpolator, X, Y, Z, safeRegion)
    
    
    assert (np.all(np.isfinite(m)))
    assert (m.shape == (11,11,1))


# def test_LinearPInterpolatorP():
#     from .LinearPInterpolatorP import interpolator
#
#     interpolator.setModelFromFile(os.path.join(os.path.dirname(Test.__file__), 'astig_theory.tif'))
#
#
#     roiHalfSize = 5
#     x, y = 0, 0
#
#     X, Y, Z, safeRegion = interpolator.getCoords(TIRFDefault, slice(x - roiHalfSize, x + roiHalfSize + 1),
#                                                  slice(y - roiHalfSize, y + roiHalfSize + 1), slice(0, 1))
#
#     m = f_Interp3d([1, x, y, 0], interpolator, X, Y, Z, safeRegion)
#
#     print m.shape
#
#     assert (np.all(np.isfinite(m)))
#     assert (m.shape == (11, 11, 1))
    