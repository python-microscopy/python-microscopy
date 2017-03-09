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
    from .CSInterpolator import interpolator
    
    interpolator.setModelFromFile(os.path.join(os.path.dirname(Test.__file__), 'astig_theory.psf'))
    
    print interpolator.interpModel.min(), interpolator.interpModel.max()
    print interpolator.interpModel.flags
    
    roiHalfSize = 5
    x, y = 10.,10.

    X, Y, Z, safeRegion = interpolator.getCoords(TIRFDefault, slice(x - roiHalfSize, x + roiHalfSize + 1),
                                                 slice(y - roiHalfSize, y + roiHalfSize + 1), slice(0, 1))

    m = f_Interp3d([1,x,y,0], interpolator, X, Y, Z, safeRegion)
    
    #print m
    
    assert(np.all(np.isfinite(m)))


def test_LinearInterpolator():
    from .LinearInterpolator import interpolator
    
    interpolator.setModelFromFile(os.path.join(os.path.dirname(Test.__file__), 'astig_theory.psf'))
    
    #print np.where(interpolator.interpModel == interpolator.interpModel.max())
    
    #print interpolator.interpModel.shape
    
    #print interpolator.interpModel[interpolator.interpModel.shape[0]/2, :, 40]

    roiHalfSize = 5
    x, y = 0., 0.
    
    X, Y, Z, safeRegion = interpolator.getCoords(TIRFDefault, slice(x - roiHalfSize, x + roiHalfSize + 1),
                                                 slice(y - roiHalfSize, y + roiHalfSize + 1), slice(0, 1))
    
    #print X, Y, Z
    
    m = f_Interp3d([1, x, y, 0], interpolator, X, Y, Z, safeRegion)
    
    #print m
    #print m.shape
    
    
    assert (np.all(np.isfinite(m)))
    assert (m.shape == (11,11,1))


# def test_LinearPInterpolatorP():
#     from .LinearPInterpolatorP import interpolator
#
#     interpolator.setModelFromFile(os.path.join(os.path.dirname(Test.__file__), 'astig_theory.psf'))
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
    