import numpy as np
import pytest
from scipy import ndimage


def gen_image(p=.95, disp=False):
    from PYME.simulation import wormlike2

    wc = wormlike2.wiglyFibre(.5e3, 1e2, 1)
    x = np.mod(wc.xp, 64)
    y = np.mod(wc.yp, 66)
    #mtim, xb, yb = np.histogram2d(x, y, [arange(0, 64), arange(0, 64)])

    print('Density = ', (1-p)/(.07**2), '/um^2')
    mt = (np.random.rand(len(x)) > p)
    m = np.histogram2d(x[mt], y[mt], [np.arange(0, 64), np.arange(0, 64)])[0].astype('f')
    
    im = ndimage.gaussian_filter(500 * m * (np.random.poisson(4, m.shape) + 1), 2)
    im2 = np.random.poisson(im + 10)
    
    
    return x[mt], y[mt], im2



def test_GaussMultifitSR():
    """
    simple test to see if the multifit algorithm is working. We should detect roughly the same number of molecules
    as we simulated. This is only a loose test, and should pick up any critical reverse compatible breaks rather than
    actual fit performance.
    """
    from PYME.localization.FitFactories import GaussMultifitSR
    from PYME.IO import MetaDataHandler

    x, y, im = gen_image()

    mdh = MetaDataHandler.NestedClassMDHandler()
    mdh['Analysis.PSFSigma'] = 140.
    mdh['Analysis.ResidualMax'] = .5
    #mdh['Analysis.subtractBackground'] = False
    mdh['Camera.ReadNoise'] = 1.0
    mdh['Camera.NoiseFactor'] = 1.0
    mdh['Camera.ElectronsPerCount'] = 1.0
    mdh['Camera.TrueEMGain'] = 1.0
    mdh['voxelsize.x'] = .07
    mdh['voxelsize.y'] = .07

    ff = GaussMultifitSR.FitFactory(np.atleast_3d(im) - 2.0, mdh)
    res = ff.FindAndFit(1.8)
    
    nSim = len(x)
    nFound = len(res)
    
    print('nFound: %d, nSim: %d' %(nFound, nSim))
    assert (nFound > 0.5*nSim and nFound < 2.0*nSim)


def test_AstigGaussGPUFitFR():
    """
    simple test to see if the multifit algorithm is working. We should detect roughly the same number of molecules
    as we simulated. This is only a loose test, and should pick up any critical reverse compatible breaks rather than
    actual fit performance.
    """
    try:
        import warpdrive
    except ImportError:
        print("PYME warp drive GPU fitting not installed")
        pytest.skip('"warpdrive" GPU fitting module not installed')
        return
    
    from PYME.localization.FitFactories import AstigGaussGPUFitFR
    from PYME.IO import MetaDataHandler
    from PYME.localization.remFitBuf import CameraInfoManager

    x, y, im = gen_image()

    mdh = MetaDataHandler.NestedClassMDHandler()
    mdh['Camera.ReadNoise'] = 1.0
    mdh['Camera.NoiseFactor'] = 1.0
    mdh['Camera.ElectronsPerCount'] = 1.0
    mdh['Camera.TrueEMGain'] = 1.0
    mdh['voxelsize.x'] = 0.7
    mdh['voxelsize.y'] = 0.7
    mdh['Analysis.DetectionFilterSize'] = 3
    mdh['Analysis.ROISize'] = 4.5
    mdh['Analysis.GPUPCTBackground'] = False

    camera_info_manager = CameraInfoManager()

    fitter = AstigGaussGPUFitFR.FitFactory(np.atleast_3d(im), mdh)
    results = fitter.FindAndFit(1, cameraMaps=camera_info_manager)

    n_simulated = len(x)
    n_detected = len(results)

    print('Detected: %d, Simulated: %d' % (n_detected, n_simulated))
    assert (n_detected > 0.5 * n_simulated and n_detected < 2.0 * n_simulated)