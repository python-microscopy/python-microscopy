import numpy as np
from scipy import ndimage

def gen_image(p=.95, disp=False):
    from PYME.Acquire.Hardware.Simulator import wormlike2

    wc = wormlike2.wiglyFibre(.5e3, 1e2, 1)
    x = np.mod(wc.xp, 64)
    y = np.mod(wc.yp, 66)
    #mtim, xb, yb = np.histogram2d(x, y, [arange(0, 64), arange(0, 64)])

    print 'Density = ', (1-p)/(.07**2), '/um^2'
    mt = (np.random.rand(len(x)) > p)
    m = np.histogram2d(x[mt], y[mt], [np.arange(0, 64), np.arange(0, 64)])[0].astype('f')
    
    im = ndimage.gaussian_filter(500 * m * (np.random.poisson(4, m.shape) + 1), 2)
    im2 = np.random.poisson(im + 10)
    
    
    return x[mt], y[mt], im2



def test_multifit():
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