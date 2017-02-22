import numpy as np

from scipy import ndimage
from scipy import stats

def gen_image(p=.95, disp=False):
    from PYME.Acquire.Hardware.Simulator import wormlike2

    wc = wormlike2.wiglyFibre(.5e3, 1e2, 1)
    x = np.mod(wc.xp, 64)
    y = np.mod(wc.yp, 66)
    #mtim, xb, yb = np.histogram2d(x, y, [arange(0, 64), arange(0, 64)])

    #print 'Density = ', (1-p)/(.07**2), '/um^2'
    mt = (np.random.rand(len(x)) > p)
    #imshow(m)
    m = np.histogram2d(x[mt], y[mt], [np.arange(0, 64), np.arange(0, 64)])[0].astype('f')
    
    #im = ndimage.gaussian_filter(m, 2)
    #imshow(im)
    
    im = ndimage.gaussian_filter(500 * m * (np.random.poisson(4, m.shape) + 1), 2)
    im2 = np.random.poisson(im + 10)
    
    #if m.sum() == 0:
    #    return 0, 0, 0, 0
    #xc, yc = np.where(m)
    
    #savefig('nr_roi_nr.pdf')
    
    # l, nl = ndimage.label(im > 7)
    # nm2 = np.array([(m.astype('f') * (l == i)).sum() for i in (np.arange(nl) + 1)])
    # nn = (xc[:, None] < (xc + 9)[None, :]) * (xc[:, None] > (xc - 9)[None, :]) * (yc[:, None] < (yc + 9)[None, :]) * (
    # yc[:, None] > (yc - 9)[None, :])
    # nm = nn.sum(0)
    # #print (nm**2).sum(), (nm2**2).sum(), nm, nm2
    # rs = np.array([(l == i).sum() for i in (np.arange(nl) + 1)])
    
    #imshow(l)
    #figure()
    #View3D(m.T)
    #print m.max()
    #print (rs*nm2**2).sum()/ (11*11*nm**2).sum()
    return x[mt], y[mt], im2#np.array([(rs * nm2 ** 2).sum(), (11 * 11 * nm ** 2).sum(), nm2.mean(), nm.mean()])



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