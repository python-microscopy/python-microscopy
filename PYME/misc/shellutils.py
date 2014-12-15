from PYME.Analysis.piecewise import piecewiseLinear
import matplotlib.pylab as plt
import numpy as np
import math

def getvar(varname, inmodule = False):
    import inspect
    frame = inspect.currentframe()
    gframe = frame.f_back
    # go up one level further if called within module
    if inmodule:
        gframe = gframe.f_back

    var = None
    try: # first for dh5view
        var = gframe.f_locals[varname]
    except:
        pass
    return var

def getmdh(inmodule = False):
    import inspect
    frame = inspect.currentframe()
    gframe = frame.f_back
    # go up one level further if called within module
    if inmodule:
        gframe = gframe.f_back

    mp = None
    try: # first for dh5view
        mp = gframe.f_locals['mdv']
    except:
        try: # alternatively VisGui
            mp = gframe.f_locals['mdp']
        except:
            pass
    finally:
        del frame

    if mp is not None:
        return mp.mdh
    else:
        return None

from PYME.Acquire import MetaDataHandler
def mdhnogui(filename):
    import tables
    h5f = tables.openFile(filename)
    mdh = MetaDataHandler.HDFMDHandler(h5f)
    return {'h5f': h5f, 'mdh': mdh}

def getDriftPars(mdh=None):
    if mdh is None:
        mdh = getmdh(inmodule=True)
    try:
        dc = mdh['DriftCorrection']
    except:
        print 'could not find DriftCorrection info'
        return None
    else:
        print 'found drift correction info'

    # estimate the number of frames or fall back to default
    try:
        frames = (mdh['Source.EndTime']-mdh['Source.StartTime'])/mdh['Source.Camera.CycleTime']
    except:
        frames = 2e4

    exec('pars = %s' % dc['Parameters'])
    a0,a1,a2,a3,a4 = [pars[v] for v in ['a0','a1','a2','a3','a4']]
    b0,b1,b2,b3,b4 = [pars[v] for v in ['b0','b1','b2','b3','b4']]

    t = np.arange(frames)
    x = 0
    y = 0
    exec ('x = %s' % dc['ExprX'])
    exec ('y = %s' % dc['ExprY'])

    plt.plot(t,x)
    plt.plot(t,y)
    plt.show()

    return pars

def findSlide(mdh=None):
    if mdh is None:
        mdh = getmdh(inmodule=True)

    try:
        slideref = mdh['Source.Sample.SlideRef']
    except:
        return None

    from PYME.Acquire import sampleInformation
    from PYME.SampleDB.samples import models

    matches = models.Slide.objects.filter(reference__contains=slideref)
    slide = matches[0]
    return slide

# from PYME.SampleDB.samples import models
# qs3 = models.Slide.objects.filter(reference__contains='22_7_10_C')
# qs2 = models.Slide.objects.filter(slideID__exact=-1394421344L)
# sample=qs2[0].sample
# sample.sampleType
# sample.species

# generate a default basename
def defaultbase():
    import os.path
    image = getvar('image',inmodule=True)
    if image is None:
        print 'could not find image'
        return
    return os.path.splitext(os.path.basename(image.filename))[0]

def saveSelection(fname):
    do = getvar('do',inmodule=True)
    if do is None:
        print 'could not find display object'
        return
    lx, ly, hx, hy = do.GetSliceSelection()
    image = getvar('image',inmodule=True)
    if image is None:
        print 'could not find image'
        return
    filen = image.filename

    print 'source file %s' % (filen)
    print 'selection ', (lx,ly,hx,hy)
    f = open(fname,'w')
    f.write("%s\n" % filen)
    for item in (lx,ly,hx,hy):
        f.write("%d\t" % item)
    f.write("\n")
    f.flush()
    f.close

def writecoords(filename,pipeline):
    n = pipeline['x'].shape[0]
    f = open(filename,'w')
    minx = pipeline['x'].min()
    miny = pipeline['y'].min()
    for i in range(n):
        f.write("%.3f %.3f %d\n" % (pipeline['x'][i]-minx,pipeline['y'][i]-miny,pipeline['t'][i]))
    
    f.close()

def randmapping(pipeline):
    pipeline.selectedDataSource.setMapping('rand1','0*x+np.random.rand(x.size)')

def binSum(binVar, indepVar, bins):
    bm = np.zeros(len(bins) - 1,dtype = indepVar.dtype)
    bs = np.zeros(len(bins) - 1)
    bn = np.zeros(len(bins) - 1, dtype='i')

    for i, el, er in zip(range(len(bm)), bins[:-1], bins[1:]):
        v = indepVar[(binVar >= el)*(binVar < er)]

        bn[i] = len(v)
        if bn[i] == 0:
            bm[i] = 0
            bs[i] = 0
        else:
            bm[i] = v.sum()
            bs[i] = v.std()

    return bn, bm, bs

def frc(image):
    from PYME.Analysis import binAvg        
    import numpy as np
    import pylab

    voxelsize = image.voxelsize

    shape = image.data.shape[0:2]
    hwin = np.sqrt(np.outer(np.hanning(shape[0]),np.hanning(shape[1])))

        #assume we have exactly 2 channels #FIXME - add a selector
        #grab image data
    imA = hwin * image.data[:,:,:,0].squeeze()
    imB = hwin * image.data[:,:,:,1].squeeze()
        
    X, Y = np.mgrid[0:float(imA.shape[0]), 0:float(imA.shape[1])]
    X = X/X.shape[0]
    Y = Y/X.shape[1]
    X = X - .5
    Y = Y - .5
    R = np.sqrt(X**2 + Y**2)
    
    H1 = pylab.fft2(imA)
    H2 = pylab.fft2(imB)
    
    ringwidth  = 1 # in pixels
    rB = np.linspace(0,0.5,0.5*imA.shape[0]/ringwidth)
    
    bn, bm, bs = binSum(R, pylab.fftshift(H1*H2.conjugate()), rB)
    
    bn1, bm1, bs1 = binSum(R, pylab.fftshift(abs(H1*H1.conjugate())), rB)
    bn2, bm2, bs2 = binSum(R, pylab.fftshift(abs(H2*H2.conjugate())), rB)
    
    bmr = np.real(bm)
    
    
    pylab.figure()
    
    ax = pylab.gca()

    freqpnm = rB/voxelsize[0]
    ax.plot(freqpnm[:-1], bmr/np.sqrt(bm1*bm2))
    ax.plot(freqpnm[:-1], 2./np.sqrt(bn/2))
    ax.plot(freqpnm[:-1], 0*bmr + 1.0/7)
    ax.plot(freqpnm[:-1], 0*bmr, '--')
    
    xt = np.array([10., 15, 20, 30, 50, 80, 100, 150])
    rt = 1.0/xt
    
    pylab.xticks(rt[::-1],['%d' % xi for xi in xt[::-1]])

    pylab.show()

    return H1, H2, R, bmr/np.sqrt(bm1*bm2), bn, bm, bm1, bm2, rB

def abscorrel(a,b):
    from scipy.fftpack import fftn, ifftn
    from pylab import fftshift, ifftshift
    import numpy as np

    F0 = fftn(a)
    Fi = ifftn(b)
    corr = abs(fftshift(ifftn(F0*Fi)))

    return corr

def cent2d(im):
    im -= im.min()
    
    im = np.maximum(im - im.max()*.75, 0)
                
    xi, yi = np.where(im)
    
    im_s = im[im>0]
    im_s/= im_s.sum()
    
    dxi =  ((xi*im_s).sum() - im.shape[0]/2.)
    dyi =  ((yi*im_s).sum() - im.shape[1]/2.)

    return [dxi,dyi]

def trackser(ref, series, frange=None):
    if frange is None:
        frange = range(series.shape[2])
    nframes = len(frange)

    dx = np.zeros(nframes)
    dy = np.zeros(nframes)
    for i in range(nframes):
        corr = abscorrel(ref,series[:,:,frange[i]].squeeze())
        dxi, dyi = cent2d(corr)
        dx[i] = dxi
        dy[i] = dyi

    return [dx,dy]

import matplotlib.pyplot as plt
def savitzky_golay(y, window_size, order, deriv=0):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    
    This code has been taken from http://www.scipy.org/Cookbook/SavitzkyGolay
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.savefig('images/golay.png')
    #plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')

from PYME.Analysis.BleachProfile import kinModels
def gphotons(pipeline):
    colourFilter = pipeline.colourFilter
    metadata = pipeline.mdh
    chans = colourFilter.getColourChans()
    channame = ''
    if len(chans) == 0:
        nph = kinModels.getPhotonNums(colourFilter, metadata)
        merr = colourFilter['error_x']
        return [channame, nph.mean(), merr.mean()]
    ret = []
    curcol = colourFilter.currentColour
    for chan in chans:
        channame = pipeline.fluorSpeciesDyes[chan]
        colourFilter.setColour(chan)
        nph = kinModels.getPhotonNums(colourFilter, metadata)
        merr = colourFilter['error_x']
        ret.append([channame,nph.mean(),merr.mean()])

    colourFilter.setColour(curcol)
    return ret

def photonconvert(data,mdh=None):
    if mdh is None:
        mdh = getmdh(inmodule=True)
    return (data-mdh['Camera.ADOffset'])*mdh['Camera.ElectronsPerCount']/mdh['Camera.TrueEMGain']

def gmesig(sig,N,Nb,voxelsize):
    siga = np.sqrt(sig*sig+voxelsize*voxelsize/12.0)
    return siga*siga/N*(16.0/9+8*math.pi*siga*siga*Nb/(N*voxelsize*voxelsize))

def gmestd(sig,N,Nb,voxelsize,mdh=None):
    return np.sqrt(gmesig(sig,N,Nb,voxelsize))
