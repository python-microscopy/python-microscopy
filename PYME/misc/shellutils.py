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

def _mcheck(mdh,key):
    if key in mdh.keys():
        return mdh[key]
    else:
        return None 

def _tformat(timeval):
    import time
    if timeval < 946684800: # timestamp for year 2000 as heuristic
        return timeval
    else:
        return "%s (%s)" % (timeval,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timeval)))

def seriestiming(mdh=None):
    if mdh is None:
        mdh = getmdh(inmodule=True)
    tinfo = {}
    if _mcheck(mdh,'StartTime'):
        tinfo['start'] = mdh['StartTime']
        tinfo['end'] = _mcheck(mdh,'EndTime')
    elif _mcheck(mdh,'Source.StartTime'):
        tinfo['start'] = mdh['Source.StartTime']
        tinfo['end'] = _mcheck(mdh,'Source.EndTime')
    else:
        print "no timing info found"
        return
    print "Start\t\t%s" % _tformat(tinfo['start'])
    if tinfo['end']:
        print "End\t\t\t%s" % _tformat(tinfo['end'])
        print "Duration\t%.2f s (%.1f min)" % (tinfo['end']-tinfo['start'],(tinfo['end']-tinfo['start'])/60.0)

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

def visguiDriftPlot(driftpane):
    parameterNames, indepVarsUsed, xCode, yCode, zCode , parameterNamesZ, varExpandCode, varExpandCodeZ = driftpane.dp.driftCorrFcn

    indepVars = driftpane.visFr.filter
    #t = np.linspace(indepVars['t'].min(), indepVars['t'].max())

    x = 0
    y = 0

    driftx=driftpane.visFr['driftx']
    drifty=driftpane.visFr['drifty']
    t = driftpane.visFr['t']

    p = [driftpane.dp.driftCorrParams[pn] for pn in parameterNames]

    exec(varExpandCode)

    x1 = eval(xCode)
    y1 = eval(yCode)

    x = driftpane.visFr['x']
    y = driftpane.visFr['y']

    xs = np.mean(x[0:10])
    ys = np.mean(y[0:10])
    x1s = np.mean(x1[0:10])
    y1s = np.mean(y1[0:10])
    
    plt.figure(1)
    plt.clf()
    p1,=plt.plot(t,x1-x1s,label='drift track X')
    p2,=plt.plot(t,x-xs,label='X raw')
    plt.legend(handles=[p1,p2])

    plt.figure(2)
    plt.clf()
    p1,=plt.plot(t,y1-y1s,label='drift track Y')
    p2,=plt.plot(t,y-ys,label='Y raw')
    plt.legend(handles=[p1,p2])
    plt.show()
    
    return (x1,y1,x,y)

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
    px = pipeline['x']
    py = pipeline['y']
    pt = pipeline['t']
    minx = px.min()
    miny = py.min()
    for i in range(n):
        f.write("%.3f %.3f %d\n" % (px[i]-minx,py[i]-miny,pt[i]))
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

import PYME.Analysis.BleachProfile.kinModels as km
def plotphotons(pipeline,color='red'):
    nph = km.getPhotonNums(pipeline.colourFilter,pipeline.mdh)
    ph_range = 6*nph.mean()
    n, bins = np.histogram(nph, np.linspace(0, ph_range, 100))
    plt.bar(bins[:-1], n, width=bins[1]-bins[0], alpha=0.4, color=color)
    return nph

def photonconvert(data,mdh=None):
    if mdh is None:
        mdh = getmdh(inmodule=True)
    return (data-mdh['Camera.ADOffset'])*mdh['Camera.ElectronsPerCount']/mdh['Camera.TrueEMGain']

def gmesig(sig,N,Nb,voxelsize):
    siga = np.sqrt(sig*sig+voxelsize*voxelsize/12.0)
    return siga*siga/N*(16.0/9+8*math.pi*siga*siga*Nb/(N*voxelsize*voxelsize))

def gmestd(sig,N,Nb,voxelsize,mdh=None):
    return np.sqrt(gmesig(sig,N,Nb,voxelsize))

# histogram with binwidth guaranteed to be one
def histone(data,binwidth=1):
    d=data.squeeze()
    plt.hist(d, bins=range(int(min(d)), int(max(d)) + binwidth, binwidth))

# this routine is designed for correlative tracking using the first (potentially averaged) image
# as reference and then determines displacement of later images aginst that one
import scipy.ndimage
from numpy.fft import *
def correltrack(data,start=0,avgover=10,pixelsize=70.0,centersize=7,centroidfrac=1.5):
    cs = centersize
    shp = [d for d in data.shape if d > 1]
    nsteps = long((shp[2]-start)/avgover)
    shh = (shp[0]/2,shp[1]/2)
    xctw=np.zeros((2*centersize+1,2*centersize+1,nsteps))
    shifts = []
    i1 = data[:,:,start:start+avgover].squeeze().mean(axis=2)
    I1 = fftn(i1)
    for i in range(nsteps):
        xc = abs(ifftshift(ifftn(I1*ifftn(data[:,:,start+i*avgover:start+(i+1)*avgover].squeeze().mean(axis=2)))))
        xct = xc-xc.min()
        xct = (xct-xct.max()/centroidfrac)*(xct > xct.max()/centroidfrac)
        xctw[:,:,i] = xct[shh[0]-cs:shh[0]+cs+1,shh[1]-cs:shh[1]+cs+1]
        shifts.append(scipy.ndimage.measurements.center_of_mass(xctw[:,:,i]))

    sh = np.array(shifts)
    t = start + np.arange(nsteps)*avgover
    sh = pixelsize*(sh-sh[0])
    return t, sh, xctw

def meanvards(dataSource, start=0, end=-1):

    nslices = dataSource.getNumSlices()
    if end < 0:
        end = nslices + end

    nframes = end - start
    xSize, ySize = dataSource.getSliceShape()

    m = np.zeros((xSize,ySize),dtype='float64')
    for frameN in range(start,end):
        m += dataSource.getSlice(frameN)
    m = m / nframes

    v = np.zeros((xSize,ySize),dtype='float64')
    for frameN in range(start,end):
        v += (dataSource.getSlice(frameN)-m)**2
    v = v / (nframes-1)

    return (m,v)

def darkCal(dataSource, integrationTimes,transitionTimes):
    ms = []
    vs = []
    endTimes = transitionTimes[1:]+[-1]
    for istart, istop in zip(transitionTimes, endTimes):
        print "starting at %d, using %d frames..." % (istart,istop-istart)
        m, v = meanvards(dataSource,istart,istop)
        ms.append(m)
        vs.append(v)
    return (ms,vs)

def darkCalfromMetadata(dataSource,mdh=None):
    if mdh is None:
        mdh = getmdh(inmodule=True)
    it,tt = (mdh['Protocol.IntegrationTimes'],mdh['Protocol.Transitions'])
    ms, vs = darkCal(dataSource,it,tt)
    return (ms,vs,it)

from scipy import stats
def isnparray(a):
    return type(a).__module__ == np.__name__
    
def dcfit(ms,integrationTimes):
    import sys
    if not isnparray(ms):
        ofs = np.dstack(ms) # offsets
    else:
        ofs = ms
    itimes = np.asarray(integrationTimes)

    sz = ofs.shape
    sz2d = sz[0:2]
    def z2d():
        return np.zeros(sz2d,dtype = 'float32')
    dc = z2d()
    offs = z2d()
    r_value = z2d()
    p_value = z2d()
    std_err = z2d()
    for x in range(sz[0]):
        print "line %d" % (x) + '\r',
        sys.stdout.flush()
        for y in range(sz[1]):
            dc[x,y], offs[x,y], r_value[x,y], p_value[x,y], std_err[x,y] = \
            stats.linregress(itimes,ofs[x,y,:])

    return (dc,offs,r_value,p_value,std_err)

def subsampidx(arraylen, percentage=10):
    newlen = percentage*1e-2*arraylen
    idx = np.random.choice(arraylen,newlen)
    return idx

from scipy.stats import gaussian_kde
import pylab
def scatterdens(x,y,subsample=1.0):
    xf = x.flatten()
    yf = y.flatten()
    if subsample < 1.0:
        idx = subsampidx(xf.size,percentage = 100*subsample)
        xs = xf[idx]
        ys = yf[idx]
    else:
        xs = xf
        ys = yf
        
    estimator = gaussian_kde([xs,ys]) 
    density = estimator.evaluate([xf,yf])
    pylab.scatter(xf,yf,c=density,marker='o',linewidth='0',zorder=3,s=40)
