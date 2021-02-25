from PYME.Analysis.piecewise import piecewiseLinear
import matplotlib.pyplot as plt
import numpy as np
import math


# should we do this with f_globals instead of f_locals?
# function to be used at shell prompt in PYME GUI shells
# can be used to search through the variable names available in the shell
def grepglobals2(expr, keyonly = False):
    # instead of passing globals can we get this via inspect?
    import inspect
    frame = inspect.currentframe()
    gframe = frame.f_back
    
    res = filter(lambda x: expr in x, gframe.f_locals)
    if len(res)> 0:
        if keyonly:
            return res
        else:
            return { key: gframe.f_locals[key] for key in res }
    else:
        return None

# convert paste board to UTF-16 little endian
# this is what pasting into the shell tab appears to require
def convertpb2utf16le():
    import os
    from sys import platform
    if platform == "darwin":
        os.system("pbpaste | iconv -f ascii -t utf-16le | pbcopy")
    else:
        raise RuntimeError('function only available on mac')
    
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

from PYME.IO import MetaDataHandler
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
        print("no timing info found")
        return
    print("Start\t\t%s" % _tformat(tinfo['start']))
    if tinfo['end']:
        print("End\t\t\t%s" % _tformat(tinfo['end']))
        print("Duration\t%.2f s (%.1f min)" % (tinfo['end']-tinfo['start'],(tinfo['end']-tinfo['start'])/60.0))

def getDriftPars(mdh=None):
    if mdh is None:
        mdh = getmdh(inmodule=True)
    try:
        dc = mdh['DriftCorrection']
    except:
        print('could not find DriftCorrection info')
        return None
    else:
        print('found drift correction info')

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
    x_raw=driftpane.visFr['x_raw']
    y_raw=driftpane.visFr['y_raw']
    t = driftpane.visFr['t']

    p = [driftpane.dp.driftCorrParams[pn] for pn in parameterNames]

    exec(varExpandCode)

    x1 = eval(xCode)
    y1 = eval(yCode)

    x = driftpane.visFr['x']
    y = driftpane.visFr['y']

    xs = np.mean(x[0:50])
    ys = np.mean(y[0:50])
    
    plt.figure(1)
    plt.clf()
    #plt.plot(t,-x1)
    plt.plot(t,x-xs)
    plt.figure(2)
    plt.clf()
    
    #plt.plot(t,-y1)
    plt.plot(t,y-ys)
    plt.show()
    
    return (x1,y1,x,y)

def findSlide(mdh=None):
    if mdh is None:
        mdh = getmdh(inmodule=True)

    try:
        slideref = mdh['Source.Sample.SlideRef']
    except:
        return None

    from PYME.Acquire import sampleInformationDjangoDirect as sampleInformation
    from PYME.SampleDB2.samples import models

    matches = models.Slide.objects.filter(reference__contains=slideref)
    slide = matches[0]
    return slide

# from PYME.SampleDB.samples import models
# qs3 = models.Slide.objects.filter(reference__contains='22_7_10_C')
# qs2 = models.Slide.objects.filter(slideID__exact=-1394421344L)
# sample=qs2[0].sample
# sample.sampleType
# sample.species

def imagestats():
    import math
    import scipy.ndimage as nd
    image = getvar('image',inmodule=True)
    if image is None:
        print('could not find image')
        return
    do = getvar('do',inmodule=True)
    if do is None:
        print('could not find display object')
        return
    
    data = image.data[:,:,do.zp].squeeze()
    dmed = nd.median(data)
    
    print("mean:\t\t%f" % data.mean())
    print("variance:\t%f" % data.var())
    print("std dev:\t%f" % data.std())
    print("median:\t\t%f" % dmed)
    print("med-sqrt:\t%f" % math.sqrt(dmed))

# generate a default basename
def defaultbase():
    import os.path
    image = getvar('image',inmodule=True)
    if image is None:
        print('could not find image')
        return
    return os.path.splitext(os.path.basename(image.filename))[0]

def saveSelection(fname):
    do = getvar('do',inmodule=True)
    if do is None:
        print('could not find display object')
        return
    lx, ly, hx, hy = do.GetSliceSelection()
    image = getvar('image',inmodule=True)
    if image is None:
        print('could not find image')
        return
    filen = image.filename

    print('source file %s' % (filen))
    print('selection ', (lx,ly,hx,hy))
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


def csvcoords(filename,pipeline,keys,fieldnames=None):
    import csv
    if fieldnames is None:
        fieldnames = keys
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='#', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        pkeys = [pipeline[key] for key in keys] # cache the pipelines as these calls may be costly
        n = pipeline['x'].shape[0]
        for i in range(n):
            writer.writerow([pkey[i] for pkey in pkeys])

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
    # import pylab
    import matplotlib.pyplot as plt
    from numpy.fft import fft2, fftshift

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
    
    H1 = fft2(imA)
    H2 = fft2(imB)
    
    ringwidth  = 1 # in pixels
    rB = np.linspace(0,0.5,0.5*imA.shape[0]/ringwidth)
    
    bn, bm, bs = binSum(R, fftshift(H1*H2.conjugate()), rB)
    
    bn1, bm1, bs1 = binSum(R, fftshift(abs(H1*H1.conjugate())), rB)
    bn2, bm2, bs2 = binSum(R, fftshift(abs(H2*H2.conjugate())), rB)
    
    bmr = np.real(bm)
    
    
    plt.figure()
    
    ax = plt.gca()

    freqpnm = rB/voxelsize[0]
    ax.plot(freqpnm[:-1], bmr/np.sqrt(bm1*bm2))
    ax.plot(freqpnm[:-1], 2./np.sqrt(bn/2))
    ax.plot(freqpnm[:-1], 0*bmr + 1.0/7)
    ax.plot(freqpnm[:-1], 0*bmr, '--')
    
    xt = np.array([10., 15, 20, 30, 50, 80, 100, 150])
    rt = 1.0/xt
    
    plt.xticks(rt[::-1],['%d' % xi for xi in xt[::-1]])

    plt.show()

    return H1, H2, R, bmr/np.sqrt(bm1*bm2), bn, bm, bm1, bm2, rB

def abscorrel(a,b):
    from numpy.fft import fftn, ifftn, fftshift, ifftshift
    # from pylab import fftshift, ifftshift
    import numpy as np

    F0 = fftn(a)
    Fi = ifftn(b)
    corr = abs(fftshift(ifftn(F0*Fi)))

    return corr

def cent2d(im,usefrac=0.25):
    im -= im.min()
    
    im = np.maximum(im - im.max()*(1.0-usefrac), 0)
                
    xi, yi = np.where(im)
    
    im_s = im[im>0]
    im_s/= im_s.sum()
    
    dxi =  ((xi*im_s).sum() - im.shape[0]/2.)
    dyi =  ((yi*im_s).sum() - im.shape[1]/2.)

    return [dxi,dyi]

def trackser(ref, series, frange=None, usefrac=0.25):
    if frange is None:
        frange = range(series.shape[2])
    nframes = len(frange)

    dx = np.zeros(nframes)
    dy = np.zeros(nframes)
    for i in range(nframes):
        corr = abscorrel(ref,series[:,:,frange[i]].squeeze())
        dxi, dyi = cent2d(corr,usefrac=usefrac)
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
    except ValueError:
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

# we ignore centroidfrac by default
def correltrack2(data,start=0,avgover=10,pixelsize=70.0,centersize=15,centroidfac=0.6,roi=[0,None,0,None]):
    cs = centersize
    shp = [d for d in data.shape if d > 1]
    nsteps = long((shp[2]-start)/avgover)
    xctw=np.zeros((2*centersize+1,2*centersize+1,nsteps))
    shifts = []
    if avgover > 1:
        ref = data[:,:,start:start+avgover].squeeze().mean(axis=2)
    else:
        ref = data[:,:,start].squeeze()
    ref = ref[roi[0]:roi[3],roi[1]:roi[3]]
    refn = ref/ref.mean() - 1
    Frefn = fftn(refn)
    shh = (ref.shape[0]/2,ref.shape[1]/2)

    for i in range(nsteps):
        comp = data[:,:,start+i*avgover:start+(i+1)*avgover].squeeze()
        if len(comp.shape) > 2:
            comp = comp.mean(axis=2)
        comp = comp[roi[0]:roi[3],roi[1]:roi[3]]
        compn = comp/comp.mean() - 1
        xc = ifftshift(np.abs(ifftn(Frefn*ifftn(compn))))
        xcm = xc.max()
        xcp = np.maximum(xc - centroidfac*xcm, 0)
        xctw[:,:,i] = xcp[shh[0]-cs:shh[0]+cs+1,shh[1]-cs:shh[1]+cs+1]
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
        print("starting at %d, using %d frames..." % (istart,istop-istart))
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
        print("line %d" % (x) + '\r',)
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
def scatterdens(x,y,subsample=1.0, s=40, xlabel=None, ylabel=None, **kwargs):
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
    print("density min, max: %f, %f" % (density.min(), density.max()))
    plt.scatter(xf,yf,c=density,marker='o',linewidth='0',zorder=3,s=s,**kwargs)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
        
    return estimator

def multicolcheck(pipeline,subsample=0.03,dA=20,xrange=[-1000,3000],yrange=[-1000,6000]):
    p = pipeline
    plt.figure()
    plt.subplot(1, 2, 1)
    estimator = scatterdens(p['fitResults_Ag'],p['fitResults_Ar'],subsample=subsample,s=10)
    plt.xlim(xrange)
    plt.ylim(yrange)
    
    x1d = np.arange(xrange[0],xrange[1],dA)
    y1d = np.arange(yrange[0],yrange[1],dA)
    x2d = x1d[:,None] * np.ones_like(y1d)[None,:]
    y2d = np.ones_like(x1d)[:,None] * y1d[None,:]
    
    imd = estimator.evaluate([x2d.flatten(),y2d.flatten()])
    imd2d = imd.reshape(x2d.shape)
    imd2d /= imd2d.max()

    #plt.figure()
    plt.subplot(1, 2, 2)
    plt.imshow(imd2d[:,::-1].transpose(),cmap=plt.get_cmap('jet'),extent=[xrange[0],xrange[1],yrange[0],yrange[1]])
    plt.grid(True)
    
    return imd2d

def intdens(image,framenum=0):
    mdh = image.mdh
    pixarea = 1e6*mdh['voxelsize.x']*mdh['voxelsize.y']
    data = image.data[:,:,framenum].squeeze()

    intdens = float(pixarea*data.sum())
    nevts = None
    try:
        nevts = int(mdh['Rendering.NEventsRendered'])
    except:
        pass
    if nevts is not None:
        print("Ratio Events/Intdens = %f" % (nevts/intdens))
    return intdens

def px(p):
    t = p['t']*p.mdh['Camera.CycleTime']
    x = p['x']-p['x'][0:10].mean()
    plt.plot(t,x)

def py(p):
    t = p['t']*p.mdh['Camera.CycleTime']
    y = p['y']-p['y'][0:10].mean()
    plt.plot(t,y)


def cumuexpfit(t,tau):
    return 1-np.exp(-t/tau)

from scipy.optimize import curve_fit
def darktimes(pipeline, mdh=None, plot=True, report=True):
    if mdh is None:
        mdh = getmdh(inmodule=True)
    t = pipeline['t']
    x = pipeline['x']
    y = pipeline['y']
    # determine darktime from gaps and reject zeros (no real gaps) 
    dts = t[1:]-t[0:-1]-1
    dtg = dts[dts>0]
    nts = dtg.shape[0]
    # now make a cumulative histogram from these
    cumux = np.sort(dtg+0.01*np.random.random(nts)) # hack: adding random noise helps us ensure uniqueness of x values
    cumuy = (1.0+np.arange(nts))/np.float(nts)
    bbx = (x.min(),x.max())
    bby = (y.min(),y.max())
    voxx = 1e3*mdh['voxelsize.x']
    voxy = 1e3*mdh['voxelsize.y']
    bbszx = bbx[1]-bbx[0]
    bbszy = bby[1]-bby[0]
    maxtd = dtg.max()
    binedges = np.arange(0,maxtd,5)
    binctrs = 0.5*(binedges[0:-1]+binedges[1:])
    h,be2 = np.histogram(dtg,bins=binedges)
    hc = np.cumsum(h)
    hcg = hc[h>0]/float(nts) # only nonzero bins and normalise
    binctrsg = binctrs[h>0]
    popth,pcovh = curve_fit(cumuexpfit,binctrsg,hcg, p0=(300.0))
    popt,pcov = curve_fit(cumuexpfit,cumux,cumuy, p0=(300.0))
    if plot:
        plt.subplot(211)
        plt.plot(cumux,cumuy,'o')
        plt.plot(cumux,cumuexpfit(cumux,popt[0]))
        plt.plot(binctrs,hc/float(nts),'o')
        plt.plot(binctrs,cumuexpfit(binctrs,popth[0]))
        plt.ylim(-0.2,1.2)
        plt.subplot(212)
        plt.semilogx(cumux,cumuy,'o')
        plt.semilogx(cumux,cumuexpfit(cumux,popt[0]))
        plt.semilogx(binctrs,hc/float(nts),'o')
        plt.semilogx(binctrs,cumuexpfit(binctrs,popth[0]))
        plt.ylim(-0.2,1.2)
        plt.show()
    if report:
        print("events: %d" % t.shape[0])
        print("dark times: %d" % nts)
        print("region: %d x %d nm (%d x %d pixel)" % (bbszx,bbszy,bbszx/voxx,bbszy/voxy))
        print("centered at %d,%d (%d,%d pixels)" % (x.mean(),y.mean(),x.mean()/voxx,y.mean()/voxy))
        print("darktime: %.1f (%.1f) frames" % (popt[0],popth[0]))
        print("qunits: %.2f" % (200/(popt[0]+popth[0])))

    return (cumux,cumuy,popt[0],pcov)

def darktimehist(ton):
    # determine darktime from gaps and reject zeros (no real gaps) 
    dts = ton[1:]-ton[0:-1]-1
    dtg = dts[dts>0]
    nts = dtg.shape[0]
    # now make a cumulative histogram from these
    cumux = np.sort(dtg+0.01*np.random.random(nts)) # hack: adding random noise helps us ensure uniqueness of x values
    cumuy = (1.0+np.arange(nts))/np.float(nts)
    popt,pcov = curve_fit(cumuexpfit,cumux,cumuy, p0=(300.0))
    
    return (cumux,cumuy,cumuexpfit(cumux,popt[0]),popt[0])


def analyze1dSeries(series,chunklength=500):
    offset = series.mean()
    chunks = int(len(series)/chunklength)
    chunkmaxs = np.array([max(series[chunk*chunklength:(chunk+1)*chunklength]) for chunk in range(chunks)])
    peakaverage = chunkmaxs.mean()
    offset = series[series < (offset+0.5*(peakaverage-offset))].mean()
    return (offset,peakaverage)

def datafrompipeline(datasource,pipeline, ctr, boxsize = 7):
    tser = np.arange(min(datasource.shape[2],pipeline['t'].max()))
    bszh = int(boxsize/2)
    rawser = np.zeros((2*bszh+1,2*bszh+1,tser.shape[0]))
    for t in range(len(tser)):
        ctrx = ctr[0,t]
        ctry = ctr[1,t]
        rawser[:,:,t] = datasource[int(ctrx)-bszh:int(ctrx)+bszh+1,int(ctry)-bszh:int(ctry)+bszh+1,t].squeeze()
    return (tser, rawser)

from io import StringIO
import sys
def darkAnalysisRawPlusPipeline(datasource, pipeline, driftPane=None, boxsize = 7, doplot = True,
                                threshfactor=0.45, mdh=None, debug=1):
    xp = pipeline['x'] # in new code use 'x_raw' and 'y_raw'!
    yp = pipeline['y']
    if mdh is None: # there may be other ways to get at the mdh, e.g. via pipeline?
        mdh = pipeline.mdh
    xpix = 1e3*mdh['voxelsize.x']
    ypix = 1e3*mdh['voxelsize.y']

# we need a new strategy for the pixel center selection
# and inclusion of drift
# strategy:
# 1. if we have filterkeys x and y (look up where to find these!) use the center of that ROI
# 2. if we have a drift time course calculate a centerpix(t), i.e. centerpix as a function of x

# for 1: use pipeline.filterKeys['x'] and pipeline.filterKeys['y']
# for 2: for a given xctr and yctr find the x_raw and y_raw; question: how to do that?
# for 2: we will have to get timecourse of shift as (1) x = x_raw + dx(t)
# for 2: if we manage to get (1) we will get (2) xctr_raw(t) = xctr-dx(t)

# for 2: (2) needs texting with bead sample

# for 2: once we have xctr_raw(t), yctr_raw(t) we need to modify datafrompipeline
# for 2: make datafrompipeline so that it accepts ctr(t) = [xctr(t),yctr(t)]!!

    try:
        bbox = [pipeline.filterkeys['x'][0],pipeline.filterkeys['x'][1],pipeline.filterkeys['y'][0],pipeline.filterkeys['y'][1]]
    except:
        bbox = [xp.min(),xp.max(),yp.min(),yp.max()]

    bboxpix = [bbox[0]/xpix, bbox[1]/xpix, bbox[2]/ypix, bbox[3]/ypix] # only for diagnosis 
    bbctr = 0.5*np.array([bbox[0]+bbox[1],bbox[2]+bbox[3]])
    t = np.arange(0,pipeline['t'].max())

    if driftPane is None:
        bbctrt = bbctr[:,None]*(np.ones((t.shape))[None,:])
    else:
        dx,dy,tt = getdriftcurves(driftPane,pipeline,t) # this should now return the desired times in all cases
        bbctrt = np.zeros((2,t.shape[0]))
        bbctrt[0,:] = bbctr[0]-dx
        bbctrt[1,:] = bbctr[1]-dy


    ctrpix = np.rint(bbctrt / np.array(xpix,ypix))

    if debug:  # FIXME - use logging module
        print('BBox (nm): ',bbox)
        print('BBox (pix): ',bboxpix)
        print('Ctr (pix): ',ctrpix[:,0])
        sys.stdout.flush()

    # return (bbox, bbctrt,ctrpix,t)

    print('extracting region from data...')
    sys.stdout.flush()
    tser, rawser = datafrompipeline(datasource,pipeline,ctrpix,boxsize = boxsize)
    
    print('analyzing data...')
    sys.stdout.flush()
    tevts = pipeline['t'].copy()
    rawm, peakav, fitev, fitr, rawthresh = analyzeDataPlusEvents(tser, rawser, tevts, doplot = doplot,
                                                      threshfactor=threshfactor, debug=debug)

    return (tser, rawser, rawm, peakav, tevts, fitev, fitr, rawthresh)

def analyzeDataPlusEvents(tser, rawser, tevts, doplot = True,
                          threshfactor=0.45, debug=1, rawthresh=None, size=6):
    rawm = rawser.mean(axis=0).mean(axis=0)
    offset, peakav = analyze1dSeries(rawm,chunklength=500)
    rawm = rawm-offset
    peakav = peakav-offset

    tp = tevts
    if rawthresh is None:
        rawthresh = threshfactor * peakav
    th = tser[rawm > (rawthresh)]

    ctp, chip, chipfit, taup = darktimehist(tp)
    ctr, chir, chirfit, taur = darktimehist(th)
    
    outstr = StringIO()

    outstr.write("events: %d (%d raw) \n" % (tp.shape[0],th.shape[0]))
    outstr.write("dark times: %d (%d raw) \n" % (ctp.shape[0],ctr.shape[0]))
    #print >>outstr, "region: %d x %d nm (%d x %d pixel)" % (bbszx,bbszy,bbszx/voxx,bbszy/voxy)
    #print >>outstr, "centered at %d,%d (%d,%d pixels)" % (x.mean(),y.mean(),x.mean()/voxx,y.mean()/voxy)
    outstr.write("darktime: ev %.1f (raw %.1f) frames \n" % (taup,taur))
    outstr.write("qunits: ev %.2f (raw %.2f), eunits: %.2f \n" % (200.0/taup,200.0/taur,tp.shape[0]/500.0))

    labelstr = str(outstr.getvalue())

    if debug:  # FIXME - use logging module
        print(labelstr)

    if doplot:
        plt.figure()
        plt.plot(tser, rawm)
        peaklevel = plt.plot(tser, peakav*np.ones(tser.shape), '--', label = 'median peak')
        events_h5r = plt.plot(tp, 1.2*rawthresh*np.ones(tp.shape),'o',c='red', label='events')
        events_raw = plt.plot(th, rawthresh * np.ones(th.shape),'o',c='blue', label='raw detected')
        plt.legend(handles=[events_raw[0], events_h5r[0], peaklevel[0]])
        
        plt.figure()
        events = plt.semilogx(ctp, chip, 'o', c='red', alpha=.5, markersize = size, label = 'events')
        eventfit = plt.semilogx(ctp, chipfit, label='event fit')
        raw = plt.semilogx(ctr, chir, 'o', c='blue', alpha=.5, markersize = size, label='raw')
        rawfit = plt.semilogx(ctr, chirfit, label='raw data fit')
        plt.ylim(-0.2,1.2)
        plt.annotate(labelstr, xy=(0.5, 0.1), xycoords='axes fraction',
                     fontsize=10)
        plt.legend(handles=[events[0],raw[0],eventfit[0],rawfit[0]],loc=4)
        
    return (rawm, peakav, (ctp, chip, chipfit, taup), (ctr, chir, chirfit, taur), rawthresh)


import pickle
def savepickled(object,fname):
    fi = open(fname,'wb')
    pickle.dump(object,fi)
    fi.close()

def loadpickled(fname):
    fi = open(fname,'r')
    return pickle.load(fi)

from PYME.DSView import dsviewer
def setdriftparsFromImg(driftPane,img = None):
    if img is None:
        img = dsviewer.openViewers[dsviewer.openViewers.keys()[0]].image
    driftPane.tXExpr.SetValue(img.mdh['DriftCorrection.ExprX'])
    driftPane.tYExpr.SetValue(img.mdh['DriftCorrection.ExprY'])
    driftPane.tZExpr.SetValue(img.mdh['DriftCorrection.ExprZ'])
    driftPane.OnDriftExprChange(None)
    destp = driftPane.dp.driftCorrParams
    srcp = img.mdh['DriftCorrection.Parameters']
    for key in destp.keys():
        if key.startswith(('a','b')):
            destp[key] = srcp[key]
    driftPane.OnDriftExprChange(None)
    return destp

def getOpenImages():
    img = dsviewer.openViewers
    return img

def setSelectionFromFilterKeys(visFr,img):
    glcv = visFr.glCanvas
    fk = img.mdh['Filter.Keys']
    x0,x1 = fk['x']
    y0,y1 = fk['y']

    glcv.selectionStart = (x0,y0)
    glcv.selectionFinish = (x1,y1)

import PYMEnf.DriftCorrection.compactFit as cf
def getdriftcurves(driftPane,pipeline,t=None):
    if t is None:
        t = pipeline['t']
    if 'driftx' in driftPane.dp.driftExprX:
        tt, dx = getdriftxyzFromEvts(pipeline,t,coordpos=0)
        tt, dy = getdriftxyzFromEvts(pipeline,t,coordpos=1)
        indepVars = { 't': tt, 'driftx': dx, 'drifty': dy }
    else:
        indepVars =  pipeline.filter

    dx,dy,tt = cf.xyDriftCurves(driftPane.dp.driftCorrFcn,driftPane.dp.driftCorrParams,indepVars,t)
    return (dx,dy,tt)


from PYME.Analysis import piecewiseMapping
from scipy.interpolate import interp1d

def getdriftxyzFromEvts(pipeline, tframes=None, coordpos=0):

    ts = []
    cs = []
    for e in pipeline.events[pipeline.events['EventName'] == 'ShiftMeasure']:
        ts.append(e['Time'])
        cs.append(float(e['EventDescr'].split(', ')[coordpos]))

    if len(ts) > 0:
        ts = np.array(ts)
        cs = np.array(cs)
    # convert time to frame numbers
    tfr = piecewiseMapping.times_to_frames(ts, pipeline.events, pipeline.mdh)
    # interpolate to desired frame set
    if tframes is not None:
        # finter = interp1d(tfr,cs,fill_value = 'extrapolate') # we need to check that we get no errors from this step
                                  # at the moment it will trip on extrapolation
        #csinter = finter(tframes)
        csinter = np.interp(tframes,tfr,cs)
        return (tframes,csinter)
    else:
        return(tfr,cs)

def zs(data,navg=100):
    n = min(navg,data.shape[0])
    dm = data[0:n].mean()
    return data-dm
