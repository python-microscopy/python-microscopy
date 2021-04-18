#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# 
#
##################

"""To be run by invoking:
    
    python angleFilter.py <file1> <file2> <file3> etc ....
    
    note that running python angleFilter.py <adirectory>/*.tif will expand out to
    the above.
    
    expects files to be in tiff format and inverted (ie skeleton is 0, other pixels 255).
    
    The calculated histograms are output as tab formatted txt, with the columns being
    left hand bin edge (in degrees), raw count, and normalised count respectively.
    
"""
from scipy import linalg, ndimage
import numpy as np
# import pylab as pl
import matplotlib.pyplot as pl
import matplotlib.cm


def genCoords(FILT_SIZE):    
    #FILT_SIZE = 5
    x, y = np.mgrid[-np.floor(FILT_SIZE/2):np.floor(FILT_SIZE/2 +1), -np.floor(FILT_SIZE/2):np.floor(FILT_SIZE/2 +1)]
    b = np.vstack([x.ravel(), y.ravel()]).T.astype('f')
    ang = np.mod(np.angle(x +  1j*y), np.pi).ravel()
    
    return FILT_SIZE, x, y, b, ang


def th(data, FILT_SIZE, x, y, b, ang):
    return (data*ang).sum()/data.sum()

def th2(data, FILT_SIZE, x, y, b, ang):
    """calculate principle axis of ROI using SVD"""
    if (data > 0).sum() < 2:
        #not enough data to calculate PA
        return -1

    try:
        pa =  linalg.svd(data[:,None]*b, full_matrices=False)[2][0]
    except:
        print (data.shape, b.shape)
    
    return np.angle(pa[0] + 1j*pa[1])%np.pi
    
def width(data,FILT_SIZE, x, y, b, ang):
    """calculate orthogonal width of data segment using data itself
    to define principle axis"""
    if (data > 0).sum() < 2:
        #not enough data to calculate PA
        return -1
        
    pa =  linalg.svd(data[:,None]*b, full_matrices=False)[2][0]
    #print pa
    
    #generate an orthogonal axis
    #trial = np.roll(pa, 1) #gauranteed not to be co-linear with pa
    #print trial
            
    #sa = np.cross(pa.T, trial.T)
    #print sa
    sa = np.array([pa[1], -1*pa[0]])
    sa /= linalg.norm(sa)
    #print sa
    
    #define a new coordinate system
    xp = pa[0]*x + pa[1]*y
    yp = sa[0]*x + sa[1]*y
    
    #create a mask of those pixels for which xp == 0
    # this could be widened to allow some averaging along the length
    mask = (np.abs(xp) < 1).ravel()
    
    dr = data*mask
    dr /= dr.sum()
    ypr = yp.ravel()#[mask]
    
    #calculate centroid
    cent = (ypr*dr).sum()
    ypr = np.abs(ypr - cent)
    
    #calculate mean absolute distance along yp
    mad = (ypr*dr).sum()
    
    #the std. deviation formula would have squareds in here
    
    return mad
    
def width_o(data, FILT_SIZE, x, y, b, ang):
    """calculate orthogonal width of data segment based on 3D data
    where first slice is the intensities, and second slice is the angle 
    in each pixel.
    """
    data = data.reshape(FILT_SIZE, FILT_SIZE, 2)
    #if (data > 0).sum() < 2:
    #    #not enough data to calculate PA
    #    return -1
        
    #pa =  linalg.svd(data[:,None]*b, full_matrices=False)[2][0]
    pai = np.exp(1j*data[FILT_SIZE/2, FILT_SIZE/2,1])
    pa = np.array([pai.real, pai.imag])
    
    #sa = np.cross(pa.T, trial.T)
    #print sa
    sa = np.array([pa[1], -1*pa[0]])
    sa /= linalg.norm(sa)
    
    #define a new coordinate system
    xp = pa[0]*x + pa[1]*y
    yp = sa[0]*x + sa[1]*y
    
    #create a mask of those pixels for which xp == 0
    # this could be widened to allow some averaging along the length
    #mask = np.aps(xp) < 1
    
    #dr = data[:,:,0][mask]
    mask = (np.abs(xp) < 1).ravel()
    
    dr = data[:,:,0].ravel()*mask
    dr /= dr.sum()
    ypr = yp.ravel()#[mask]
    
    #calculate centroid
    cent = (ypr*dr).sum()
    ypr = np.abs(ypr - cent)
    
    #calculate mean absolute distance along yp
    mad = (ypr*dr).sum()
    #the std. deviation formula would have squareds in here
    
    return mad

    
def angle_filter(data, FILT_SIZE=5):
    return ndimage.generic_filter(data.astype('f'), th2, size=FILT_SIZE, extra_arguments=genCoords(FILT_SIZE))
    
def width_filter(data, angles=None, FILT_SIZE=5):
    if angles is None:
        #estimate angle from intensity data
        return ndimage.generic_filter(data.astype('f'), width, FILT_SIZE, extra_arguments=genCoords(FILT_SIZE))
    else:
        d = np.concatenate([data[:,:,None], angles[:,:,None]], 2)
        return ndimage.generic_filter(d.astype('f'), width_o, [FILT_SIZE, FILT_SIZE, 2], extra_arguments=genCoords(FILT_SIZE))[:,:,0].squeeze()

def roi_at(data, x0, y0, FILT_SIZE, x, y, b, ang):
    xi = x + x0
    yi = y + y0
    
    return data[xi, yi]
    
def width_filter_m(data, mask, angles=None, FILT_SIZE=5):
    d = data.astype('f')
    res = np.zeros_like(d) - 1
    
    xm, ym = np.where(mask)
    coords = genCoords(FILT_SIZE)
    am = np.ones_like(coords[1])
    for xi, yi in zip(xm, ym):
        dr = roi_at(d, xi, yi, *coords)
                
        if angles is None:
            res[xi, yi] = width(dr, *coords)
        else:
            #print dr.shape, 
            #print dr.shape, am.shape
            res[xi, yi] = width_o(np.concatenate([dr[:,:,None], angles[xi, yi, None]*am[:,:,None]], 2), *coords)
            
    return res
        
def angle_filter_m(mask, FILT_SIZE=5):
    d = mask.astype('f')
    res = np.zeros_like(d) - 1
    
    xm, ym = np.where(mask)
    coords = genCoords(FILT_SIZE)
    
    for xi, yi in zip(xm, ym):
        dr = roi_at(d, xi, yi, *coords)

        res[xi, yi] = th2(dr.ravel(), *coords)
                
    return res

    
    
def fold(thet):
    return thet +(-thet + (np.pi - thet))*(thet > np.pi/2)
    
def angHist(theta):
    #theta = theta - np.pi*(theta > np.pi/2)
    n, e = np.histogram(theta*180/np.pi, 20, [0, 180])
    nn = n.astype('f')/n.sum()
    
    w = e[1] - e[0]
    
    #pl.figure()
    for i in range(len(n)):
        #print i, e[i], nn[i], n[i], w
        pl.bar(e[i], nn[i], w, color=matplotlib.cm.hsv((e[i] +  w/2)/180))
        
    pl.xlabel('Angle [degrees]')
    pl.ylabel('Normalised frequency')
    
def angHist2(theta):
    n, e = np.histogram(theta*180/np.pi, 10, [0, 90])
    nn = n.astype('f')/n.sum()
    
    w = e[1] - e[0]
    
    pl.bar(e[:-1], nn, w)
        
    pl.xlabel('Angle [degrees]')
    pl.ylabel('Normalised frequency')
    
    
def procSkelFile(filename, disp=True):
    from PYME.contrib.gohlke import tifffile
    import os
    
    #generate a stub for our output files
    fs = os.path.split(os.path.splitext(filename)[0])
    outdir = os.path.join(fs[0], 'angle')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fstub = os.path.join(fs[0], 'angle', fs[1])
    #print fstub
    
    #load tiff file and invert
    im = (255 - tifffile.TIFFfile(filename).asarray().astype('f'))
    imt = angle_filter(im)
    
    imc = (im[:,:,None]*matplotlib.cm.hsv(imt/np.pi)[:,:,:3] + (255 - im)[:,:,None]).astype('uint8')
    imc2 = (im[:,:,None]*matplotlib.cm.hsv(fold(imt)/(np.pi))[:,:,:3] + (255 - im)[:,:,None]).astype('uint8')
    
    theta = imt[im > 0]
    
    if disp:
        fig = pl.figure()
        pl.subplot(121)
        pl.imshow(imc, interpolation='nearest')
        
        pl.subplot(222)
        angHist(theta)
        
        pl.subplot(224)
        angHist2(fold(theta))
        
        fig.savefig(fstub + '_angle.pdf')
        
    tifffile.imsave(fstub + '_angle.tif', imc)
    tifffile.imsave(fstub + '_angle_fold.tif', imc2)
        
    n, e = np.histogram(theta*180/np.pi, 20, [0, 180])
    nn = n.astype('f')/n.sum()
    
    np.savetxt(fstub + '_angle_hist.txt', np.vstack([e[:-1], n, nn]).T, delimiter='\t')
    
    n, e = np.histogram(fold(theta)*180/np.pi, 10, [0, 90])
    nn = n.astype('f')/n.sum()
    
    np.savetxt(fstub + '_fold_angle_hist.txt', np.vstack([e[:-1], n, nn]).T, delimiter='\t')
    
    
if __name__ == '__main__':
    import glob
    import sys
    
    filenames = sys.argv[1:]
    print(filenames)
    
    for fn in filenames:
        print(fn)
        procSkelFile(fn)
    
    