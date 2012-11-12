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

'''To be run by invoking:
    
    python angleFilter.py <file1> <file2> <file3> etc ....
    
    note that running python angleFilter.py <adirectory>/*.tif will expand out to
    the above.
    
    expects files to be in tiff format and inverted (ie skeleton is 0, other pixels 255).
    
    The calculated histograms are output as tab formatted txt, with the columns being
    left hand bin edge (in degrees), raw count, and normalised count respectively.
    
'''
from scipy import linalg, ndimage
import numpy as np
import pylab as pl

FILT_SIZE = 5
x, y = np.mgrid[-2:3, -2:3]
b = np.vstack([x.ravel(), y.ravel()]).T
ang = np.mod(np.angle(x +  1j*y), np.pi).ravel()


def th(data):
    return (data*ang).sum()/data.sum()

def th2(data):
    '''calculate principle axis of ROI using SVD'''
    if (data > 0).sum() < 2:
        #not enough data to calculate PA
        return -1
        
    pa =  linalg.svd(data[:,None]*b, full_matrices=False)[2][0]
    return np.angle(pa[0] + 1j*pa[1])%np.pi
    
    
def angle_filter(data):
    return ndimage.generic_filter(data.astype('f'), th2, FILT_SIZE)
    
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
        pl.bar(e[i], nn[i], w, color=pl.cm.hsv((e[i] +  w/2)/180))
        
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
    from PYME.gohlke import tifffile
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
    
    imc = (im[:,:,None]*pl.cm.hsv(imt/np.pi)[:,:,:3] + (255 - im)[:,:,None]).astype('uint8')
    
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
    print filenames
    
    for fn in filenames:
        print fn
        procSkelFile(fn)
    
    