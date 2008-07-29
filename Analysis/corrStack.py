import numpy
import scipy
from scipy.fftpack import fftn, ifftn, ifftshift, fftshift
from math import *
from pylab import *
import scipy.ndimage

#def genMask(shape, frac):
#    m = numpy.zeros(shape)
    

def calcShift(im1, im2, fc = 0.7):
    xm = im1.shape[0]/2.0
    ym = im1.shape[1]/2.0

    #print (xm, ym)

    m = numpy.ones(im1.shape)
    
    m[(floor(xm) - round(fc*xm)):(ceil(xm) + round(fc*xm) + 1),(floor(ym) - round(fc*ym)):(ceil(ym) + round(fc*ym) + 1)] = 0

    m = scipy.ndimage.gaussian_filter(m, 5)

    s = 5
    m2 = numpy.ones(im1.shape, 'f')
    m2[0:2*s,:] = 0
    m2[-(2*s + 1):, :] = 0
    m2[:,0:2*s] = 0
    m2[:,-(2*s + 1):] = 0

    m2 = scipy.ndimage.gaussian_filter(m2, s)

    CR = ifftshift(ifftn(fftn((im1 - im1.min())*m2)*ifftshift(m)*ifftn((im2 - im2.min())*m2)))
    #print im1.shape
    #plot(scipy.absolute(CR[round(xm), :]))
    #figure(1)
    #imshow(fftshift(m))

    #figure(2)
    #imshow(scipy.log10(scipy.absolute(fftn(im1))))
    
    xr = range(floor(xm) - 3, ceil(xm) + 5)
    yr = range(floor(ym) - 3, ceil(ym) + 5)

    #print xr
    #print yr

    X,Y = numpy.mgrid[xr[0]:(xr[-1] + 1),yr[0]:(yr[-1] + 1)]

    X = X.astype('f') - xm
    Y = Y.astype('f') - ym

    #print X
    #print Y

    #print scipy.absolute(CR).shape

    CR_C = scipy.absolute(CR)[xr[0]:(xr[-1] + 1),yr[0]:(yr[-1] + 1)]
    #CR_C = CR_C - CR_C.min()

    #figure(1)

    #imshow(scipy.absolute(CR))

    #figure(2)
    #imshow(CR_C, interpolation='nearest')

    #figure(3)
    #imshow(scipy.absolute((ifftn(fftn((im1 - im1.min())*m2)*fftshift(m)))))

    #print CR_C.shape

    dx = (CR_C*X).sum()/CR_C.sum()
    dy = (CR_C*Y).sum()/CR_C.sum()

    return(dx,dy)
