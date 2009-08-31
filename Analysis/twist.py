import numpy
import scipy.ndimage
from pylab import *

tcAng = None
tcZ = None

def calcTwist(im, X, Y):
    an = numpy.mod(numpy.angle(X[:,None] + (1j*Y)[None, :]), numpy.pi)

    #imshow(an)
    
    I = numpy.argsort(an.ravel())

    ar = an.ravel()[I]
    imr = im.ravel()[I]

    imrs = scipy.ndimage.gaussian_filter(imr*(imr > (imr.max()/2.)), 100, 0, mode='wrap')

    #plot(ar, imrs)

    return(ar[imrs.argmax()])

def twistCal(ps, X, Y, Z):
    global tcAng, tcZ

    a = [calcTwist(ps[:,:,i],X, Y) for i in range(len(Z))]

    af = scipy.ndimage.gaussian_filter(a, 1)
    af.sort()

    tcAng = af
    tcZ = Z

def getZ(twist):
    return numpy.interp(twist, tcAng, tcZ)