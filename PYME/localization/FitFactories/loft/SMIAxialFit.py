#!/usr/bin/python

##################
# SMIAxialFit.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import numpy as np
from scipy.signal import interpolate
import scipy.ndimage as ndimage
# from pylab import *
import matplotlib.pyplot as plt

from _fithelpers import *

def f_SMIAxial(p, z, ysmooth, background=0):
    """model fcn for SMI axial profile"""
    k, z0, A, r = p
    return A*ysmooth*(1 - r + r*np.cos(k*(z - z0))**2) + background

class SMIAxialFitResult:
    def __init__(self, fitResults, metadata, background, ysmooth, ind3, pos=None, resultCode=None, profile=None, startParams=None, peakPoss=None):
        self.fitResults = fitResults
        self.metadata = metadata
        self.pos = pos
        self.resultCode=resultCode
        self.background = background
        self.ysmooth = ysmooth
        self.profile = profile
        self.peakPoss = peakPoss
        self.ind3 = ind3
        self.startParams = startParams
    
    def k(self):
        return self.fitResults[0]

    def z0(self):
        return self.fitResults[1]

    def A(self):
        return self.fitResults[2]

    def r(self):
        return self.fitResults[3]
   
    def size(self, umod=0, calibrationCurve="OldSphere"):
        return getattr(self, 'calibrationCurve%s' % calibrationCurve)(umod)(self.r())

    def calibrationCurveGauss(self,umod):
        """generates a callable object which can be evaluated to map a measured modulation depth to a
        the FWHM of a Gaussian using the k-value derived from the fit and a given
        unmodulating fraction"""

        k = 2*self.k()/(self.metadata.voxelsize.z*1e3)
        class fwhm_calc:
            def __init__(self,k,umod):
                self.k = k
                self.umod = umod
            def __call__(self, r):
                return 2*np.sqrt(4*np.log(0.5)*np.log(r + self.umod)/(self.k**2))
        return fwhm_calc(k, umod)

    def calibrationCurveSphere(self):
        pass

    def calibrationCurveOldSphere(self, unmod):
        """generates a function which can be evaluated to map a measured modulation depth to a
        diameter based on the (slightly erroneous) spherical model used up to this point in the 
        Matlab sviewer implementation and using the k-value derived from the fit and a given
        unmodulating fraction"""

        k = self.k()/(self.metadata.voxelsize.z*1e3)
        #for a range of diameters up to the first zero of the calibration curve (the position of the zero crossing at k*d = 4.5 was determined graphicaly)
        d =  np.arange( (4.51 + .1)/k, 0, -.1/k) # note reversed order s.t. mod will be monotonically increasing (as required by interp1d of its x values)
        mod = (3*(1-unmod)*(np.sin(k*d) - k*d*np.cos(k*d))/((k**3)*(d**3)))
        return interpolate.interp1d(mod,d, bounds_error=False)
        
    def evalFit(self):
        return f_SMIAxial(self.fitResults, self.ind3, self.ysmooth[self.ind3], self.background)

    def renderFit(self):
        plt.clf()
        plt.plot(self.profile, 'x-')
        plt.plot(self.ysmooth)

        plt.plot((self.peakPoss['i'],self.peakPoss['i']), (0, self.profile.max()))
        plt.plot((self.peakPoss['i2'],self.peakPoss['i2']), (0, self.profile.max()))
        plt.plot((self.peakPoss['i3'],self.peakPoss['i3']), (0, self.profile.max()))
        
        plt.plot(self.ind3, self.evalFit(), lw=2)


class SMIAxialFitFactory:
    def __init__(self, data, metadata, roiHalfSize=2, backgroundHalfSize=5):
        self.data = data
        self.metadata = metadata
        self.roi= roiHalfSize
        self.backRoi = backgroundHalfSize

    def __FindFitRegion(self, ysmooth, ysmooth2):
        ind3 = np.where((ysmooth.__gt__(ysmooth2)) * (ysmooth > ysmooth.max()/3))[0]
        
        #print ind3
        #Take only the longest peak
        idiff = ind3[2:] - ind3[1:-1]
        idiff[-1] = 1
        
        i4 = (np.array(idiff) - 1).cumsum()
        max_i = 0
        nm = 0

        for i in range(i4.max() + 1):# which is longest
            ni = len(np.where(i4 == i)[0])
            if (ni > nm):
                max_i = i
                nm = ni

        ind3 = ind3[i4 == max_i]
        
        return ind3

    def __CalcStartParams(self, profile):
        xi = np.arange(0, len(profile) - 1, .1)
        ti = interpolate.interp1d(range(len(profile)), profile, 'cubic')(xi)

        i_ = ti.argmax()
        v = ti[i_]
        i = xi[i_]
    
        if( i > 20): #we are not at the beginning of the AID   
            #and the next max, so we can guestimate some initial parameters for our fit
            lb = round(10*0.300/self.metadata.voxelsize.z)
            i3_ = ti[(i_-lb):(i_-round(10*.100/self.metadata.voxelsize.z))].argmax()
            i3_ = i3_ + i_ -lb
            
    
            #the next minimum
            i2_ = ti[i3_:i_].argmin()
            i2_ = i2_ + i3_
            
        else:
            #and the next max, so we can guestimate some initial parameters for our fit
            lb = round(10*.100/self.metadata.voxelsize.z)
            i3_ = ti[(i_ + lb):(i_+round(10*0.300/self.metadata.voxelsize.z))].argmax()
            i3_ = i3_ + i_ +lb
            
    
            #the next minimum
            i2_ = ti[i_:i3_].argmin()
            i2_ = i2_ + i_
        
        v3 = ti[i3_]
        i3 = xi[i3_]
    
        dx = abs(i3-i)
        
        v2 = ti[i2_]
        i2 = xi[i2_]

        return ([np.pi/dx,np.mod(i,dx),2/(1+v2/v),1-v2/v], {'i':i, 'v':v, 'i2':i2,'v2':v2, 'i3':i3,'v3':v3})
        

    def FromPoint(self, xpos, ypos, gaussFitBackground=None):
        #extract axial profile
        #print (max((xpos - self.roi), 0), min((xpos + self.roi + 1),self.data.shape[0]))
        profile = self.data[max((xpos - self.roi), 0):min((xpos + self.roi + 1),self.data.shape[0]), 
                    max((ypos - self.roi), 0):min((ypos + self.roi + 1), self.data.shape[1]), :].astype('f').mean(0).mean(0)

        #plot(profile)
        #print profile

        if not (self.backRoi is None): #perform background subtraction
            backg = self.data[max((xpos - self.backRoi), 0):min((xpos + self.backRoi + 1),self.data.shape[0]), 
                    max((ypos - self.backRoi), 0):min((ypos + self.backRoi + 1), self.data.shape[1]), :].astype('f').mean(0).mean(0)
            #print backg.shape
            profile -= backg
            back = 0
        else:
            if gaussFitBackground is None:
                raise Exception('No background defined', 'either a background ROI or a background value derived from e.g. the Gaussian fit must be given')
            else:
                back = gaussFitBackground

        #extract ysmooth
        ysmooth = ndimage.gaussian_filter1d(profile, 2.5*self.metadata.voxelsize.z/.04) - back
        ysmooth2 = ndimage.gaussian_filter1d(profile, 10*self.metadata.voxelsize.z/.04) - back

        #define region to be fitted
        ind3 = self.__FindFitRegion(ysmooth, ysmooth2)
        
        #estimate some start parameters...
        (startParameters, peakPoss) = self.__CalcStartParams(profile)

        #do the fit
        #print startParameters
        #print ind3
        try:
            (res, resCode) = FitModel(f_SMIAxial, startParameters, profile[ind3], ind3, ysmooth[ind3], back)
        except TypeError as e:
            res = [-1, -1, -1, -1]
            resCode = -1

        return SMIAxialFitResult(res, self.metadata, back, ysmooth, ind3,(xpos, ypos), resCode, profile, startParameters, peakPoss)
