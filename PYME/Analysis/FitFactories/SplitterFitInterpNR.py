#!/usr/bin/python

##################
# LatGaussFitFRTC.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import scipy
#from scipy.signal import interpolate
#import scipy.ndimage as ndimage
from pylab import *
import copy_reg
import numpy
import types

from PYME.Analysis._fithelpers import *

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)


def f_Interp3d2c(p, interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, axialShift, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    Ag, Ar, x0, y0, z0, bG, bR = p

    #make sure our model is big enough to stretch to our current position
#    xm = len(Xg)/2
#    dx = min((interpolator.shape[0] - len(Xg))/2, xm) - 2
#
#    ym = len(Yg)/2
#    dy = min((interpolator.shape[1] - len(Yg))/2, ym) - 2
#
#    x0 = min(max(x0, Xg[xm - dx]), Xg[dx + xm])
#    y0 = min(max(y0, Yg[ym - dy]), Yg[dy + ym])
#    z0 = min(max(z0, max(Zg[0], Zr[0]) + interpolator.IntZVals[2]), min(Zg[0], Zr[0]) + interpolator.IntZVals[-2])

    #currently just come to a hard stop when the optimiser tries to leave the safe region
    #prob. not ideal, for a number of reasons
    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0] + axialShift), safeRegion[2][1] - axialShift)

    g = interpolator.interp(Xg - x0 + 1, Yg - y0 + 1, Zg[0] - z0 + 1)*Ag + bG
    r = interpolator.interp(Xr - x0 + 1, Yr - y0 + 1, Zr[0] - z0 + 1)*Ar + bR

    return numpy.concatenate((g,r), 2)
    
def f_J_Interp3d2c(p,interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, axialShift, *args):
    '''generate the jacobian - for use with _fithelpers.weightedJacF'''
    Ag, Ar, x0, y0, z0, bG, bR = p

    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0] + axialShift), safeRegion[2][1] - axialShift)

    g = interpolator.interp(Xg - x0 + 1, Yg - y0 + 1, Zg[0] - z0 + 1)
    r = interpolator.interp(Xr - x0 + 1, Yr - y0 + 1, Zr[0] - z0 + 1)
    
    gx, gy, gz = interpolator.interpG(Xg - x0 + 1, Yg - y0 + 1, Zg[0] - z0 + 1)
    rx, ry, rz = interpolator.interpG(Xr - x0 + 1, Yr - y0 + 1, Zr[0] - z0 + 1)
    
    bg = np.ones_like(gx)
    zb = np.zeros_like(gx)
    
    dAg = numpy.concatenate((g,zb), 2).ravel()[:,None]
    dAr = numpy.concatenate((zb,r), 2).ravel()[:,None]
    dX = numpy.concatenate((Ag*gx,Ar*rx), 2).ravel()[:,None]
    dY = numpy.concatenate((Ag*gy,Ar*ry), 2).ravel()[:,None]
    dZ = numpy.concatenate((Ag*gz,Ar*rz), 2).ravel()[:,None]
    dBg = numpy.concatenate((bg,zb), 2).ravel()[:,None]
    dBr = numpy.concatenate((zb,bg), 2).ravel()[:,None]
    
    #r = r.reshape((-1, 7))
    return numpy.hstack([dAg, dAr, dX, dY, dZ, dBg, dBr])

        
def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n


#fresultdtype=[('tIndex', '<i4'),('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]),('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4')]),
              ('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4')]),
              ('startParams', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4')]), 
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('x2', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y2', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
              ('nchi2', '<f4')]


def PSFFitResultR(fitResults, metadata, startParams, slicesUsed=None, resultCode=-1, fitErr=None, nchi2=-1):
	if slicesUsed == None:
		slicesUsed = ((-1,-1,-1),(-1,-1,-1))
	else: 		
		slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)))

	if fitErr == None:
		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

	#print slicesUsed

	tIndex = metadata.tIndex

	#print fitResults.dtype
	#print fitErr.dtype
	#print fitResults
	#print fitErr
	#print tIndex
	#print slicesUsed
	#print resultCode


	return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), startParams.astype('f'), resultCode, slicesUsed, nchi2)], dtype=fresultdtype) 
 
def BlankResult(metadata):
    r = numpy.zeros(1, fresultdtype)
    r['tIndex'] = metadata.tIndex
    r['fitError'].view('5f4')[:] = -5e3
    return r
		

def getDataErrors(im, metadata):
    dataROI = im - metadata.getEntry('Camera.ADOffset')

    return scipy.sqrt(metadata.getEntry('Camera.ReadNoise')**2 + (metadata.getEntry('Camera.NoiseFactor')**2)*metadata.getEntry('Camera.ElectronsPerCount')*metadata.getEntry('Camera.TrueEMGain')*dataROI)/metadata.getEntry('Camera.ElectronsPerCount')    



class InterpFitFactory:
    def __init__(self, data, metadata, fitfcn=f_Interp3d2c, background=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in 
        metadata. '''
        self.data = data
        self.background = background
        self.metadata = metadata
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        
        if not 'D' in dir(fitfcn): #single function provided - use numerically estimated jacobian
            self.solver = FitModelWeighted_
        else: #should be a tuple containing the fit function and its jacobian
            self.solver = FitModelWeightedJac_


        interpModule = metadata.Analysis.InterpModule
        self.interpolator = __import__('PYME.Analysis.FitFactories.Interpolators.' + interpModule , fromlist=['PYME', 'Analysis','FitFactories', 'Interpolators']).interpolator

        if 'Analysis.EstimatorModule' in metadata.getEntryNames():
            estimatorModule = metadata.Analysis.EstimatorModule
        else:
            estimatorModule = 'astigEstimator'

        self.startPosEstimator = __import__('PYME.Analysis.FitFactories.zEstimators.' + estimatorModule , fromlist=['PYME', 'Analysis','FitFactories', 'zEstimators'])

        if fitfcn == f_Interp3d2c:
            if 'PSFFile' in metadata.getEntryNames():
                if self.interpolator.setModelFromMetadata(metadata):
                    print 'model changed'
                    self.startPosEstimator.splines.clear()

                if not 'z' in self.startPosEstimator.splines.keys():
                    self.startPosEstimator.calibrate(self.interpolator, metadata)
            else:
                self.interpolator.genTheoreticalModel(metadata)
                
    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on
        #setModel(md.PSFFile, md)
        interpolator = __import__('PYME.Analysis.FitFactories.Interpolators.' + md.Analysis.InterpModule , fromlist=['PYME', 'Analysis','FitFactories', 'Interpolators']).interpolator
        
        if 'Analysis.EstimatorModule' in md.getEntryNames():
            estimatorModule = md.Analysis.EstimatorModule
        else:
            estimatorModule = 'astigEstimator'

        #this is just here to make sure we clear our calibration when we change models        
        startPosEstimator = __import__('PYME.Analysis.FitFactories.zEstimators.' + estimatorModule , fromlist=['PYME', 'Analysis','FitFactories', 'zEstimators'])        
        
        if interpolator.setModelFromFile(md.PSFFile, md):
            print 'model changed'
            startPosEstimator.splines.clear()

        Xg, Yg, Zg, safeRegion = interpolator.getCoords(md, slice(-roiHalfSize,roiHalfSize + 1), slice(-roiHalfSize,roiHalfSize + 1), slice(0,1))
        
        DeltaX = md.chroma.dx.ev(x, y)
        DeltaY = md.chroma.dy.ev(x, y)

        Xr = Xg + DeltaX
        Yr = Yg + DeltaY
        Zr = Zg + md.Analysis.AxialShift

        return f_Interp3d2c(params, interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, md.Analysis.AxialShift), Xg.ravel()[0], Yg.ravel()[0], Zg.ravel()[0]
		
        
    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        #if (z == None): # use position of maximum intensity
        #    z = self.data[x,y,:].argmax()
	
        x = round(x)
        y = round(y)
        
        #pixel size in nm
        vx = 1e3*self.metadata.voxelsize.x
        vy = 1e3*self.metadata.voxelsize.y
        
        #position in nm from camera origin
        x_ = (x + self.metadata.Camera.ROIPosX - 1)*vx
        y_ = (y + self.metadata.Camera.ROIPosY - 1)*vy
        
        #look up shifts
        DeltaX = self.metadata.chroma.dx.ev(x_, y_)
        DeltaY = self.metadata.chroma.dy.ev(x_, y_)
        
        #find shift in whole pixels
        dxp = int(DeltaX/vx)
        dyp = int(DeltaY/vy)
        
        #find ROI which works in both channels
        #if dxp < 0:
        x01 = max(x - roiHalfSize, max(0, dxp))
        x11 = min(max(x01, x + roiHalfSize), self.data.shape[0] + min(0, dxp))
        x02 = x01 - dxp
        x12 = x11 - dxp
        
        y01 = max(y - roiHalfSize, max(0, dyp))
        y11 = min(max(y + roiHalfSize,  y01), self.data.shape[1] + min(0, dyp))
        y02 = y01 - dyp
        y12 = y11 - dyp
        
        xslice = slice(x01, x11)
        xslice2 = slice(x02, x12) 
        
        yslice = slice(y01, y11)
        yslice2 = slice(y02, y12)
        
        zslice = slice(0,2)
        
        #print x, y, x01, x11, y01, y11, '\t', dxp, dyp
        
        #print key
        #xslice, yslice, zslice = key

         #cut region out of data stack
        dataROI = self.data[xslice, yslice, 0:2] - self.metadata.Camera.ADOffset
        #print dataROI.shape, xslice, yslice, xslice2, yslice2
        dataROI[:,:,1] = self.data[xslice2, yslice2, 1] - self.metadata.Camera.ADOffset
        
        if min(dataROI.shape[:2]) < 4: # too small to fit
            return BlankResult(self.metadata)
        
        #dataROI -= self.metadata.Camera.ADOffset

        #average in z
        #dataMean = dataROI.mean(2) - self.metadata.CCD.ADOffset

        #generate grid to evaluate function on        
        Xg, Yg, Zg, safeRegion = self.interpolator.getCoords(self.metadata, xslice, yslice, zslice)


        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...

        Xr = Xg + DeltaX - vx*dxp
        Yr = Yg + DeltaY - vy*dyp
        Zr = Zg + self.metadata.Analysis.AxialShift
        
        if len(Xg.shape) > 1: #X is a matrix
            X_ = Xg[:, 0, 0]
            Y_ = Yg[0, :, 0]
        else:
            X_ = Xg
            Y_ = Yg
        
        
        #a buffer so we can avoid allocating memory each time we evaluate the model function
        #buf = numpy.zeros(dataROI.shape, order='F')

        #estimate errors in data
        nSlices = 1#dataROI.shape[2]
        
        #sigma = scipy.sqrt(self.metadata.CCD.ReadNoise**2 + (self.metadata.CCD.noiseFactor**2)*self.metadata.CCD.electronsPerCount*self.metadata.CCD.EMGain*dataROI)/self.metadata.CCD.electronsPerCount
        sigma = scipy.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*scipy.maximum(dataROI, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount + 1


#        if not self.background == None and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
#            #print 'bgs'
#            if len(numpy.shape(self.background)) > 1:
#                bgROI = self.background[xslice, yslice, zslice] - self.metadata.Camera.ADOffset
#
#                dataROI = dataROI - bgROI
#            else:
#                dataROI = dataROI - (self.background - self.metadata.Camera.ADOffset)
                
        if not self.background == None and len(numpy.shape(self.background)) > 1 and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
            bgROI = self.background[xslice, yslice, 0:2] - self.metadata.Camera.ADOffset
            bgROI[:,:,1] = self.background[xslice2, yslice2, 1] - self.metadata.Camera.ADOffset

            dataROI = np.maximum(dataROI - bgROI, -sigma)

        #estimate some start parameters...
        Ag = dataROI[:,:,0].max() - dataROI[:,:,0].min() #amplitude
        Ar = dataROI[:,:,1].max() - dataROI[:,:,1].min() #amplitude        
        
        z0 = 0
        if Ag > Ar: #use prightest channel for start parameter estimation
            startParams = self.startPosEstimator.getStartParameters(dataROI[:,:,:1], X_, Y_)
        else:
            startParams = self.startPosEstimator.getStartParameters(dataROI[:,:,1:], X_, Y_)
            z0 = self.metadata.Analysis.AxialShift

        startParameters = [Ag*startParams[0]/(Ag + Ar), Ar*startParams[0]/(Ag + Ar), startParams[1], startParams[2], z0 + startParams[3], dataROI[:,:,0].min(),dataROI[:,:,1].min()]

        #print dataROI.shape
	
        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, self.interpolator,Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, np.abs(self.metadata.Analysis.AxialShift))

        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x) * (infodict['fvec'] * infodict['fvec']).sum() / (len(dataROI.ravel())- len(res)))
        except Exception, e:
            pass

        #normalised Chi-squared
        nchi2 = (infodict['fvec']**2).sum()/(dataROI.size - res.size)

	#print res, fitErrors, resCode
        return PSFFitResultR(res, self.metadata, np.array(startParams), (xslice, yslice, xslice2, yslice2), resCode, fitErrors, nchi2)
        #return PSFFitResultR(res, self.metadata, , resCode, fitErrors, numpy.array(startParameters), nchi2)
    
        

#so that fit tasks know which class to use
FitFactory = InterpFitFactory
FitResult = PSFFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray
