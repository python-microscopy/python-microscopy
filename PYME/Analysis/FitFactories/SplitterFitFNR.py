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

#import PYME.Analysis.twoColour as twoColour

from PYME.Analysis.cModels.gauss_app import *

#from scipy import weave

from PYME.Analysis._fithelpers import *

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)


def f_gauss2d2c(p, Xg, Yg, Xr, Yr):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    Ag,Ar, x0, y0, s, bG, bR, b_x, b_y  = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    r = genGauss(Xr,Yr,Ar,x0,y0,s,bR,b_x,b_y)
    #r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    g = genGauss(Xg,Yg,Ag,x0,y0,s,bG,b_x,b_y)
    #g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....
    
    return numpy.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)

def f_gauss2d2cA(p, Xg, Yg, Xr, Yr, Arr):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    #Ag,Ar, x0, y0, s, bG, bR, b_x, b_y  = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    #genGaussInArray(Arr[:,:,0], Xr,Yr,Ar,x0,y0,s,bR,b_x,b_y)
    #r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    #genGaussInArray(Arr[:,:,1],Xg,Yg,Ag,x0,y0,s,bG,b_x,b_y)
    #genSplitGaussInArray(Arr,Xg, Yg,Xr,Yr,Ag, Ar,x0,y0,s,bG,b_x,b_y)
    genSplitGaussInArrayPVec(p,Xg, Yg,Xr,Yr,Arr)
    #g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....

    return Arr #numpy.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)

def f_gauss2d2ccb(p, Xg, Yg, Xr, Yr):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    Ag,Ar, x0, y0, s, bG, bR = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    r = genGauss(Xr,Yr,Ar,x0,y0,s,bR = p)
    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    g = genGauss(Xg,Yg,Ag,x0,y0,s,bG = p)
    g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....

    return numpy.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)


        
def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n


#fresultdtype=[('tIndex', '<i4'),('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]),('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4')]),
              ('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4')]),
              ('startParams', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4')]), 
              ('subtractedBackground', [('g','<f4'),('r','<f4')])
              ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]


def GaussianFitResultR(fitResults, metadata, startParams, slicesUsed=None, resultCode=-1, fitErr=None, background = None):
	if slicesUsed == None:
		slicesUsed = ((-1,-1,-1),(-1,-1,-1))
	else: 		
		slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)))

	if fitErr == None:
		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')
  
      if background  == None:
          background = numpy.zeros(2, 'f')

	#print slicesUsed

	tIndex = metadata.tIndex

	#print fitResults.dtype
	#print fitErr.dtype
	#print fitResults
	#print fitErr
	#print tIndex
	#print slicesUsed
	#print resultCode


	return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), startParams.astype('f'), background, resultCode, slicesUsed)], dtype=fresultdtype) 
 
def BlankResult(metadata):
    r = numpy.zeros(1, fresultdtype)
    r['tIndex'] = metadata.tIndex
    r['fitError'].view('5f4')[:] = -5e3
    return r
		
def splWrap(*args):
    #print ''
    #args = args[:]
    return splitGaussWeightedMisfit(*args).copy()
    
    
def genFitImage(fitResults, metadata):
    #if fitfcn == f_Interp3d:
    #    if 'PSFFile' in metadata.getEntryNames():
    #        setModel(metadata.getEntry('PSFFile'), metadata)
    #    else:
    #        genTheoreticalModel(metadata)

    xslice = slice(*fitResults['slicesUsed']['x'])
    yslice = slice(*fitResults['slicesUsed']['y'])
    
    vx = 1e3*metadata.voxelsize.x
    vy = 1e3*metadata.voxelsize.y
    
    #position in nm from camera origin
    x_ = (xslice.start + metadata.Camera.ROIPosX - 1)*vx
    y_ = (yslice.start + metadata.Camera.ROIPosY - 1)*vy
    
    #look up shifts
    DeltaX = metadata.chroma.dx.ev(x_, y_)
    DeltaY = metadata.chroma.dy.ev(x_, y_)
    
    dxp = int(DeltaX/vx)
    dyp = int(DeltaY/vy)
    
    Xg = vx*scipy.mgrid[xslice]
    Yg = vy*scipy.mgrid[yslice]

    #generate a corrected grid for the red channel
    #note that we're cheating a little here - for shifts which are slowly
    #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
    #similarly for y. For slowly varying shifts the following should be
    #equivalent to this. For rapidly varying shifts all bets are off ...

    Xr = Xg + DeltaX - vx*dxp
    Yr = Yg + DeltaY - vy*dyp

    #X = 1e3*metadata.getEntry('voxelsize.x')*scipy.mgrid[xslice]
    #Y = 1e3*metadata.getEntry('voxelsize.y')*scipy.mgrid[yslice]
    #Z = array([0]).astype('f')
    #P = scipy.arange(0,1.01,.01)

    #im = fitfcn(fitResults['fitResults'], X, Y, Z, P).reshape(len(X), len(Y))
    #buf = numpy.zeros()
    d = np.zeros([Xr.shape[0], Yr.shape[0], 2], order='F')
    s = np.ones_like(d)
    buf = np.zeros(d.size)
    print d.shape, Xr.shape, Yr.shape
    im = -splWrap(np.array(list(fitResults['fitResults'])), d, s, Xg, Yg, Xr, Yr, buf).reshape(d.shape, order='F')

    return np.hstack([im[:,:,0], im[:,:,1]]).squeeze()
    



class GaussianFitFactory:
    def __init__(self, data, metadata, fitfcn=genSplitGaussInArrayPVec, background=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in 
        metadata. '''
        self.data = data
        self.background = background
        self.metadata = metadata
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        
        if False:#'D' in dir(fitfcn): #function has jacobian
            self.solver = FitModelWeightedJac
        else:
            self.solver = FitModelWeighted
		
        
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
        Xg = vx*scipy.mgrid[xslice]
        Yg = vy*scipy.mgrid[yslice]

        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...

        #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata.chroma.dx,self.metadata.chroma.dy)
        
        
       

        Xr = Xg + DeltaX - vx*dxp
        Yr = Yg + DeltaY - vy*dyp
        
        
        #a buffer so we can avoid allocating memory each time we evaluate the model function
        #buf = numpy.zeros(dataROI.shape, order='F')

        #print DeltaX
        #print DeltaY

        #estimate some start parameters...
        Ag = dataROI[:,:,0].max() - dataROI[:,:,0].min() #amplitude
        Ar = dataROI[:,:,1].max() - dataROI[:,:,1].min() #amplitude

        #figure()
        #imshow(np.hstack([dataROI[:,:,0], dataROI[:,:,1]]), interpolation='nearest')

        #print Ag
        #print Ar

        x0 =  Xg.mean()
        y0 =  Yg.mean()

        #startParameters = [Ag, Ar, x0, y0, 250/2.35, dataROI[:,:,0].min(),dataROI[:,:,1].min(), .001, .001]

	
        #estimate errors in data
        nSlices = 1#dataROI.shape[2]
        
        #sigma = scipy.sqrt(self.metadata.CCD.ReadNoise**2 + (self.metadata.CCD.noiseFactor**2)*self.metadata.CCD.electronsPerCount*self.metadata.CCD.EMGain*dataROI)/self.metadata.CCD.electronsPerCount
        sigma = scipy.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*scipy.maximum(dataROI, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount


        if not self.background == None and len(numpy.shape(self.background)) > 1 and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
            bgROI = self.background[xslice, yslice, 0:2] - self.metadata.Camera.ADOffset
            bgROI[:,:,1] = self.background[xslice2, yslice2, 1] - self.metadata.Camera.ADOffset

            dataROI = np.maximum(dataROI - bgROI, -sigma)

        #startParameters = [Ag, Ar, x0, y0, 250/2.35, dataROI[:,:,0].min(),dataROI[:,:,1].min(), .001, .001]
        startParameters = numpy.array([Ag, Ar, x0, y0, 250/2.35])

	
        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, Xg, Yg, Xr, Yr, buf)
        buf = numpy.zeros(dataROI.size)
        #(res, cov_x, infodict, mesg, resCode) = FitWeightedMisfitFcn(splitGaussWeightedMisfit, startParameters, dataROI, sigma, Xg, Yg, Xr, Yr)
        (res, cov_x, infodict, mesg, resCode) = FitWeightedMisfitFcn(splWrap, startParameters, dataROI, sigma, Xg, Yg, Xr, Yr, buf)

        #print infodict['nfev']


        

        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataROI.ravel())- len(res)))
        except Exception, e:
            pass

        #res = hstack([res, array([0,0,0,0])])
        #fitErrors = hstack([fitErrors, array([0,0,0,0])])

	#print res, fitErrors, resCode
        return GaussianFitResultR(res, self.metadata, startParameters,(xslice, yslice), resCode, fitErrors, bgROI.mean(0).mean(0))

    
        

#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray
