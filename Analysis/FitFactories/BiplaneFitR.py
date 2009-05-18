import scipy
#from scipy.signal import interpolate
#import scipy.ndimage as ndimage
from pylab import *
import copy_reg
import numpy
import types

import PYME.Analysis.twoColour as twoColour

#from PYME.Analysis.cModels.gauss_app import *
from PYME.PSFGen.ps_app import *


#from scipy import weave

from PYME.Analysis._fithelpers import *

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)

def f_PSF3dBP(p, X1, Y1, X2,Y2, Z, P, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    A, x0, y0, z0, b = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b
    g1 = genWidefieldPSF(X1, Y1, Z[0], P,A*1e3, x0, y0, z0, *args) + b
    g2 = genWidefieldPSF(X2, Y2, Z[1], P,A*1e3, x0, y0, z0, *args) + b

    return numpy.concatenate((g1.reshape(g1.shape + (1,)),g2.reshape(g1.shape + (1,))), 2)

class PSFFitResult:
    def __init__(self, fitResults, metadata, slicesUsed=None, resultCode=None, fitErr=None):
        self.fitResults = fitResults
        self.metadata = metadata
        self.slicesUsed = slicesUsed
        self.resultCode=resultCode
        self.fitErr = fitErr

    def A(self):
        return self.fitResults[0]

    def x0(self):
        return self.fitResults[1]

    def y0(self):
        return self.fitResults[2]

    def z0(self):
        return self.fitResults[3]

    def background(self):
        return self.fitResults[4]

    def renderFit(self):
        #X,Y = scipy.mgrid[self.slicesUsed[0], self.slicesUsed[1]]
        #return f_gauss2d(self.fitResults, X, Y)
        X = 1e3*self.metadata.voxelsize.x*scipy.mgrid[self.slicesUsed[0]]
        Y = 1e3*self.metadata.voxelsize.y*scipy.mgrid[self.slicesUsed[1]]
        Z = 1e3*self.metadata.voxelsize.z*scipy.mgrid[self.slicesUsed[2]]
        P = scipy.arange(0,1.01,.1)
        return f_PSF3d(self.fitResults, X, Y, Z, P, 2*scipy.pi/525, 1.47, 10e3)
        #pass

def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n


fresultdtype=[('tIndex', '<i4'),('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]),('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def PSFFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
	if slicesUsed == None:
		slicesUsed = ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
	else:
		slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)),(slicesUsed[2].start,slicesUsed[2].stop,replNoneWith1(slicesUsed[2].step)))

	if fitErr == None:
		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

	#print slicesUsed

	tIndex = metadata.tIndex


	return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, slicesUsed)], dtype=fresultdtype)

		

class BiplaneFitFactory:
    def __init__(self, data, metadata, fitfcn=f_PSF3dBP):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in 
        metadata. '''
        self.data = data
        self.metadata = metadata
	self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
	if type(fitfcn) == types.FunctionType: #single function provided - use numerically estimated jacobian
		self.solver = FitModelWeighted
	else: #should be a tuple containing the fit function and its jacobian
		self.solver = FitModelWeightedJac
		
        
    def __getitem__(self, key):
        #print key
        xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice] - self.metadata.Camera.ADOffset

        #average in z
        #dataMean = dataROI.mean(2) - self.metadata.CCD.ADOffset

        #generate grid to evaluate function on        
        Xg = 1e3*self.metadata.voxelsize.x*scipy.mgrid[xslice]
        Yg = 1e3*self.metadata.voxelsize.y*scipy.mgrid[yslice]

        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...

        #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata.chroma.dx,self.metadata.chroma.dy)

        #Xr = Xg + DeltaX
        #Yr = Yg + DeltaY

        Xr = Xg
        Yr = Yg

        Z = array([0, self.metadata.Splitter.zOffset])
        P = scipy.arange(0,1.01,.01)

        #print DeltaX
        #print DeltaY

        #estimate some start parameters...
        A = dataROI.max() - dataROI.min() #amplitude

        #figure()
        #imshow(dataROI[:,:,1], interpolation='nearest')

        #print Ag
        #print Ar

        x0 =  Xg.mean()
        y0 =  Yg.mean()
        z0 = 0.

        startParameters = [A, x0, y0, z0, dataROI.min()]


        #estimate errors in data
        nSlices = 1#dataROI.shape[2]

        sigma = scipy.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*dataROI)/self.metadata.Camera.ElectronsPerCount


        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, Xg, Yg, Xr, Yr, Z, P)



        fitErrors=None
        try:
            fitErrors = scipy.sqrt(scipy.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataROI.ravel())- len(res)))
        except Exception, e:
            pass

        #print res, fitErrors, resCode
        return PSFFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors)

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        #if (z == None): # use position of maximum intensity
        #    z = self.data[x,y,:].argmax()

        x = round(x)
        y = round(y)

        return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]),
            max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]), 0:2]
        

#so that fit tasks know which class to use
FitFactory = BiplaneFitFactory
FitResult = PSFFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray
