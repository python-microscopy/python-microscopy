#!/usr/bin/python

##################
# ofind_xcorr.py
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
import scipy.signal
import scipy.ndimage as ndimage
import numpy

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
from scipy.fftpack import fftn, ifftn, ifftshift

#import fftw3

from PYME.contrib import pad
from PYME.IO.FileUtils.nameUtils import getFullExistingFilename
from PYME.IO.load_psf import load_psf
from scipy.spatial import kdtree

import fftw3f as fftw3
from PYME.Deconv import fftwWisdom

#from wiener import resizePSF

fftwWisdom.load_wisdom()
#import weave
#import cDec
#from PYME import pad
#import dec

NTHREADS = 1
FFTWFLAGS = ['measure']

#import pylab
PSFFileName = None
PSFSize = None

FTW = None
#cachedPSF = None
#cachedOTF2 = None
#cachedOTFH = None
#autocorr = None

class fftwWeiner:
    def __init__(self, ps, vox, PSSize):
        ps = ps.max(2)
        ps = ps - ps.min()

        ps = ps*scipy.signal.hanning(ps.shape[0])[:,None]*scipy.signal.hanning(ps.shape[1])[None,:]
        ps = ps/ps.sum()
        #PSFFileName = PSFFilename

        pw = (numpy.array(PSSize) - ps.shape)/2.
        pw1 = numpy.floor(pw)
        pw2 = numpy.ceil(pw)

        self.cachedPSF = pad.with_constant(ps, ((pw2[0], pw1[0]), (pw2[1], pw1[1])), (0,))
        self.cachedOTFH = (ifftn(self.cachedPSF)*self.cachedPSF.size).astype('complex64')
        self.cachedOTF2 = (self.cachedOTFH*fftn(self.cachedPSF)).astype('complex64')

        self.weinerFT = fftw3.create_aligned_array(self.cachedOTFH.shape, 'complex64')
        self.weinerR = fftw3.create_aligned_array(self.cachedOTFH.shape, 'float32')

        self.planForward = fftw3.Plan(self.weinerR, self.weinerFT, flags = FFTWFLAGS, nthreads=NTHREADS)
        self.planInverse = fftw3.Plan(self.weinerFT, self.weinerR, direction='reverse', flags = FFTWFLAGS, nthreads=NTHREADS)
        
        fftwWisdom.save_wisdom()
        
        self.otf2mean = self.cachedOTF2.mean()

    def filter(self, data, lamb):
        l2 = lamb**2
        self.weinerR[:] = data.astype('float')

        self.planForward()

        self.weinerFT[:] = self.weinerFT[:]*self.cachedOTFH*(l2 + self.otf2mean)/(l2 + self.cachedOTF2)

        self.planInverse()

        return ifftshift(self.weinerR)

    def correlate(self, data):
        self.weinerR[:] = data.astype('float')

        self.planForward()

        self.weinerFT[:] = self.weinerFT[:]*self.cachedOTFH

        self.planInverse()

        return ifftshift(self.weinerR)



#def preparePSF(PSFFilename, PSSize):
#    global PSFFileName, cachedPSF, cachedOTF2, cachedOTFH, autocorr
#    if (not (PSFFileName == PSFFilename)) or (not (cachedPSF.shape == PSSize)):
#        fid = open(getFullExistingFilename(PSFFilename), 'rb')
#        ps, vox = cPickle.load(fid)
#        fid.close()
#        ps = ps.max(2)
#        ps = ps - ps.min()
#        #ps = ps*(ps > 0)
#        ps = ps*scipy.signal.hanning(ps.shape[0])[:,None]*scipy.signal.hanning(ps.shape[1])[None,:]
#        ps = ps/ps.sum()
#        PSFFileName = PSFFilename
#        pw = (numpy.array(PSSize) - ps.shape)/2.
#        pw1 = numpy.floor(pw)
#        pw2 = numpy.ceil(pw)
#        cachedPSF = pad.with_constant(ps, ((pw2[0], pw1[0]), (pw2[1], pw1[1])), (0,))
#        #cachedOTFH = fftw3.create_aligned_array(cachedPSF.shape, 'complex64')
#        cachedOTFH = ifftn(cachedPSF)*cachedPSF.size
#        #cachedOTF2 = fftw3.create_aligned_array(cachedPSF.shape, 'complex64')
#        cachedOTF2 = cachedOTFH*fftn(cachedPSF)
#        #autocorr = ifftshift(ifftn(cachedOTF2)).real
        
def preparePSF(md, PSSize):
    global PSFFileName, PSFSize, FTW

    PSFFilename = md['PSFFile']    
    
    if (not (PSFFileName == PSFFilename)) or (not (PSFSize == PSSize)):
        try:
            ps, vox = md['taskQueue'].getQueueData(md['dataSourceID'], 'PSF')
        except:
            #fid = open(getFullExistingFilename(PSFFilename), 'rb')
            #ps, vox = pickle.load(fid)
            #fid.close()
            load_psf(PSFFilename)
        
        FTW = fftwWeiner(ps,vox, PSSize)

        PSFFileName = PSFFilename
        PSFSize = PSSize



class OfindPoint:
    def __init__(self, x, y, z=None, detectionThreshold=None):
        """Creates a point object, potentially with an undefined z-value."""
        self.x = x
        self.y = y
        self.z = z
        self.detectionThreshold = detectionThreshold

#a hack so we'll be able to do something like ObjectIdentifier.x[i]
class PseudoPointList:
    def __init__(self,parent, varName):
        self.parent = parent
        self.varName = varName

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, key):
        tmp = self.parent[key]
        if not '__len__' in dir(tmp):
            return tmp.__dict__[self.varName]
        else:
            tm2 = []
            for it in tmp:
                tm2.append(it.__dict__[self.varName])
            return tm2

    #def __iter__(self):
    #    self.curpos = -1
    #    return self

    #def next(self):
    #    curpos += 1
    #    if (curpos >= len(self)):
    #        raise StopIteration
    #    return self[self.curpos]

class ObjectIdentifier(list):
    def __init__(self, data, PSFFilename, filterRadiusHighpass=5, lamb = 5e-5):
        """Creates an Identifier object to be used for object finding, takes a 2D or 3D slice
        into a data stack (data), and a filtering mode (filterMode, one of ["fast", "good"])
        where "fast" performs a z-projection and then filters, wheras "good" filters in 3D before 
        projecting. The parameters filterRadiusLowpass and filterRadiusHighpass control the bandpass filter
        used to identify 'point-like' features. filterRadiusZ is the radius used for the axial smoothing filter"""
        self.data = data
        self.filterRadiusHighpass = filterRadiusHighpass
        self.lamb = lamb
        preparePSF(PSFFilename, data.shape[:2])
        
    def __FilterData2D(self,data):
        #lowpass filter to suppress noise
        #a = ndimage.gaussian_filter(data.astype('f'), self.filterRadiusLowpass)
        a = FTW.filter(data, self.lamb)
        #lowpass filter again to find background
        b = ndimage.gaussian_filter(a, self.filterRadiusHighpass)
        return 24*(a - b)

    def __FilterThresh2D(self,data):
        #lowpass filter to suppress noise
        #a = ndimage.gaussian_filter(data.astype('f'), self.filterRadiusLowpass)
        #a = ifftshift(ifftn((fftn(data.astype('f'))*cachedOTFH))).real
        #a = ifftshift(ifftn((fftn(data.astype('f'))*cachedOTFH)*(self.lamb**2 + cachedOTF2.mean())/(self.lamb**2 + cachedOTF2))).real
        #lowpass filter again to find background
        #b = ndimage.gaussian_filter(a, self.filterRadiusHighpass)
        return FTW.correlate(data)
    
    def __FilterDataFast(self):
        #project data
        if len(self.data.shape) == 2: #if already 2D, do nothing
            projData = self.data
        else:
            projData = self.data.max(2) 
        return self.__FilterData2D(projData)
        
    def __FilterData(self):
        #If we've already done the filtering, return our saved copy
        if ("filteredData" in dir(self)):
            return self.filteredData
        else: #Otherwise do filtering
            self.filteredData = self.__FilterDataFast()
            self.filteredData *= (self.filteredData > 0)
            return self.filteredData

    def __Debounce(self, xs, ys, radius=4):
        if len(xs) <= 1:
            return xs, ys
        
        kdt = kdtree.KDTree(scipy.array([xs,ys]).T)

        xsd = []
        ysd = []

        for xi, yi in zip(xs, ys):
            neigh = kdt.query_ball_point([xi,yi], radius)

            if len(neigh) > 1:
                Ii = self.filteredData[xi,yi]

                In = self.filteredData[xs[neigh].astype('i'),ys[neigh].astype('i')].max()

                if not Ii < In:
                    xsd.append(xi)
                    ysd.append(yi)

            else:
                xsd.append(xi)
                ysd.append(yi)

        return xsd, ysd





    def FindObjects(self, thresholdFactor, numThresholdSteps="default", blurRadius=1.5, mask=None, splitter=None, debounceRadius=4):
        """Finds point-like objects by subjecting the data to a band-pass filtering (as defined when 
        creating the identifier) followed by z-projection and a thresholding procedure where the 
        threshold is progressively decreased from a maximum value (half the maximum intensity in the image) to a 
        minimum defined as [thresholdFactor]*the mode (most frequently occuring value, 
        should correspond to the background) of the image. The number of steps can be given as 
        [numThresholdSteps], with defualt being 5 when filterMode="fast" and 10 for filterMode="good".
        At each step the thresholded image is blurred with a Gaussian of radius [blurRadius] to 
        approximate the image of the points found in that step, and subtracted from the original, thus
        removing the objects from the image such that they are not detected at the lower thresholds.
        This allows the detection of objects which are relatively close together and spread over a 
        large range of intenstities. A binary mask [mask] may be applied to the image to specify a region
        (e.g. a cell) in which objects are to be detected.
        A copy of the filtered image is saved such that subsequent calls to FindObjects with, e.g., a
        different thresholdFactor are faster."""
        
        #save a copy of the parameters.
        self.thresholdFactor = thresholdFactor
        self.estSN = False
        
        if (numThresholdSteps == "default"):
            self.numThresholdSteps = 5
        elif (numThresholdSteps == 'Estimate S/N'):
            self.numThresholdSteps = 0
            self.estSN = True
        else:
            self.numThresholdSteps = int(numThresholdSteps)
            
        self.blurRadius = blurRadius
        self.mask = mask
        #clear the list of previously found points
        del self[:]
        
        #do filtering
        filteredData = self.__FilterData()
        
        #apply mask
        if not (self.mask is None):
            maskedFilteredData = filteredData*self.mask
        else:
            maskedFilteredData = filteredData
        #manually mask the edge pixels
        maskedFilteredData[:, :5] = 0
        maskedFilteredData[:, -5:] = 0
        maskedFilteredData[-5:, :] = 0
        maskedFilteredData[:5, :] = 0
        
        if self.numThresholdSteps > 0:
            #determine (approximate) mode
            N, bins = scipy.histogram(maskedFilteredData, bins=200)
            posMax = N.argmax() #find bin with maximum number of counts
            modeApp = bins[posMax:(posMax+1)].mean() #bins contains left-edges - find middle of most frequent bin
            #catch the corner case where the mode could be zero - this is highly unlikely, but if it were
            #to occur one would no longer be able to influence the threshold with threshFactor
            if (abs(modeApp) < 1): 
                modeApp = 1
            #calc thresholds
            self.lowerThreshold = modeApp*self.thresholdFactor
            self.upperThreshold = maskedFilteredData.max()/2
            
        else:
            if self.estSN:
                self.lowerThreshold = self.thresholdFactor*scipy.sqrt(scipy.median(self.data.ravel()))
            else:
                if type(self.thresholdFactor) == float:
                    self.lowerThreshold = self.thresholdFactor
                else:
                    self.lowerThreshold = scipy.absolute(self.__FilterThresh2D(self.thresholdFactor))
        
        X,Y = scipy.mgrid[0:maskedFilteredData.shape[0], 0:maskedFilteredData.shape[1]]
        X = X.astype('f')
        Y = Y.astype('f')
    
        #store x, y, and thresholds
        xs = []
        ys = []
        ts = []

        if (self.numThresholdSteps == 0): #don't do threshold scan - just use lower threshold (faster)
            im = maskedFilteredData
            (labeledPoints, nLabeled) = ndimage.label(im > self.lowerThreshold)
            
            objSlices = ndimage.find_objects(labeledPoints)
            
            #loop over objects
            for i in range(nLabeled):
                #measure position
                #x,y = ndimage.center_of_mass(im, labeledPoints, i)
                imO = im[objSlices[i]]
                x = (X[objSlices[i]]*imO).sum()/imO.sum()
                y = (Y[objSlices[i]]*imO).sum()/imO.sum()

                #and add to list
                #self.append(OfindPoint(x,y,detectionThreshold=self.lowerThreshold))
                xs.append(x)
                ys.append(y)
                ts.append(self.lowerThreshold)
        else: #do threshold scan (default)
            #generate threshold range - note slightly awkard specification of lowwer and upper bounds as the stop bound is excluded from arange
            self.thresholdRange = scipy.arange(self.upperThreshold, self.lowerThreshold - (self.upperThreshold - self.lowerThreshold)/(self.numThresholdSteps -1), - (self.upperThreshold - self.lowerThreshold)/(self.numThresholdSteps))
            #get a working copy of the filtered data
            im = maskedFilteredData.copy()
        
            #use for quickly deterimining the number of pixels in a slice (there must be a better way)
            corrWeightRef = scipy.ones(im.shape)
            
        
            for threshold in self.thresholdRange:
                #apply threshold and label regions
                (labeledPoints, nLabeled) = ndimage.label(im > threshold)
                #initialise correction weighting mask
                corrWeights = scipy.zeros(im.shape, 'f')
            
                #get 'adress' of each object
                objSlices = ndimage.find_objects(labeledPoints)
                #loop over objects
                for i in range(1, nLabeled):
                    #measure position
                    #x,y = ndimage.center_of_mass(im, labeledPoints, i)
                    nPixels = corrWeightRef[objSlices[i]].sum()
                    imO = im[objSlices[i]]
                    x = (X[objSlices[i]]*imO).sum()/imO.sum()
                    y = (Y[objSlices[i]]*imO).sum()/imO.sum()
                    #and add to list
                    #self.append(OfindPoint(x,y,detectionThreshold=threshold))
                    xs.append(x)
                    ys.append(y)
                    ts.append(threshold)

                    #now work out weights for correction image (N.B. this is somewhat emperical)
                    corrWeights[objSlices[i]] = 1.0/scipy.sqrt(nPixels)
                #calculate correction matrix
                corr = ndimage.gaussian_filter(2*self.blurRadius*scipy.sqrt(2*scipy.pi)*1.7*im*corrWeights, self.blurRadius)
                #subtract from working image
                im -= corr
                #pylab.figure()
                #pylab.imshow(corr)
                #pylab.colorbar()
                #pylab.figure()
                #pylab.imshow(im)
                #pylab.colorbar()
                #clip border pixels again
                im[0:5, 0:5] = 0
                im[0:5, -5:] = 0
                im[-5:, -5:] = 0
                im[-5:, 0:5] = 0

                #print len(xs)

        xs = scipy.array(xs)
        ys = scipy.array(ys)

        #print len(xs)

        if splitter:
            ys = ys + (ys > im.shape[1]/2)*(im.shape[1] - 2*ys)

        xs, ys = self.__Debounce(xs, ys, debounceRadius)

        for x, y, t in zip(xs, ys, ts):
            self.append(OfindPoint(x,y,t))


        #create pseudo lists to allow indexing along the lines of self.x[i]
        self.x = PseudoPointList(self, 'x')
        self.y = PseudoPointList(self, 'y')
