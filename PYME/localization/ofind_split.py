#!/usr/bin/python

##################
# ofind.py
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

#import scipy
import numpy
import math
from scipy import ndimage
from scipy.ndimage import _nd_image, _ni_support
from scipy.spatial import ckdtree
#import pylab

def calc_gauss_weights(sigma):
    """calculate a gaussian filter kernel (adapted from scipy.ndimage.filters.gaussian_filter1d)"""
    sd = float(sigma)
    # make the length of the filter equal to 4 times the standard
    # deviations:
    lw = int(4.0 * sd + 0.5)
    weights = numpy.zeros(2 * lw + 1, 'float64')
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp

    return weights/sum

#default filter sizes
filtRadLowpass = 1
filtRadHighpass = 3

#precompute our filter weights
weightsLowpass = calc_gauss_weights(filtRadLowpass)
weightsHighpass = calc_gauss_weights(filtRadHighpass)

#print weightsLowpass.dtype

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
    def __init__(self, data, filterMode="fast", filterRadiusLowpass=1, filterRadiusHighpass=3, filterRadiusZ=4):
        """Creates an Identifier object to be used for object finding, takes a 2D or 3D slice
        into a data stack (data), and a filtering mode (filterMode, one of ["fast", "good"])
        where "fast" performs a z-projection and then filters, wheras "good" filters in 3D before 
        projecting. The parameters filterRadiusLowpass and filterRadiusHighpass control the bandpass filter
        used to identify 'point-like' features. filterRadiusZ is the radius used for the axial smoothing filter"""
        global filtRadLowpass, filtRadHighpass, weightsLowpass, weightsHighpass

        self.data = data
        self.filterMode = filterMode
        self.filterRadiusLowpass = filterRadiusLowpass
        self.filterRadiusHighpass = filterRadiusHighpass
        self.filterRadiusZ = filterRadiusZ

        if not filterRadiusLowpass == filtRadLowpass:
            #recompute weights for different filter size
            filtRadLowpass = filterRadiusLowpass
            weightsLowpass = calc_gauss_weights(filtRadLowpass)

        if not filterRadiusHighpass == filtRadHighpass:
            #recompute weights for different filter size
            filtRadHighpass = filterRadiusHighpass
            weightsHighpass = calc_gauss_weights(filtRadHighpass)


    def __FilterData2D(self,data):
        mode = _ni_support._extend_mode_to_code("reflect")
        #lowpass filter to suppress noise
        #a = ndimage.gaussian_filter(data.astype('f4'), self.filterRadiusLowpass)
        #print data.shape

        output, a = _ni_support._get_output(None, data)
        _nd_image.correlate1d(data, weightsLowpass, 0, output, mode, 0,0)
        _nd_image.correlate1d(output, weightsLowpass, 1, output, mode, 0,0)

        #print numpy.absolute(a - a_).mean()

        #lowpass filter again to find background
        #b = ndimage.gaussian_filter(a, self.filterRadiusHighpass)

        output, b = _ni_support._get_output(None, data)
        _nd_image.correlate1d(data, weightsHighpass, 0, output, mode, 0,0)
        _nd_image.correlate1d(output, weightsHighpass, 1, output, mode, 0,0)

        return a - b

    
    def __FilterDataFast(self):
        #project data
        if len(self.data.shape) == 2: #if already 2D, do nothing
            #projData = self.data
            return [self.__FilterData2D(self.data)]
        else:
            #projData = self.data.max(2) 
            return [self.__FilterData2D(self.data[:,:,i]) for i in range(self.data.shape[2])]

        #return self.__FilterData2D(projData)
        

    def __FilterDataGood(self):
        #lowpass filter to suppress noise
        a = ndimage.gaussian_filter(self.data, (self.filterRadiusLowpass,self.filterRadiusLowpass, self.filterRadiusZ) )
        #lowpass filter again to find background
        b = ndimage.gaussian_filter(a, (self.filterRadiusHighpass, self.filterRadiusHighpass, 0))
    
        #z-projection
        projData = (a.astype('int16') - b.astype('int16')).max(2)

        return self.__FilterData2D(projData) #filter again in 2D (to get rid of the projected components of the cone above and below widefield points)

    def __FilterData(self):
        #If we've already done the filtering, return our saved copy
        if ("filteredData" in dir(self)):
            return self.filteredData
        else: #Otherwise do filtering
            if (self.filterMode == "fast"):
                self.filteredData = self.__FilterDataFast()
            else:
                self.filteredData = self.__FilterDataGood()
            self.filteredData *= (self.filteredData > 0)
            return self.filteredData

    def __Debounce(self, xs, ys, radius=4):
        if len(xs) < 2:
            return xs, ys
        
        kdt = ckdtree.cKDTree(numpy.array([xs,ys]).T)

        xsd = []
        ysd = []

        for xi, yi in zip(xs, ys):
            #neigh = kdt.query_ball_point([xi,yi], radius)
            dn, neigh = kdt.query(numpy.array([xi,yi]), 5)

            neigh = neigh[dn < radius]

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





    def FindObjects(self, thresholdFactor, numThresholdSteps="default", blurRadius=1.5, mask=None, splitter=None, debounceRadius=4, maskEdgeWidth=5):
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
            if (self.filterMode == "fast"):
                self.numThresholdSteps = 5
            else:
                self.numThresholdSteps = 10
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
        if maskEdgeWidth and filteredData.shape[1] > maskEdgeWidth:
            maskedFilteredData[:, :maskEdgeWidth] = 0
            maskedFilteredData[:, -maskEdgeWidth:] = 0
            maskedFilteredData[-maskEdgeWidth:, :] = 0
            maskedFilteredData[:maskEdgeWidth, :] = 0
        
        if self.numThresholdSteps > 0:
            #determine (approximate) mode
            N, bins = numpy.histogram(maskedFilteredData, bins=200)
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
                self.lowerThreshold = self.thresholdFactor*numpy.sqrt(numpy.median(self.data.ravel()))
            else:
                self.lowerThreshold = self.thresholdFactor

        
        X,Y = numpy.mgrid[0:maskedFilteredData.shape[0], 0:maskedFilteredData.shape[1]]
        #X = X.astype('f')
        #Y = Y.astype('f')

        #store x, y, and thresholds
        xs = []
        ys = []
        ts = []
    
        if (self.numThresholdSteps == 0): #don't do threshold scan - just use lower threshold (faster)
            im = maskedFilteredData
            imt = im > self.lowerThreshold
            #imt = ndimage.binary_erosion(im >self.lowerThreshold)
            (labeledPoints, nLabeled) = ndimage.label(imt)
            
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
            self.thresholdRange = numpy.arange(self.upperThreshold, self.lowerThreshold - (self.upperThreshold - self.lowerThreshold)/(self.numThresholdSteps -1), - (self.upperThreshold - self.lowerThreshold)/(self.numThresholdSteps))

            #get a working copy of the filtered data
            im = maskedFilteredData.copy()

        

            #use for quickly deterimining the number of pixels in a slice (there must be a better way)
            corrWeightRef = numpy.ones(im.shape)
            

        
            for threshold in self.thresholdRange:
                #apply threshold and label regions
                (labeledPoints, nLabeled) = ndimage.label(im > threshold)

                #initialise correction weighting mask
                corrWeights = numpy.zeros(im.shape, 'f')
            
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
                    corrWeights[objSlices[i]] = 1.0/numpy.sqrt(nPixels)

                #calculate correction matrix
                corr = ndimage.gaussian_filter(2*self.blurRadius*numpy.sqrt(2*numpy.pi)*1.7*im*corrWeights, self.blurRadius)

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

                print((len(xs)))

        xs = numpy.array(xs)
        ys = numpy.array(ys)

       # if splitter:
       #     ys = ys + (ys > im.shape[1]/2)*(im.shape[1] - 2*ys)

        xs, ys = self.__Debounce(xs, ys, debounceRadius)

        for x, y, t in zip(xs, ys, ts):
            self.append(OfindPoint(x,y,t))


        #create pseudo lists to allow indexing along the lines of self.x[i]
        self.x = PseudoPointList(self, 'x')
        self.y = PseudoPointList(self, 'y')
