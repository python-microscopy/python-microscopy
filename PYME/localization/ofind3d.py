#!/usr/bin/python

###############
# ofind3d.py
#
# Copyright David Baddeley, 2012
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
################
#!/usr/bin/python

##################
# ofind3d.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import scipy
import scipy.ndimage as ndimage
# import pylab
import numpy as np
#from PYME.DSView import View3D

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
    def __init__(self, data, filterRadiusLowpass=1, filterRadiusHighpass=3, filterRadiusLowpassZ=1, filterRadiusHighpassZ = 1):
        """Creates an Identifier object to be used for object finding, takes a 2D or 3D slice
        into a data stack (data). The parameters filterRadiusLowpass and filterRadiusHighpass control the bandpass filter
        used to identify 'point-like' features. """

        self.data = data
        
        self.filterRadiusLowpass = filterRadiusLowpass
        self.filterRadiusHighpass = filterRadiusHighpass
        self.filterRadiusLowpassZ = filterRadiusLowpassZ
        self.filterRadiusHighpassZ = filterRadiusHighpassZ

    def __FilterData2D(self,data):
        #lowpass filter to suppress noise
        a = ndimage.gaussian_filter(data.astype('f'), self.filterRadiusLowpass)
        #lowpass filter again to find background
        b = ndimage.gaussian_filter(a, self.filterRadiusHighpass)

        return a - b

    def __FilterData3D(self,data):
        #lowpass filter to suppress noise
        a = ndimage.gaussian_filter(data.astype('f'), [self.filterRadiusLowpass,self.filterRadiusLowpass, self.filterRadiusLowpassZ] )
        #lowpass filter again to find background
        b = ndimage.gaussian_filter(a, [self.filterRadiusHighpass, self.filterRadiusHighpass, self.filterRadiusHighpassZ])

        return a - b  
    

    def __FilterData(self):
        #If we've already done the filtering, return our saved copy
        if ("filteredData" in dir(self)):
            return self.filteredData
        else: #Otherwise do filtering
            if (len(self.data.shape) == 3 and self.data.shape[2] > 1):
                self.filteredData = self.__FilterData3D(self.data)
            else:
                self.filteredData = self.__FilterData2D(self.data)
            self.filteredData *= (self.filteredData > 0)
            return self.filteredData


    def FindObjects(self, thresholdFactor, numThresholdSteps="default", blurRadius=1.5, blurRadiusZ=1.5, mask=None):
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
            self.numThresholdSteps = 10
        elif (numThresholdSteps == 'Estimate S/N'):
            self.numThresholdSteps = 0
            self.estSN = True
        else:
            self.numThresholdSteps = int(numThresholdSteps)
            
        self.blurRadius = blurRadius
        self.blurRadiusZ = blurRadiusZ
        self.mask = mask

        #clear the list of previously found points
        del self[:]
        
        #do filtering
        filteredData = np.maximum(self.__FilterData(), 0)
        
        #apply mask
        if not (self.mask is None):
            maskedFilteredData = filteredData*self.mask
        else:
            maskedFilteredData = filteredData

        #manually mask the edge pixels
        if maskedFilteredData.ndim == 3 and maskedFilteredData.shape[2] > 1:
            maskedFilteredData[0:5, 0:5, :] = 0
            maskedFilteredData[0:5, -5:,:] = 0
            maskedFilteredData[-5:, -5:,:] = 0
            maskedFilteredData[-5:, 0:5,:] = 0
            maskedFilteredData[:,:,:3] = 0
            maskedFilteredData[:,:,-3:] = 0
        else:
            maskedFilteredData[0:5, 0:5] = 0
            maskedFilteredData[0:5, -5:] = 0
            maskedFilteredData[-5:, -5:] = 0
            maskedFilteredData[-5:, 0:5] = 0
        
        if self.numThresholdSteps > 0:
            #determine (approximate) mode
            #N, bins = scipy.histogram(maskedFilteredData, bins=200)
            #posMax = N.argmax() #find bin with maximum number of counts
            #modeApp = bins[posMax:(posMax+1)].mean() #bins contains left-edges - find middle of most frequent bin
            
            #modeApp = np.median(maskedFilteredData)

            #catch the corner case where the mode could be zero - this is highly unlikely, but if it were
            #to occur one would no longer be able to influence the threshold with threshFactor
            #if (abs(modeApp) < 1): 
            #    modeApp = 1
            
            modeApp = maskedFilteredData.max()/100.

            #calc thresholds
            self.lowerThreshold = modeApp*self.thresholdFactor 
            self.upperThreshold = maskedFilteredData.max()/2 
            
        else:
            if self.estSN:
                self.lowerThreshold = self.thresholdFactor*scipy.sqrt(scipy.median(self.data.ravel()))
            else:
                self.lowerThreshold = self.thresholdFactor

        
        X,Y,Z = scipy.mgrid[0:maskedFilteredData.shape[0], 0:maskedFilteredData.shape[1],0:maskedFilteredData.shape[2]]
    
        if (self.numThresholdSteps == 0): #don't do threshold scan - just use lower threshold (faster)
            im = maskedFilteredData
            #View3D(im)
            (labeledPoints, nLabeled) = ndimage.label(im > self.lowerThreshold)
            
            objSlices = ndimage.find_objects(labeledPoints)
            
            #loop over objects
            for i in range(nLabeled):
                #measure position
                #x,y = ndimage.center_of_mass(im, labeledPoints, i)
                imO = im[objSlices[i]]
                x = (X[objSlices[i]]*imO).sum()/imO.sum()
                y = (Y[objSlices[i]]*imO).sum()/imO.sum()
                z = (Z[objSlices[i]]*imO).sum()/imO.sum()
                #and add to list
                self.append(OfindPoint(x,y,z,detectionThreshold=self.lowerThreshold))
        else: #do threshold scan (default)

            #generate threshold range - note slightly awkard specification of lowwer and upper bounds as the stop bound is excluded from arange
            #self.thresholdRange = scipy.arange(self.upperThreshold, self.lowerThreshold - (self.upperThreshold - self.lowerThreshold)/(self.numThresholdSteps -1), - (self.upperThreshold - self.lowerThreshold)/(self.numThresholdSteps))
            self.thresholdRange = scipy.logspace(np.log10(self.upperThreshold), np.log10(self.lowerThreshold), self.numThresholdSteps)
            print(('Thresholds:', self.thresholdRange))

            #get a working copy of the filtered data
            im = maskedFilteredData.copy()

        

            #use for quickly deterimining the number of pixels in a slice (there must be a better way)
            corrWeightRef = scipy.ones(im.shape)
            

        
            for threshold in self.thresholdRange:
                #View3D(im)
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
                    z = (Z[objSlices[i]]*imO).sum()/imO.sum()
                    #and add to list
                    self.append(OfindPoint(x,y,z,detectionThreshold=threshold))

                    #now work out weights for correction image (N.B. this is somewhat emperical)
                    corrWeights[objSlices[i]] = 1.0/scipy.sqrt(nPixels)

                #calculate correction matrix
                corr = ndimage.gaussian_filter(((2*self.blurRadius)**2)*(2*self.blurRadiusZ)*1.5*im*corrWeights, [self.blurRadius, self.blurRadius, self.blurRadiusZ])
                #View3D([im, corr, corrWeights])

                #subtract from working image
                im  = np.maximum(im - corr, 0)

                #pylab.figure()
                #pylab.imshow(corr)
                #pylab.colorbar()

                #pylab.figure()
                #pylab.imshow(im)
                #pylab.colorbar()

                #clip border pixels again
                if maskedFilteredData.ndim == 3 and maskedFilteredData.shape[2] > 1:
                    im[0:5, 0:5, :] = 0
                    im[0:5, -5:,:] = 0
                    im[-5:, -5:,:] = 0
                    im[-5:, 0:5,:] = 0
    
                    im[:,:,:3]  = 0
                    im[:,:,-3:] = 0
                else:
                    im[0:5, 0:5] = 0
                    im[0:5, -5:] = 0
                    im[-5:, -5:] = 0
                    im[-5:, 0:5] = 0

                print((len(self)))

        #create pseudo lists to allow indexing along the lines of self.x[i]
        self.x = PseudoPointList(self, 'x')
        self.y = PseudoPointList(self, 'y')
        self.z = PseudoPointList(self, 'z')
