#!/usr/bin/python

##################
# LatGaussFitFR.py
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

import numpy as np
from . import fitCommon

class FitFactory(object):
    def __init__(self, data, metadata, fitfcn=None, background=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. '''
        self.data = data
        self.background = background
        self.metadata = metadata
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        
    def getROIAtPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        '''Helper fcn to extract ROI from frame at given x,y, point. 
        
        Returns:
            X - x coordinates of pixels in ROI in nm
            Y - y coordinates of pixels in ROI
            data - raw pixel data of ROI
            background - extimated background for ROI
            sigma - estimated error (std. dev) of pixel values
            xslice - x slice into original data array used to get ROI
            yslice - y slice into original data array
            zslice - z slice into original data array
        '''
        if (z == None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()

        x = round(x)
        y = round(y)

        xslice = slice(max((x - roiHalfSize), 0),min((x + roiHalfSize + 1),self.data.shape[0]))
        yslice = slice(max((y - roiHalfSize), 0),min((y + roiHalfSize + 1), self.data.shape[1]))
        zslice = slice(max((z - axialHalfSize), 0),min((z + axialHalfSize + 1), self.data.shape[2]))
		
        
        dataROI = self.data[xslice, yslice, zslice]

        #average in z
        dataMean = dataROI.mean(2) - self.metadata.Camera.ADOffset

        #generate grid to evaluate function on        
        X = 1e3*self.metadata.voxelsize.x*np.mgrid[xslice]
        Y = 1e3*self.metadata.voxelsize.y*np.mgrid[yslice]
	
        #estimate errors in data
        nSlices = dataROI.shape[2]
        
        sigma = np.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*np.maximum(dataMean, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount

        if not self.background == None and len(np.shape(self.background)) > 1 and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
            bgROI = self.background[xslice, yslice, zslice]

            #average in z
            bgMean = bgROI.mean(2) - self.metadata.Camera.ADOffset            
        else: 
            bgMean = 0
            
        return X, Y, dataMean, bgMean, sigma, xslice, yslice, zslice

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        '''This should be overridden in derived classes to actually do the fitting.
        The function which gets implemented should return a numpy record array, of the
        dtype defined in the module level FitResultsDType variable (the calling function
        uses FitResultsDType to pre-allocate an array for the results)'''
        
        raise NotImplementedError('This function should be over-ridden in derived class')