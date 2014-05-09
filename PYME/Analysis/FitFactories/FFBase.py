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
        
    def getSplitROIAtPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        '''Helper fcn to extract ROI from frame at given x,y, point from a multi-channel image. 
        
        Returns:
            Xg - x coordinates of pixels in ROI in nm (channel 1)
            Yg - y coordinates of pixels in ROI (chanel 1)
            Xr - x coordinates of pixels in ROI in nm (channel 2)
            Yr - y coordinates of pixels in ROI (chanel 2)
            data - raw pixel data of ROI
            background - extimated background for ROI
            sigma - estimated error (std. dev) of pixel values
            xslice - x slice into original data array used to get ROI (channel 1)
            yslice - y slice into original data array (channel 1)
            xslice2 - x slice into original data array used to get ROI (channel 2)
            yslice2 - y slice into original data array (channel 2)
        '''
        
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
        

         #cut region out of data stack
        dataROI = self.data[xslice, yslice, 0:2] - self.metadata.Camera.ADOffset
        dataROI[:,:,1] = self.data[xslice2, yslice2, 1] - self.metadata.Camera.ADOffset
        
        nSlices = 1
        sigma = np.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*np.maximum(dataROI, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount


        if not self.background == None and len(np.shape(self.background)) > 1 and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
            bgROI = self.background[xslice, yslice, 0:2] - self.metadata.Camera.ADOffset
            bgROI[:,:,1] = self.background[xslice2, yslice2, 1] - self.metadata.Camera.ADOffset
        else:
            bgROI = 0

 

        #generate grid to evaluate function on        
        Xg = vx*np.mgrid[xslice]
        Yg = vy*np.mgrid[yslice]

        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...

        #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata.chroma.dx,self.metadata.chroma.dy)
        

        Xr = Xg + DeltaX - vx*dxp
        Yr = Yg + DeltaY - vy*dyp
        
            
        return Xg, Yg, Xr, Yr, dataROI, bgROI, sigma, xslice, yslice, xslice2, yslice2

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        '''This should be overridden in derived classes to actually do the fitting.
        The function which gets implemented should return a numpy record array, of the
        dtype defined in the module level FitResultsDType variable (the calling function
        uses FitResultsDType to pre-allocate an array for the results)'''
        
        raise NotImplementedError('This function should be over-ridden in derived class')