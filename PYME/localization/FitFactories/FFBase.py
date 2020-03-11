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
# from scipy import ndimage

from PYME.IO.MetaDataHandler import get_camera_roi_origin

class FFBase(object):
    def __init__(self, data, metadata, fitfcn=None, background=None, noiseSigma=None, roi_offset=[0,0]):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        self.data = data
        self.background = background
        self.metadata = metadata
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        self.noiseSigma = noiseSigma
        self.roi_offset = roi_offset # offset (x, y) from camera ROI to permit best common ROI for both channels when splitting
        self._pixel_size = None
        self._conversion_factor = 1e3  # um to nm

    @property
    def pixel_size(self):
        # return the pixel size in nm
        if self._pixel_size is None:
            self._pixel_size = []
            self._pixel_size.append(self._conversion_factor*self.metadata.voxelsize.x)
            self._pixel_size.append(self._conversion_factor*self.metadata.voxelsize.y)
            self._pixel_size.append(self._conversion_factor*self.metadata.voxelsize.z)
        return self._pixel_size
    
    def getData(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15, dim=2, multichannel=False):

        # Create absolute bounds
        xl0 = (x - roiHalfSize)
        xl1 = 0
        xu0 = (x + roiHalfSize + 1)
        xu1 = self.data.shape[0]
        yl0 = (y - roiHalfSize)
        yl1 = 0
        yu0 = (y + roiHalfSize + 1)
        yu1 = self.data.shape[1]

        if multichannel:
            #find ROI which works in both channels
            #look up shifts
            if not self.metadata.getOrDefault('Analysis.FitShifts', False):
                #pixel size in nm
                vx, vy, _ = self.pixel_size
                #position in nm from camera origin
                roi_x0, roi_y0 = get_camera_roi_origin(self.metadata)
                x_ = (x + roi_x0) * vx
                y_ = (y + roi_y0) * vy
                DeltaX = self.metadata.chroma.dx.ev(x_, y_)
                DeltaY = self.metadata.chroma.dy.ev(x_, y_)
                if DeltaX != 0:
                    #find shift in whole pixels
                    dxp = int(DeltaX/vx)
                    xl1 = max(0, dxp)
                    xu1 = self.data.shape[0] + min(0, dxp)
                if DeltaY != 0:
                    dyp = int(DeltaY/vy)
                    yl1 = max(0, dyp)
                    yu1 = self.data.shape[1] + min(0, dyp)
            else:
                DeltaX = 0
                DeltaY = 0
                dxp = 0
                dyp = 0
            
            # TODO: These lines have extra calculations of xl, yl,
            # but it's probably still cheaper than another if multichannel statement
            xu0 = max(max(xl0, xl1),x+roiHalfSize+1)
            yu0 = max(max(yl0, yl1),y+roiHalfSize+1)

        # Create slice bounds
        xl = max(xl0, xl1)
        xu = min(xu0, xu1)
        yl = max(yl0, yl1)
        yu = min(yu0, yu1)
        zl = max((z - axialHalfSize), 0)
        zu = min((z + axialHalfSize + 1), self.data.shape[2])

        xl, xu, yl, yu, zl, zu = int(xl), int(xu), int(yl), int(yu), int(zl), int(zu)

        # create slices
        xslice = slice(xl,xu)
        yslice = slice(yl,yu)

        if multichannel:
            # Nooo, another multichannel if statement
            xl2 = xl - dxp
            xu2 = yu - dxp
            yl2 = yl - dyp
            yu2 = yu - dyp
            xl2, xu2, yl2, yu2 = int(xl2), int(xu2), int(yl2), int(yu2)
            xslice2 = slice(xl2, xu2)
            yslice2 = slice(yl2, yu2)

            #cut region out of data stack
            dataROI = np.copy(self.data[xslice, yslice, 0:2])
            #print dataROI.shape
            dataROI[:,:,1] = self.data[xslice2, yslice2, 1]

            # Force nSlices to be 1
            zslice = slice(0,1)

            return dataROI, xslice, yslice, zslice, xslice2, yslice2, DeltaX, DeltaY
        else:
            zslice = slice(zl,zu)
            dataROI = self.data[xslice, yslice, zslice]

            if dim == 2:
                dataROI = dataROI.mean(2)

            return dataROI, xslice, yslice, zslice

    def getGrid(self, xslice, yslice, zslice, DeltaX=None, DeltaY=None, dim=2, multichannel=False):
        #pixel size in nm
        vx, vy, vz = self.pixel_size

        # Create a grid to evaluate on
        if dim == 2:
            X = np.mgrid[xslice]
            Y = np.mgrid[yslice]
            Z = None
        elif dim==3:    
            X, Y, Z = np.mgrid[xslice, yslice, zslice]
            Z = vz*Z

        X = vx*(X + self.roi_offset[0])
        Y = vy*(Y + self.roi_offset[1])

        if multichannel:
            dxp, dyp = int(DeltaX/vx), int(DeltaY/vy)
            Xr = X + DeltaX - vx*dxp
            Yr = Y + DeltaY - vy*dyp

            return X, Y, Xr, Yr

        return X, Y, Z

    def getSigma(self, dataROI, xslice, yslice, zslice, xslice2=None, yslice2=None, multichannel=False):
        # nSlices = dataROI.shape[2]
        nSlices = (zslice.stop - zslice.start)
        if zslice.step is not None:
            nSlices = int(round(nSlices/zslice.step))

        if self.noiseSigma is None:
            sigma = (np.sqrt(self.metadata.Camera.ReadNoise**2 + 
                           (self.metadata.Camera.NoiseFactor**2)*
                           self.metadata.Camera.ElectronsPerCount*
                           self.metadata.Camera.TrueEMGain*
                           (np.maximum(dataROI, 1) + 1)/nSlices)/
                           self.metadata.Camera.ElectronsPerCount)
        else:
            if multichannel:
                sigma = self.noiseSigma[xslice, yslice, 0:2]
                sigma[:,:,1] = self.noiseSigma[xslice2, yslice2, 1]
            else:
                sigma = self.noiseSigma[xslice, yslice, zslice]

        if multichannel:
            from scipy import ndimage
            sigma = ndimage.maximum_filter(sigma, [3,3,0])
        
        return sigma

    def getBackground(self, dataROI, xslice, yslice, zslice, xslice2=None, yslice2=None, dim=2, multichannel=False):
        if not self.background is None and len(np.shape(self.background)) > 1 and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
            if multichannel:
                bgROI = self.background[xslice, yslice, 0:2]
                bgROI[:,:,1] = self.background[xslice2, yslice2, 1]
            else:
                bgROI = self.background[xslice, yslice, zslice]
                if dim == 2:
                    bgROI = bgROI.mean(2)
        else: 
            bgROI = np.zeros_like(dataROI)
        return bgROI

    def getROIAtPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15, dim=2):
        """Helper fcn to extract ROI from frame at given x,y, point.
        
        Returns:
            X - x coordinates of pixels in ROI in nm
            Y - y coordinates of pixels in ROI
            Z - z coordinates of ROI in nm (optional)
            data - raw pixel data of ROI
            background - extimated background for ROI
            sigma - estimated error (std. dev) of pixel values
            xslice - x slice into original data array used to get ROI
            yslice - y slice into original data array
            zslice - z slice into original data array
        """
        
        if (dim !=2) and (dim != 3):
            raise ValueError('Invalid dimension {}'.format(dim))

        # x = int(round(x))
        # y = int(round(y))

        # #pixel size in nm
        # vx = 1e3 * self.metadata.voxelsize.x
        # vy = 1e3 * self.metadata.voxelsize.y
        
        # roiHalfSize = int(roiHalfSize)
        
        if (z is None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()

        # TODO: Do we need this round? Or is floor acceptable?
        x = round(x)
        y = round(y)
        z = round(z)

        x, y, z, roiHalfSize, axialHalfSize = int(x), int(y), int(z), int(roiHalfSize), int(axialHalfSize)

        # xslice = slice(int(max((x - roiHalfSize), 0)),int(min((x + roiHalfSize + 1),self.data.shape[0])))
        # yslice = slice(int(max((y - roiHalfSize), 0)),int(min((y + roiHalfSize + 1), self.data.shape[1])))
        # zslice = slice(int(max((z - axialHalfSize), 0)),int(min((z + axialHalfSize + 1), self.data.shape[2])))
        
        # dataROI = self.data[xslice, yslice, zslice]

        # #average in z
        # dataMean = dataROI.mean(2)

        dataMean, xslice, yslice, zslice = self.getData(x, y, z, roiHalfSize, axialHalfSize, dim=dim)

        # #generate grid to evaluate function on        
        # X = vx*(np.mgrid[xslice] + self.roi_offset[0])
        # Y = vy*(np.mgrid[yslice] + self.roi_offset[1])

        X, Y, Z = self.getGrid(xslice, yslice, zslice, dim=dim)
	
        # #estimate errors in data
        # nSlices = dataROI.shape[2]
        
        # #sigma = np.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*np.maximum(dataMean, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount
        # ### Fixed for better Poisson noise approx
        # if self.noiseSigma is None:
        #     sigma = np.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*(np.maximum(dataMean, 1) + 1)/nSlices)/self.metadata.Camera.ElectronsPerCount
        # else:
        #     sigma = self.noiseSigma[xslice, yslice, zslice]

        sigma = self.getSigma(dataMean, xslice, yslice, zslice)

        # if not self.background is None and len(np.shape(self.background)) > 1 and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
        #     bgROI = self.background[xslice, yslice, zslice]

        #     #average in z
        #     bgMean = bgROI.mean(2)
        # else: 
        #     bgMean = 0

        bgMean = self.getBackground(dataMean, xslice, yslice, zslice, dim=dim)
        
        if dim == 2:
            return X, Y, dataMean, bgMean, sigma, xslice, yslice, zslice
        elif dim == 3:
            return X, Y, Z, dataMean, bgMean, sigma, xslice, yslice, zslice
        
    def getSplitROIAtPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        """Helper fcn to extract ROI from frame at given x,y, point from a multi-channel image.
        
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
        """
        
        # x = round(x)
        # y = round(y)

        if (z is None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()

        # TODO: Do we need this round? Or is floor acceptable?
        x = round(x)
        y = round(y)
        z = round(z)

        x, y, z, roiHalfSize, axialHalfSize = int(x), int(y), int(z), int(roiHalfSize), int(axialHalfSize)

        # #pixel size in nm
        # vx = 1e3*self.metadata.voxelsize.x
        # vy = 1e3*self.metadata.voxelsize.y
        
        # #position in nm from camera origin
        # roi_x0, roi_y0 = get_camera_roi_origin(self.metadata)
        # x_ = (x + roi_x0)*vx
        # y_ = (y + roi_y0)*vy
        
        
        # #look up shifts
        # if not self.metadata.getOrDefault('Analysis.FitShifts', False):
        #     DeltaX = self.metadata.chroma.dx.ev(x_, y_)
        #     DeltaY = self.metadata.chroma.dy.ev(x_, y_)
        # else:
        #     DeltaX = 0
        #     DeltaY = 0
        
        # #find shift in whole pixels
        # dxp = int(DeltaX/vx)
        # dyp = int(DeltaY/vy)
        
        # #find ROI which works in both channels
        # #if dxp < 0:
        # x01 = max(x - roiHalfSize, max(0, dxp))
        # x11 = min(max(x01, x + roiHalfSize + 1), self.data.shape[0] + min(0, dxp))
        # x02 = x01 - dxp
        # x12 = x11 - dxp
        
        # y01 = max(y - roiHalfSize, max(0, dyp))
        # y11 = min(max(y + roiHalfSize + 1,  y01), self.data.shape[1] + min(0, dyp))
        # y02 = y01 - dyp
        # y12 = y11 - dyp
        
        # xslice = slice(int(x01), int(x11))
        # xslice2 = slice(int(x02), int(x12))
        
        # yslice = slice(int(y01), int(y11))
        # yslice2 = slice(int(y02), int(y12))
        
        # #print xslice2, yslice2
        

        #  #cut region out of data stack
        # dataROI = np.copy(self.data[xslice, yslice, 0:2])
        # #print dataROI.shape
        # dataROI[:,:,1] = self.data[xslice2, yslice2, 1]
        
        # nSlices = 1

        dataROI, xslice, yslice, zslice, xslice2, yslice2, DeltaX, DeltaY = self.getData(x, y, z, roiHalfSize, axialHalfSize, dim=2, multichannel=True)

        #sigma = np.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*np.maximum(dataROI, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount
        #phConv = self.metadata.Camera.ElectronsPerCount/self.metadata.Camera.TrueEMGain
        #nPhot = dataROI*phConv
        
        # if self.noiseSigma is None:
        #     sigma = np.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*(self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*np.maximum(dataROI, 1) + self.metadata.Camera.TrueEMGain*self.metadata.Camera.TrueEMGain))/self.metadata.Camera.ElectronsPerCount
        # else:
        #     sigma = self.noiseSigma[xslice, yslice, 0:2]
        #     sigma[:,:,1] = self.noiseSigma[xslice2, yslice2, 1]
            
        # sigma = ndimage.maximum_filter(sigma, [3,3,0])

        sigma = self.getSigma(dataROI, xslice, yslice, zslice, xslice2, yslice2, multichannel=True)

        # if self.metadata.getOrDefault('Analysis.subtractBackground', True) :
        #     #print 'bgs'
        #     if not self.background is None and len(np.shape(self.background)) > 1:
        #         bgROI = self.background[xslice, yslice, 0:2]
        #         bgROI[:,:,1] = self.background[xslice2, yslice2, 1]
        #     else:
        #         bgROI = np.zeros_like(dataROI) + (self.background if self.background else 0)
        # else:
        #     bgROI = np.zeros_like(dataROI)

        bgROI = self.getBackground(dataROI, xslice, yslice, zslice, xslice2, yslice2, dim=2, multichannel=True)

 

        # #generate grid to evaluate function on        
        # Xg = vx*(np.mgrid[xslice] + self.roi_offset[0])
        # Yg = vy*(np.mgrid[yslice] + self.roi_offset[1])

        # #generate a corrected grid for the red channel
        # #note that we're cheating a little here - for shifts which are slowly
        # #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        # #similarly for y. For slowly varying shifts the following should be
        # #equivalent to this. For rapidly varying shifts all bets are off ...

        # #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata.chroma.dx,self.metadata.chroma.dy)
        

        # Xr = Xg + DeltaX - vx*dxp
        # Yr = Yg + DeltaY - vy*dyp

        Xg, Yg, Xr, Yr = self.getGrid(xslice, yslice, zslice, DeltaX, DeltaY, dim=2, multichannel=True)
        
            
        return Xg, Yg, Xr, Yr, dataROI, bgROI, sigma, xslice, yslice, xslice2, yslice2

    def getMultiviewROIAtPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        # TODO: Identical to getSplitROIAtPoint
        """Helper fcn to extract ROI from frame at given x,y, point from a multi-channel image.

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
        """
    
        # x = round(x)
        # y = round(y)

        if (z is None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()

        # TODO: Do we need this round? Or is floor acceptable?
        x = round(x)
        y = round(y)
        z = round(z)

        x, y, z, roiHalfSize, axialHalfSize = int(x), int(y), int(z), int(roiHalfSize), int(axialHalfSize)
    
        # #pixel size in nm
        # vx = 1e3 * self.metadata.voxelsize.x
        # vy = 1e3 * self.metadata.voxelsize.y
    
        # #position in nm from camera origin
        # roi_x0, roi_y0 = get_camera_roi_origin(self.metadata)
        # x_ = (x + roi_x0) * vx
        # y_ = (y + roi_y0) * vy
    
        # #look up shifts
        # if not self.metadata.getOrDefault('Analysis.FitShifts', False):
        #     DeltaX = self.metadata.chroma.dx.ev(x_, y_)
        #     DeltaY = self.metadata.chroma.dy.ev(x_, y_)
        # else:
        #     DeltaX = 0
        #     DeltaY = 0
    
        # #find shift in whole pixels
        # dxp = int(DeltaX / vx)
        # dyp = int(DeltaY / vy)
    
        # #find ROI which works in both channels
        # #if dxp < 0:
        # x01 = max(x - roiHalfSize, max(0, dxp))
        # x11 = min(max(x01, x + roiHalfSize + 1), self.data.shape[0] + min(0, dxp))
        # x02 = x01 - dxp
        # x12 = x11 - dxp
    
        # y01 = max(y - roiHalfSize, max(0, dyp))
        # y11 = min(max(y + roiHalfSize + 1, y01), self.data.shape[1] + min(0, dyp))
        # y02 = y01 - dyp
        # y12 = y11 - dyp
    
        # xslice = slice(int(x01), int(x11))
        # xslice2 = slice(int(x02), int(x12))
    
        # yslice = slice(int(y01), int(y11))
        # yslice2 = slice(int(y02), int(y12))
    
        # #print xslice2, yslice2
    
    
        # #cut region out of data stack
        # dataROI = self.data[xslice, yslice, 0:2]
        # #print dataROI.shape
        # dataROI[:, :, 1] = self.data[xslice2, yslice2, 1]
    
        # nSlices = 1

        dataROI, xslice, yslice, zslice, xslice2, yslice2, DeltaX, DeltaY = self.getData(x, y, z, roiHalfSize, axialHalfSize, dim=2, multichannel=True)
        #sigma = np.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*np.maximum(dataROI, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount
        #phConv = self.metadata.Camera.ElectronsPerCount/self.metadata.Camera.TrueEMGain
        #nPhot = dataROI*phConv
    
        # if self.noiseSigma is None:
        #     sigma = np.sqrt(self.metadata.Camera.ReadNoise ** 2 + (self.metadata.Camera.NoiseFactor ** 2) * (
        #     self.metadata.Camera.ElectronsPerCount * self.metadata.Camera.TrueEMGain * np.maximum(dataROI,
        #                                                                                           1) + self.metadata.Camera.TrueEMGain * self.metadata.Camera.TrueEMGain)) / self.metadata.Camera.ElectronsPerCount
        # else:
        #     sigma = self.noiseSigma[xslice, yslice, 0:2]
        #     sigma[:, :, 1] = self.noiseSigma[xslice2, yslice2, 1]
    
        # sigma = ndimage.maximum_filter(sigma, [3, 3, 0])

        sigma = self.getSigma(dataROI, xslice, yslice, zslice, xslice2, yslice2, multichannel=True)
    
        # if self.metadata.getOrDefault('Analysis.subtractBackground', True):
        #     #print 'bgs'
        #     if not self.background is None and len(np.shape(self.background)) > 1:
        #         bgROI = self.background[xslice, yslice, 0:2]
        #         bgROI[:, :, 1] = self.background[xslice2, yslice2, 1]
        #     else:
        #         bgROI = np.zeros_like(dataROI) + self.background
        # else:
        #     bgROI = np.zeros_like(dataROI)

        bgROI = self.getBackground(dataROI, xslice, yslice, zslice, xslice2, yslice2, dim=2, multichannel=True)
    
        # #generate grid to evaluate function on
        # Xg = vx * (np.mgrid[xslice] + self.roi_offset[0])
        # Yg = vy * (np.mgrid[yslice] + self.roi_offset[1])

    
        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...
    
        #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata.chroma.dx,self.metadata.chroma.dy)
    
    
        # Xr = Xg + DeltaX - vx * dxp
        # Yr = Yg + DeltaY - vy * dyp

        Xg, Yg, Xr, Yr = self.getGrid(xslice, yslice, zslice, DeltaX, DeltaY, dim=2, multichannel=True)
    
        return Xg, Yg, Xr, Yr, dataROI, bgROI, sigma, xslice, yslice, xslice2, yslice2

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        """This should be overridden in derived classes to actually do the fitting.
        The function which gets implemented should return a numpy record array, of the
        dtype defined in the module level FitResultsDType variable (the calling function
        uses FitResultsDType to pre-allocate an array for the results)"""
        
        raise NotImplementedError('This function should be over-ridden in derived class')


FitFactory = FFBase
FitResultsDType = None  # Implemented in a derived class
