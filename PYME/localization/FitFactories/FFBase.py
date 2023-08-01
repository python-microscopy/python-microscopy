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
from scipy import ndimage
import six

from PYME.IO.MetaDataHandler import get_camera_roi_origin
from PYME.Analysis.points import twoColour

def get_shiftmap(shiftmap):
    ' Get shiftmap from metadata entry. Module level function to permit future caching'
    
    if isinstance(shiftmap, six.string_types):
        return twoColour.ShiftModel.from_md_entry(shiftmap)
    else:
        return shiftmap

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
        
        self._shift_x = None
        self._shift_y = None
        
    @property
    def shift_x(self):
        if self._shift_x is None:
            self._shift_x = get_shiftmap(self.metadata['chroma.dx'])
            
        return self._shift_x

    @property
    def shift_y(self):
        if self._shift_y is None:
            self._shift_y = get_shiftmap(self.metadata['chroma.dy'])
    
        return self._shift_y
        
    def _calc_sigma(self, data, n_slices_averaged=1):
        """ NOTE: This is a fallback and will normally not be used - fit factories should get noiseSigma passed in from
        remFitBuf which uses camera maps if available. Refer to the `calcSigma()` method in remFitBuf for details.
        """
        read_noise, noise_factor,e_per_count, em_gain = float(self.metadata.Camera.ReadNoise), float(self.metadata.Camera.NoiseFactor), float(self.metadata.Camera.ElectronsPerCount), float(self.metadata.Camera.TrueEMGain)
        
        return np.sqrt(read_noise ** 2 + (noise_factor ** 2) * e_per_count * em_gain * (np.maximum(data, 1) + em_gain**2) / n_slices_averaged) / e_per_count
    
    def _get_roi(self, x, y, z, roiHalfSize, axialHalfSize):
        """ common code between 2D and 3D  ROI extraction"""
        x = int(round(x))
        y = int(round(y))
        if (z is None): # use position of maximum intensity
            z = self.data[x, y, :].argmax()
    
        roiHalfSize = int(roiHalfSize)
        axialHalfSize = int(axialHalfSize)
    
        xslice = slice(int(max((x - roiHalfSize), 0)), int(min((x + roiHalfSize + 1), self.data.shape[0])))
        yslice = slice(int(max((y - roiHalfSize), 0)), int(min((y + roiHalfSize + 1), self.data.shape[1])))
        zslice = slice(int(max((z - axialHalfSize), 0)), int(min((z + axialHalfSize + 1), self.data.shape[2])))
    
        data = self.data[xslice, yslice, zslice]
        sigma = self.noiseSigma[xslice, yslice, zslice] if (self.noiseSigma is not None) else None
        if (not self.background is None) and (not np.isscalar(self.background)) and (self.metadata.get('Analysis.subtractBackground', True)):
            background = self.background[xslice, yslice, zslice]
        else:
            background = 0
            
        return xslice, yslice, zslice, data, sigma, background
        

    def getROIAtPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        """Helper fcn to extract ROI from frame at given x,y, point.

        Parameters
        ----------
        x : int
            ROI center position, x [pixels] relative to self.roi_offset
        y : int 
            ROI center position, y [pixels] relative to self.roi_offset
        z : int
            ROI center position, z [pixels or frame]. Optional
        roiHalfSize : int
            lateral ROI extent. Lateral ROI size will be (2 * roiHalfSize) + 1
        axialHalfSize : int
            axial ROI extent. Axial ROI size will be (2 * axialHalfSize) + 1
        
        Returns
        -------
            X - x coordinates of pixels in ROI in nm
            Y - y coordinates of pixels in ROI
            data - raw pixel data of ROI, averaged in Z if axialHalfSize is > 0 and
                self.data dim 2 > 1 (which it usually is not).
            background - extimated background for ROI
            sigma - estimated error (std. dev) of pixel values
            xslice - x slice into original data array used to get ROI
            yslice - y slice into original data array
            zslice - z slice into original data array
        """
        xslice, yslice, zslice, dataROI, sigma, bgROI = self._get_roi(x, y, z, roiHalfSize, axialHalfSize)


        #average in z
        dataMean = dataROI.mean(2)
        bgMean = bgROI if np.isscalar(bgROI) else bgROI.mean(2)
        
        if sigma is None:
            #estimate errors in data
            sigma = self._calc_sigma(dataMean, n_slices_averaged=dataROI.shape[2])

        #pixel size in nm
        vx, vy, _ = self.metadata.voxelsize_nm
        
        #generate grid to evaluate function on
        X = vx * (np.mgrid[xslice] + self.roi_offset[0])
        Y = vy * (np.mgrid[yslice] + self.roi_offset[1])
            
        return X, Y, dataMean, bgMean, sigma, xslice, yslice, zslice

    def get3DROIAtPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        """Helper fcn to extract ROI from frame at given x,y, point.

        Returns:
            X - x coordinates of pixels in ROI in nm
            Y - y coordinates of pixels in ROI
            Z - z coordinates of pixels in ROI
            data - raw pixel data of ROI
            background - extimated background for ROI
            sigma - estimated error (std. dev) of pixel values
            xslice - x slice into original data array used to get ROI
            yslice - y slice into original data array
            zslice - z slice into original data array
        """
        xslice, yslice, zslice, dataROI, sigma, bgROI = self._get_roi(x, y, z, roiHalfSize, axialHalfSize)
    
        if sigma is None:
            sigma = self._calc_sigma(dataROI)

        #pixel size in nm
        vx, vy, vz = self.metadata.voxelsize_nm

        #generate grid to evaluate function on
        X = vx * (np.mgrid[xslice] + self.roi_offset[0])
        Y = vy * (np.mgrid[yslice] + self.roi_offset[1])
        Z = vz * (np.mgrid[zslice])
    
        return X, Y, Z, dataROI, bgROI, sigma, xslice, yslice, zslice

        
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
        
        x = round(x)
        y = round(y)
        
        roiHalfSize = int(roiHalfSize)
        
        #pixel size in nm
        vx, vy, _ = self.metadata.voxelsize_nm
        
        #position in nm from camera origin
        roi_x0, roi_y0 = get_camera_roi_origin(self.metadata)
        x_ = (x + roi_x0)*vx
        y_ = (y + roi_y0)*vy
        
        
        #look up shifts
        if not self.metadata.getOrDefault('Analysis.FitShifts', False):
            DeltaX = self.shift_x.ev(x_, y_)
            DeltaY = self.shift_y.ev(x_, y_)
        else:
            DeltaX = 0
            DeltaY = 0
        
        #find shift in whole pixels
        dxp = int(DeltaX/vx)
        dyp = int(DeltaY/vy)
        
        #find ROI which works in both channels
        x01 = max(x - roiHalfSize, max(0, dxp))
        x11 = min(max(x01, x + roiHalfSize + 1), self.data.shape[0] + min(0, dxp))
        x02 = x01 - dxp
        x12 = x11 - dxp
        
        y01 = max(y - roiHalfSize, max(0, dyp))
        y11 = min(max(y + roiHalfSize + 1,  y01), self.data.shape[1] + min(0, dyp))
        y02 = y01 - dyp
        y12 = y11 - dyp
        
        xslice = slice(int(x01), int(x11))
        xslice2 = slice(int(x02), int(x12))
        
        yslice = slice(int(y01), int(y11))
        yslice2 = slice(int(y02), int(y12))
        

         #cut region out of data stack
        dataROI = np.copy(self.data[xslice, yslice, 0:2])
        dataROI[:,:,1] = self.data[xslice2, yslice2, 1]
        
        if self.noiseSigma is None:
            sigma = self._calc_sigma(dataROI)
        else:
            sigma = self.noiseSigma[xslice, yslice, 0:2]
            sigma[:,:,1] = self.noiseSigma[xslice2, yslice2, 1]
            
        sigma = ndimage.maximum_filter(sigma, [3,3,0])


        if self.metadata.getOrDefault('Analysis.subtractBackground', True) :
            #print 'bgs'
            if not self.background is None and len(np.shape(self.background)) > 1:
                bgROI = self.background[xslice, yslice, 0:2]
                bgROI[:,:,1] = self.background[xslice2, yslice2, 1]
            else:
                bgROI = np.zeros_like(dataROI) + (self.background if self.background else 0)
        else:
            bgROI = np.zeros_like(dataROI)

 

        #generate grid to evaluate function on        
        Xg = vx*(np.mgrid[xslice] + self.roi_offset[0])
        Yg = vy*(np.mgrid[yslice] + self.roi_offset[1])

        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...

        #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata['chroma.dx'],self.metadata['chroma.dy'])
        

        Xr = Xg + DeltaX - vx*dxp
        Yr = Yg + DeltaY - vy*dyp
        
            
        return Xg, Yg, Xr, Yr, dataROI, bgROI, sigma, xslice, yslice, xslice2, yslice2

    def getMultiviewROIAtPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        """Helper fcn to extract ROI from frame at given x,y, point from a multi-channel image.
        
        WARNING: EXPERIMENTAL WORK IN PROGRESS!!! This will eventually replace getSplitROIAtPoint and generalise to higher
        dimensional splitting (e.g. 4-quadrant systems such as the 4Pi-SMS) but is not useful in it's current form.

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
    
        x = round(x)
        y = round(y)
    
        #pixel size in nm
        vx, vy, _ = self.metadata.voxelsize_nm
    
        #position in nm from camera origin
        roi_x0, roi_y0 = get_camera_roi_origin(self.metadata)
        x_ = (x + roi_x0) * vx
        y_ = (y + roi_y0) * vy
    
        #look up shifts
        if not self.metadata.getOrDefault('Analysis.FitShifts', False):
            DeltaX = self.shift_x.ev(x_, y_)
            DeltaY = self.shift_y.ev(x_, y_)
        else:
            DeltaX = 0
            DeltaY = 0
    
        #find shift in whole pixels
        dxp = int(DeltaX / vx)
        dyp = int(DeltaY / vy)
    
        #find ROI which works in both channels
        #if dxp < 0:
        x01 = max(x - roiHalfSize, max(0, dxp))
        x11 = min(max(x01, x + roiHalfSize + 1), self.data.shape[0] + min(0, dxp))
        x02 = x01 - dxp
        x12 = x11 - dxp
    
        y01 = max(y - roiHalfSize, max(0, dyp))
        y11 = min(max(y + roiHalfSize + 1, y01), self.data.shape[1] + min(0, dyp))
        y02 = y01 - dyp
        y12 = y11 - dyp
    
        xslice = slice(int(x01), int(x11))
        xslice2 = slice(int(x02), int(x12))
    
        yslice = slice(int(y01), int(y11))
        yslice2 = slice(int(y02), int(y12))
    
        #print xslice2, yslice2
    
    
        #cut region out of data stack
        dataROI = self.data[xslice, yslice, 0:2]
        #print dataROI.shape
        dataROI[:, :, 1] = self.data[xslice2, yslice2, 1]
    
        nSlices = 1
        #sigma = np.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*np.maximum(dataROI, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount
        #phConv = self.metadata.Camera.ElectronsPerCount/self.metadata.Camera.TrueEMGain
        #nPhot = dataROI*phConv
    
        if self.noiseSigma is None:
            sigma = self._calc_sigma(dataROI)
        else:
            sigma = self.noiseSigma[xslice, yslice, 0:2]
            sigma[:, :, 1] = self.noiseSigma[xslice2, yslice2, 1]
    
        sigma = ndimage.maximum_filter(sigma, [3, 3, 0])
    
        if self.metadata.getOrDefault('Analysis.subtractBackground', True):
            #print 'bgs'
            if not self.background is None and len(np.shape(self.background)) > 1:
                bgROI = self.background[xslice, yslice, 0:2]
                bgROI[:, :, 1] = self.background[xslice2, yslice2, 1]
            else:
                bgROI = np.zeros_like(dataROI) + self.background
        else:
            bgROI = np.zeros_like(dataROI)
    
        #generate grid to evaluate function on
        Xg = vx * (np.mgrid[xslice] + self.roi_offset[0])
        Yg = vy * (np.mgrid[yslice] + self.roi_offset[1])
    
        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...
    
        #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata['chroma.dx'],self.metadata['chroma.dy'])
    
    
        Xr = Xg + DeltaX - vx * dxp
        Yr = Yg + DeltaY - vy * dyp
    
        return Xg, Yg, Xr, Yr, dataROI, bgROI, sigma, xslice, yslice, xslice2, yslice2

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        """This should be overridden in derived classes to actually do the fitting.
        The function which gets implemented should return a numpy record array, of the
        dtype defined in the module level FitResultsDType variable (the calling function
        uses FitResultsDType to pre-allocate an array for the results)"""
        
        raise NotImplementedError('This function should be over-ridden in derived class')
        
        
FitFactory = FFBase

DESCRIPTION = ''  # What type of object does this fitter fit?
LONG_DESCRIPTION = ''  # A longer description
USE_FOR = ''  # Type of localization data (e.g. 2D single color)
