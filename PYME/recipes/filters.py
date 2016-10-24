# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:11:05 2015

@author: david
"""
from .base import register_module, ModuleBase, Filter, Float, Enum, CStr, Bool, Int, View, Item#, Group
from scipy import ndimage
#from PYME.IO.image import ImageStack
import numpy as np

@register_module('GaussianFilter')    
class GaussianFilter(Filter):
    sigmaY = Float(1.0)
    sigmaX = Float(1.0)
    sigmaZ = Float(1.0)
    #def __init__(self, **kwargs):
    #    pass
    @property
    def sigmas(self):
        return [self.sigmaX, self.sigmaY, self.sigmaZ]
    
    def applyFilter(self, data, chanNum, frNum, im):
        return ndimage.gaussian_filter(data, self.sigmas[:len(data.shape)])
    
    def completeMetadata(self, im):
        im.mdh['Processing.GaussianFilter'] = self.sigmas

@register_module('MedianFilter')         
class MedianFilter(Filter):
    sizeX = Float(1.0)
    sizeY = Float(1.0)
    sizeZ = Float(1.0)
    #def __init__(self, **kwargs):
    #    pass
    @property
    def sigmas(self):
        return [self.sizeX, self.sizeY, self.sizeZ]
    
    def applyFilter(self, data, chanNum, frNum, im):
        return ndimage.median_filter(data, self.sigmas[:len(data.shape)])
    
    def completeMetadata(self, im):
        im.mdh['Processing.MedianFilter'] = self.sigmas
        
@register_module('DespeckleFilter')         
class DespeckleFilter(Filter):
    sizeX = Int(3)
    sizeY = Int(3)
    sizeZ = Int(3)
    nPix = Int(3)
    
    def _filt(self, data):
        v = data[data.size/2]
        
        dv = np.abs(data - v)
        
        I = np.argsort(dv)
        return np.median(data[I[:self.nPix]])
        
    #def __init__(self, **kwargs):
    #    pass
    @property
    def sigmas(self):
        return [self.sizeX, self.sizeY, self.sizeZ]
    
    def applyFilter(self, data, chanNum, frNum, im):
        return ndimage.generic_filter(data, self._filt, self.sigmas[:len(data.shape)])
    
    def completeMetadata(self, im):
        im.mdh['Processing.DespeckleFilter'] = self.sigmas

@register_module('MeanFilter') 
class MeanFilter(Filter):
    sizeX = Float(1.0)
    sizeY = Float(1.0)
    sizeZ = Float(1.0)
    #def __init__(self, **kwargs):
    #    pass
    @property
    def sigmas(self):
        return [self.sizeX, self.sizeY, self.sizeZ]
    
    def applyFilter(self, data, chanNum, frNum, im):
        return ndimage.mean_filter(data, self.sigmas[:len(data.shape)])
    
    def completeMetadata(self, im):
        im.mdh['Processing.MeanFilter'] = self.sigmas

@register_module('Zoom')         
class Zoom(Filter):
    zoom = Float(1.0)
    
    def applyFilter(self, data, chanNum, frNum, im):
        return ndimage.zoom(data, self.zoom)
    
    def completeMetadata(self, im):
        im.mdh['Processing.Zoom'] = self.zoom
        im.mdh['voxelsize.x'] = im.mdh['voxelsize.x']/self.zoom
        im.mdh['voxelsize.y'] = im.mdh['voxelsize.y']/self.zoom
        
        if not self.processFramesIndividually:
            im.mdh['voxelsize.z'] = im.mdh['voxelsize.z']/self.zoom

@register_module('MaskEdges')
class MaskEdges(Filter):
    widthPixels = Int(10)

    def applyFilter(self, data, chanNum, frNum, im):
        dm = data.copy()
        dm[:self.widthPixels, :] = 0
        dm[-self.widthPixels:, :] = 0
        dm[:, :self.widthPixels] = 0
        dm[:, -self.widthPixels:] = 0
        return dm
            
@register_module('DoGFilter')         
class DoGFilter(Filter):
    """Difference of Gaussians"""
    sigmaY = Float(1.0)
    sigmaX = Float(1.0)
    sigmaZ = Float(1.0)
    
    sigma2Y = Float(1.0)
    sigma2X = Float(1.0)
    sigma2Z = Float(1.0)

    #def __init__(self, **kwargs):
    #    pass
    @property
    def sigmas(self):
        return [self.sigmaX, self.sigmaY, self.sigmaZ]
        
    @property
    def sigma2s(self):
        return [self.sigma2X, self.sigma2Y, self.sigma2Z]
    
    def applyFilter(self, data, chanNum, frNum, im):
        return ndimage.gaussian_filter(data, self.sigmas[:len(data.shape)]) - ndimage.gaussian_filter(data, self.sigma2s[:len(data.shape)])
    
    def completeMetadata(self, im):
        im.mdh['Processing.GaussianFilter'] = self.sigmas



 
#d = {}
#d.update(locals())
#moduleList = [c for c in d if _issubclass(c, ModuleBase) and not c == ModuleBase]       
