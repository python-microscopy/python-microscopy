from .BaseDataSource import XYZTCDataSource, XYZTCWrapper
import numpy as np
from scipy import ndimage

class XYZTCDriftCorrectSource(XYZTCDataSource):
    moduleName = 'DriftCorrectDataSource'
    '''
    def __init__(self, datasource, x_mapping, y_mapping, x_scale=1.0, y_scale=1.0):
        if (not isinstance(datasource, XYZTCDataSource)) and (not datasource.ndim == 5) :
            datasource = XYZTCWrapper.auto_promote(datasource)
        
        self._datasource = datasource
        size_z, size_t, size_c = datasource.shape[2:]

        self._x_map = x_mapping # a piecewise mapping object
        self._x_scale = x_scale #allows conversion between different camera pixel units. TODO - make it an affine transformation matrix to allow for rotation as well.
        self._y_map = y_mapping
        self._y_scale = y_scale
        
        XYZTCDataSource.__init__(self, input_order=datasource._input_order, size_z=size_z, size_t=size_t, size_c=size_c)
    '''
    def __init__(self, datasource, x_mapping, y_mapping, px0, py0, px1, py1, relative_rotation_angle):
        if (not isinstance(datasource, XYZTCDataSource)) and (not datasource.ndim == 5) :
            datasource = XYZTCWrapper.auto_promote(datasource)
        
        self._datasource = datasource
        size_z, size_t, size_c = datasource.shape[2:]

        self._x_map = x_mapping # a piecewise mapping object
        self._y_map = y_mapping
        self._xx_scale = px0/px1
        self._xy_scale = px0/py1
        self._yx_scale = py0/px1
        self._yy_scale = py0/py1
        self._theta = relative_rotation_angle     # prepare for an affine transformation to allow for scaling and rotation
        
        XYZTCDataSource.__init__(self, input_order=datasource._input_order, size_z=size_z, size_t=size_t, size_c=size_c)

    def getSlice(self, ind):
        sl = self._datasource.getSlice(ind)
        x0 = self._x_map(np.array([ind+1]))
        y0 = self._y_map(np.array([ind+1]))
        x1 = x0*self._xx_scale*np.cos(self._theta) + y0*self._yx_scale*np.sin(self._theta)
        y1 = y0*self._yy_scale*np.cos(self._theta) - x0*self._xy_scale*np.sin(self._theta)
        #return ndimage.shift(sl, [-self._x_scale*self._x_map(np.array([ind+1])), -self._y_scale*self._y_map(np.array([ind+1]))], order=3, mode='nearest') 
        return ndimage.shift(sl, [-x1, -y1], order=3, mode='nearest') 
    
   # proxy original data source attributes 
    def __getattr__(self, item):
        return getattr(self._datasource, item)
        
    def getSliceShape(self):
        return self._datasource.getSliceShape()

    def getNumSlices(self):
        return self._datasource.getNumSlices()

    def getEvents(self):
        return self._datasource.getEvents()

    @property
    def is_complete(self):
        return self._datasource.is_complete()
    
