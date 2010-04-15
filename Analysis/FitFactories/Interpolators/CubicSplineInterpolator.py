from baseInterpolator import __interpolator
from numpy import *
from scipy import ndimage

class CubicSplineInterpolator(__interpolator):
    def _precompute(self):
        '''function which is called after model loading and can be
        overridden to allow for interpolation specific precomputations'''
         #do the spline filtering here rather than in interpolation
        self.interpModel = ndimage.spline_filter(self.interpModel)


    def interp(self, X, Y, Z):
        '''do actual interpolation at values given'''

        x1 = X/self.dx + len(self.IntXVals)/2.
        y1 = Y/self.dy + len(self.IntYVals)/2.
        z1 = Z/self.dz + len(self.IntZVals)/2.

        return ndimage.interpolation.map_coordinates(self.interpModel, [x1, y1, z1], mode='nearest', prefilter=False).squeeze()

    def getCoords(self, metadata, xslice, yslice, zslice):
        '''placeholder to be overrriden to return coordinates needed for interpolation'''
        #generate grid to evaluate function on
        X,Y,Z = scipy.mgrid[xslice, yslice, :1]

        X = 1e3*self.metadata.voxelsize.x*X
        Y = 1e3*self.metadata.voxelsize.y*Y
        Z = 1e3*self.metadata.voxelsize.z*Z

        return X, Y, Z

interpolator = CubicSplineInterpolator()
        