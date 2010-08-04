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

        #print x1

        return ndimage.interpolation.map_coordinates(self.interpModel, array([x1, y1, z1]), mode='nearest', prefilter=False).squeeze()

    def getCoords(self, metadata, xslice, yslice, zslice):
        '''placeholder to be overrriden to return coordinates needed for interpolation'''
        #generate grid to evaluate function on
        X,Y,Z = mgrid[xslice, yslice, :1]

        X = 1e3*metadata.voxelsize.x*X
        Y = 1e3*metadata.voxelsize.y*Y
        Z = 1e3*metadata.voxelsize.z*Z

        #what region is 'safe' (ie we won't run out of model to interpret)
        #for these slices...
        xm = X.shape[0]/2
        dx = min((interpolator.shape[0] - X.shape[0])/2, xm) - 2

        ym = X.shape[1]/2
        dy = min((interpolator.shape[1] - X.shape[1])/2, ym) - 2

        safeRegion = ((X[xm-dx, 0,0], X[xm+dx, 0, 0]), (Y[0, ym-dy, 0], Y[0, ym+dy, 0]),(Z[0,0,0] + self.IntZVals[2], Z[0,0,0] + self.IntZVals[-2]))

        return X, Y, Z, safeRegion

interpolator = CubicSplineInterpolator()
        