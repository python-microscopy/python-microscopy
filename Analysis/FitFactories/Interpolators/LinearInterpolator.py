from baseInterpolator import __interpolator
from numpy import *

class LinearInterpolator(__interpolator):
    def interp(self, X, Y, Z):
        '''do actual interpolation at values given'''

        X = atleast_1d(X)
        Y = atleast_1d(Y)
        Z = atleast_1d(Z)

        ox = X[0]
        oy = Y[0]
        oz = Z[0]

        rx = (ox % self.dx)/self.dx
        ry = (oy % self.dy)/self.dy
        rz = (oz % self.dz)/self.dz

        fx = int(len(self.IntXVals)/2) + int(ox/self.dx)
        fy = int(len(self.IntYVals)/2) + int(oy/self.dy)
        fz = int(len(self.IntZVals)/2) + int(oz/self.dz)

        #print fx
        #print rx, ry, rz

        xl = len(X)
        yl = len(Y)
        zl = len(Z)

        #print xl

        m000 = self.interpModel[fx:(fx+xl),fy:(fy+yl),fz:(fz+zl)]
        m100 = self.interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),fz:(fz+zl)]
        m010 = self.interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),fz:(fz+zl)]
        m110 = self.interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),fz:(fz+zl)]

        m001 = self.interpModel[fx:(fx+xl),fy:(fy+yl),(fz+1):(fz+zl+1)]
        m101 = self.interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),(fz+1):(fz+zl+1)]
        m011 = self.interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),(fz+1):(fz+zl+1)]
        m111 = self.interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),(fz+1):(fz+zl+1)]

        #print m000.shape

    #    m = scipy.sum([((1-rx)*(1-ry)*(1-rz))*m000, ((rx)*(1-ry)*(1-rz))*m100, ((1-rx)*(ry)*(1-rz))*m010, ((rx)*(ry)*(1-rz))*m110,
    #        ((1-rx)*(1-ry)*(rz))*m001, ((rx)*(1-ry)*(rz))*m101, ((1-rx)*(ry)*(rz))*m011, ((rx)*(ry)*(rz))*m111], 0)

        m = ((1-rx)*(1-ry)*(1-rz))*m000 + ((rx)*(1-ry)*(1-rz))*m100 + ((1-rx)*(ry)*(1-rz))*m010 + ((rx)*(ry)*(1-rz))*m110+((1-rx)*(1-ry)*(rz))*m001+ ((rx)*(1-ry)*(rz))*m101+ ((1-rx)*(ry)*(rz))*m011+ ((rx)*(ry)*(rz))*m111
        #print m.shape
        return m

    def getCoords(self, metadata, xslice, yslice, zslice):
        '''placeholder to be overrriden to return coordinates needed for interpolation'''
        #generate grid to evaluate function on
        X = 1e3*metadata.voxelsize.x*mgrid[xslice]
        Y = 1e3*metadata.voxelsize.y*mgrid[yslice]
        Z = array([0]).astype('f')

        return X, Y, Z

interpolator = LinearInterpolator()
        