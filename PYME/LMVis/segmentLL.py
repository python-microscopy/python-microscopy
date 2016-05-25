#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################


import numpy as np
from scipy.spatial import ckdtree

def Gauss2D(Xv,Yv, A,x0,y0,s):
    from PYME.localization.cModels.gauss_app import genGauss
    r = genGauss(Xv,Yv,A,x0,y0,s,0,0,0)
    #r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....
    return r

class segmenter(object):
    def __init__(self, startImage, x,y, sx, imageBounds, pixelSize):
        self.im = startImage
        self.x = x
        self.y = y
        self.sx = sx
        self.imageBounds = imageBounds
        self.pixelSize = pixelSize
        
        self.fuzz = 3*np.median(sx)
        self.roiSize = int(self.fuzz/pixelSize)
        self.fuzz = pixelSize*(self.roiSize)
        
        self.mask = np.ones(self.im.shape)
    
        #print imageBounds.x0
        #print imageBounds.x1
        #print fuzz
        #print roiSize
    
        #print pixelSize
    
        self.X = np.arange(imageBounds.x0,imageBounds.x1, pixelSize)
        self.Y = np.arange(imageBounds.y0,imageBounds.y1, pixelSize)
    
        #print X
        
        self.ctval = 1e-4
        

    
        #record our image resolution so we can plot pts with a minimum size equal to res (to avoid missing small pts)
        self.delX = np.absolute(self.X[1] - self.X[0]) 
        
        self.kdt = ckdtree.cKDTree(np.array([self.x,self.y]).T)
    
    
    def globalLHood(self):
        lh = 0
        for xi, yi, sxi in zip(self.x, self.y, self.sx):
          lh += self.pointlHood(self.im, xi, yi, sxi)
          
        return lh
    
    
    def pointlHood(self, im, xi,yi, sxi):       

        #im = self.im
        
        ix = np.absolute(self.X - xi).argmin()
        iy = np.absolute(self.Y - yi).argmin()
        
        imp = Gauss2D(self.X[max(ix - self.roiSize, 0):(min(ix + self.roiSize, im.shape[0])+1)], self.Y[max(iy - self.roiSize, 0):(min(iy + self.roiSize, im.shape[1])+1)],1, xi,yi,max(sxi, self.delX))
        #print 'r'
        imp[np.isnan(imp)] = self.ctval
        imp = np.maximum(imp, self.ctval)
        
    
        
        l = (imp*im[max(ix - self.roiSize, 0):(min(ix + self.roiSize, im.shape[0])+1), max(iy - self.roiSize, 0):(min(iy + self.roiSize, im.shape[1])+1)]).sum()
        
        lh = np.log(max(l, 1e-2))
            
        return lh/self.im.sum()
        
    def probePoint(self, x_i, y_i):
        #test to see whether toggling a pixel improves or worsens log likelihood
        
        #if there are no points near, then don't bother doing anythin more
        if not self.mask[x_i, y_i]:
            return False
            
        xi = x_i*self.pixelSize +  self.imageBounds.x0
        yi = y_i*self.pixelSize +  self.imageBounds.y0
        
        dn, neigh = self.kdt.query(np.array([xi,yi]), 50)

        neigh = neigh[dn < 100]

        if len(neigh) > 1:
            #calculate old likelihood
            lh = 0
            for j in neigh:
                lh += self.pointlHood(self.im, self.x[j], self.y[j], self.sx[j])
            
            #toggle pixel
            self.im[x_i, y_i] = 1 - self.im[x_i, y_i]
            lhn = 0
            for j in neigh:
                lhn += self.pointlHood(self.im, self.x[j], self.y[j], self.sx[j])
                
            if lhn > lh:
                #toggling improves image
                return True
            else:
                #switch back
                self.im[x_i, y_i] = 1 - self.im[x_i, y_i]
                return False
                
        else:
            #this pixel has no events near it and should never be on
            self.mask[x_i, y_i] = 0
            
            if self.im[x_i, y_i]:
                self.im[x_i, y_i] = 0
                return True
            else:
                return False
                
            
                
            
