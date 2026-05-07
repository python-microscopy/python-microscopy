#!/usr/bin/python

###############
# baseInterpolator.py
#
# Copyright David Baddeley, 2012
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
################
#from numpy import *
import numpy as np
from PYME.IO.image import ImageStack
#import cPickle
from PYME.IO.FileUtils.nameUtils import getFullExistingFilename
from PYME.IO.load_psf import load_psf

class dummy(object):
    pass

class __interpolator:
    def __init__(self):
        self.IntXVals = None
        self.IntYVals = None
        self.IntZVals = None

        self.interpModel = None
        self.interpModelName = None

        self.dx = None
        self.dy = None
        self.dz = None
        
        self.SplitPSF = False
        self.PSF2Offset = 0

    def setModelFromMetadata(self, md):
        """load the model from file - returns True if the model changed, False if
        an existing model was reused"""
        #global IntXVals, IntYVals, IntZVals, interpModel, interpModelName, dx, dy, dz

        modName = md['PSFFile']

        if not modName == self.interpModelName:
            #special case for theoretical models
            if modName.startswith('ZMODEL:'):
                params = eval(modName[7:])
                mod, voxelsize = self.genTheoreticalModelZernike(md, **params)
            else: 
                #try and get psf from task queue
                #if not md.taskQueue == None:
                try:
                    mod, voxelsize = md['taskQueue'].getQueueData(md['dataSourceID'], 'PSF')
                except:
                    #mf = open(getFullExistingFilename(modName), 'rb')
                    #mod, voxelsize = load(mf)
                    #mf.close()
                    #mf = ImageStack(filename=modName)
                    #mod = mf.data[:,:,:].astype('f')
                    #voxelsize = mf.voxelsize
                    mod, voxelsize = load_psf(modName)


            self.setModel(modName, mod, voxelsize)

            #print 'model changed'
            return True #model changed
        else:
            return False #model not changed

    def setModelFromFile(self, modName, md=None):
        """load the model from file - returns True if the model changed, False if
        an existing model was reused"""
        #global IntXVals, IntYVals, IntZVals, interpModel, interpModelName, dx, dy, dz

        if not modName == self.interpModelName:
            #special case for theoretical models
            if modName.startswith('ZMODEL:'):
                #print modName[7:]
                params = eval(modName[7:])
                mod, voxelsize = self.genTheoreticalModelZernike(md, **params)
            else: 
                #mf = open(getFullExistingFilename(modName), 'rb')
                #mod, voxelsize = load(mf)
                #mf.close()
                # mf = ImageStack(filename=modName)
                # mod = mf.data[:,:,:].astype('f')
                # voxelsize = mf.voxelsize
                mod, voxelsize_nm = load_psf(modName)

            self.setModel(modName, mod, voxelsize_nm)

            #print 'model changed'
            return True #model changed
        else:
            return False #model not changed

    def setModel(self, modName, mod, voxelsize):
        self.interpModelName = modName
        
        #print mod.min(), mod.max()

        #if not voxelsize.x == md.voxelsize.x:
        #    raise RuntimeError("PSF and Image voxel sizes don't match")
        
        if mod.shape[0] == 2*mod.shape[1]: 
            #using a split model - we have 2 PSFs side by side
            self.SplitPSF = True
            self.PSF2Offset = voxelsize.x*mod.shape[1]
            self.IntXVals = voxelsize.x*np.mgrid[-(mod.shape[1]/2.):(mod.shape[0]-mod.shape[1]/2.)]
            self.IntYVals = voxelsize.y*np.mgrid[-(mod.shape[1]/2.):(mod.shape[1]/2.)]
            self.IntZVals = voxelsize.z*np.mgrid[-(mod.shape[2]/2.):(mod.shape[2]/2.)]
        else:
            self.SplitPSF = False
            self.PSF2Offset = 0
            self.IntXVals = voxelsize.x*np.mgrid[-(mod.shape[0]/2.):(mod.shape[0]/2.)]
            self.IntYVals = voxelsize.y*np.mgrid[-(mod.shape[1]/2.):(mod.shape[1]/2.)]
            self.IntZVals = voxelsize.z*np.mgrid[-(mod.shape[2]/2.):(mod.shape[2]/2.)]

        self.dx = voxelsize.x
        self.dy = voxelsize.y
        self.dz = voxelsize.z

        self.interpModel = (mod/mod[:,:, int(mod.shape[2]/2)].sum()).astype('f') #normalise to 1
        self.shape = mod.shape

        self._precompute()
        
    def centre2d(self,psf):
        sz = psf.shape
        psfc = psf[:,:,sz[2]/2].squeeze()
        X, Y = np.meshgrid(np.arange(sz[0]),np.arange(sz[1]))
        xc = (X*psfc).sum()/psfc.sum()
        yc = (Y*psfc).sum()/psfc.sum()
        psfs = np.roll(np.roll(psf,int(np.round(sz[0]/2-xc)),0),int(np.round(sz[1]/2-yc)),1)
        return psfs
    
    def genTheoreticalModelZernike(self, md, zmodes={}, nDesign=1.51, nSample=1.51, NA=1.47, wavelength=700):
        from PYME.Analysis.PSFGen import fourierHNA
        from PYME.IO import MetaDataHandler
        zs = np.arange(-1e3, 1e3, 50)
        
        voxelsize = MetaDataHandler.VoxelSize(md.voxelsize_nm)
        voxelsize.z = 50.

        ps = fourierHNA.GenZernikeDPSF(zs, voxelsize.x, zmodes,lamb=wavelength, NA = NA, n=nDesign, ns=nSample)
        psc = self.centre2d(ps) #FIXME: why is this needed / useful?
        
        return psc, voxelsize

    def genTheoreticalModel(self, md):
        from PYME.Analysis.PSFGen.ps_app import genWidefieldPSF
        
        vs = md.voxelsize_nm

        if not self.dx == vs.x and not self.dy == vs.y and not self.dz == vs.z:

            self.IntXVals = vs.x*np.mgrid[-20:20]
            self.IntYVals = vs.y*np.mgrid[-20:20]
            self.IntZVals = vs.z*np.mgrid[-20:20]

            self.dx, self.dy, self.dx = vs

            P = np.arange(0,1.01,.01)

            interpModel = genWidefieldPSF(self.IntXVals, self.IntYVals, self.IntZVals, P,1e3, 0, 0, 0, 2*np.pi/525, 1.47, 10e3)

            self.interpModel = interpModel/interpModel.max() #normalise to 1

            self.shape = interpModel.shape

            self._precompute()

            return True
        else:
            return False

    def _precompute(self):
        """placeholder function which is called after model loading and can be
        overridden to allow for interpolation specific precomputations"""
        pass

    def interp(self, X, Y, Z):
        """placeholder to be overrriden to do actual interpolation at values given"""
        raise RuntimeError('Not Implemented')

    def getCoords(self, metadata, xslice, yslice, zslice):
        """placeholder to be overrriden to return coordinates needed for interpolation"""
        raise RuntimeError('Not Implemented')

