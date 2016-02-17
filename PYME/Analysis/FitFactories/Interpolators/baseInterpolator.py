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
from numpy import *
#import cPickle
from PYME.ParallelTasks.relativeFiles import getFullExistingFilename

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
        '''load the model from file - returns True if the model changed, False if
        an existing model was reused'''
        #global IntXVals, IntYVals, IntZVals, interpModel, interpModelName, dx, dy, dz

        modName = md.PSFFile

        if not modName == self.interpModelName:
            #special case for theoretical models
            if modName.startswith('ZMODEL:'):
                params = eval(modName[7:])
                mod, voxelsize = self.genTheoreticalModelZernike(md, **params)
            else: 
                #try and get psf from task queue
                #if not md.taskQueue == None:
                try:
                    mod, voxelsize = md.taskQueue.getQueueData(md.dataSourceID, 'PSF')
                except:
                    mf = open(getFullExistingFilename(modName), 'rb')
                    mod, voxelsize = load(mf)
                    mf.close()

            self.setModel(modName, mod, voxelsize)

            #print 'model changed'
            return True #model changed
        else:
            return False #model not changed

    def setModelFromFile(self, modName, md=None):
        '''load the model from file - returns True if the model changed, False if
        an existing model was reused'''
        #global IntXVals, IntYVals, IntZVals, interpModel, interpModelName, dx, dy, dz

        if not modName == self.interpModelName:
            #special case for theoretical models
            if modName.startswith('ZMODEL:'):
                #print modName[7:]
                params = eval(modName[7:])
                mod, voxelsize = self.genTheoreticalModelZernike(md, **params)
            else: 
                mf = open(getFullExistingFilename(modName), 'rb')
                mod, voxelsize = load(mf)
                mf.close()

            self.setModel(modName, mod, voxelsize)

            #print 'model changed'
            return True #model changed
        else:
            return False #model not changed

    def setModel(self, modName, mod, voxelsize):
        self.interpModelName = modName

        #if not voxelsize.x == md.voxelsize.x:
        #    raise RuntimeError("PSF and Image voxel sizes don't match")
        
        if mod.shape[0] == 2*mod.shape[1]: 
            #using a split model - we have 2 PSFs side by side
            self.SplitPSF = True
            self.PSF2Offset = 1e3*voxelsize.x*mod.shape[1]
            self.IntXVals = 1e3*voxelsize.x*mgrid[-(mod.shape[1]/2.):(mod.shape[0]-mod.shape[1]/2.)]
            self.IntYVals = 1e3*voxelsize.y*mgrid[-(mod.shape[1]/2.):(mod.shape[1]/2.)]
            self.IntZVals = 1e3*voxelsize.z*mgrid[-(mod.shape[2]/2.):(mod.shape[2]/2.)]
        else:
            self.SplitPSF = False
            self.PSF2Offset = 0
            self.IntXVals = 1e3*voxelsize.x*mgrid[-(mod.shape[0]/2.):(mod.shape[0]/2.)]
            self.IntYVals = 1e3*voxelsize.y*mgrid[-(mod.shape[1]/2.):(mod.shape[1]/2.)]
            self.IntZVals = 1e3*voxelsize.z*mgrid[-(mod.shape[2]/2.):(mod.shape[2]/2.)]

        self.dx = voxelsize.x*1e3
        self.dy = voxelsize.y*1e3
        self.dz = voxelsize.z*1e3

        self.interpModel = (mod/mod[:,:, mod.shape[2]/2].sum()).astype('f') #normalise to 1
        self.shape = mod.shape

        self._precompute()
        
    def genTheoreticalModelZernike(self, md, zmodes={}, nDesign=1.51, nSample=1.51, NA=1.47, wavelength=700):
        from PYME.PSFGen import fourierHNA
        zs = arange(-1e3, 1e3, 50)
        
        voxelsize = dummy()
        voxelsize.x = md['voxelsize.x']
        voxelsize.y = md['voxelsize.x']
        voxelsize.z = .05

        ps = fourierHNA.GenZernikeDPSF(zs, 1e3*voxelsize.x, zmodes,lamb=wavelength, NA = NA, n=nDesign, ns=nSample)
        
        return ps, voxelsize

    def genTheoreticalModel(self, md):
        from PYME.PSFGen.ps_app import *

        if not self.dx == md.voxelsize.x*1e3 and not self.dy == md.voxelsize.y*1e3 and not self.dz == md.voxelsize.z*1e3:

            self.IntXVals = 1e3*md.voxelsize.x*mgrid[-20:20]
            self.IntYVals = 1e3*md.voxelsize.y*mgrid[-20:20]
            self.IntZVals = 1e3*md.voxelsize.z*mgrid[-20:20]

            self.dx = md.voxelsize.x*1e3
            self.dy = md.voxelsize.y*1e3
            self.dz = md.voxelsize.z*1e3

            P = arange(0,1.01,.01)

            interpModel = genWidefieldPSF(self.IntXVals, self.IntYVals, self.IntZVals, P,1e3, 0, 0, 0, 2*pi/525, 1.47, 10e3)

            self.interpModel = interpModel/interpModel.max() #normalise to 1

            self.shape = interpModel.shape

            self._precompute()

            return True
        else:
            return False

    def _precompute(self):
        '''placeholder function which is called after model loading and can be
        overridden to allow for interpolation specific precomputations'''
        pass

    def interp(self, X, Y, Z):
        '''placeholder to be overrriden to do actual interpolation at values given'''
        raise RuntimeError('Not Implemented')

    def getCoords(self, metadata, xslice, yslice, zslice):
        '''placeholder to be overrriden to return coordinates needed for interpolation'''
        raise RuntimeError('Not Implemented')

