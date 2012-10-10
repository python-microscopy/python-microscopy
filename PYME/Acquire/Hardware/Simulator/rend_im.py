#!/usr/bin/python

##################
# rend_im.py
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

from PYME.PSFGen import *
from scipy import *
import fluor
from PYME.Analysis import MetaData
from PYME.Analysis import cInterp
import cPickle

from PYME.ParallelTasks.relativeFiles import getFullExistingFilename

#import threading
#tLock = threading.Lock()

def renderIm(X, Y, z, points, roiSize, A):
    #X = mgrid[xImSlice]
    #Y = mgrid[yImSlice]
    im = zeros((len(X), len(Y)), 'f')

    P = arange(0,1.01,.1)

    for (x0,y0,z0) in points:
        ix = abs(X - x0).argmin()
        iy = abs(Y - y0).argmin()

        imp =genWidefieldPSF(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)], z, P,A*1e3, x0,y0,z0,depthInSample=0)
        #print imp.shape
        im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp[:,:,0]
    
    #print im.shape
    return im



IntXVals = None
IntYVals = None
IntZVals = None

interpModel = None

dx = None
dy = None
dz = None

def genTheoreticalModel(md):
    global IntXVals, IntYVals, IntZVals, interpModel, dx, dy, dz

    if not dx == md.voxelsize.x*1e3 or not dy == md.voxelsize.y*1e3 or not dz == md.voxelsize.z*1e3:

        IntXVals = 1e3*md.voxelsize.x*mgrid[-20:20]
        IntYVals = 1e3*md.voxelsize.y*mgrid[-20:20]
        IntZVals = 1e3*md.voxelsize.z*mgrid[-20:20]

        dx = md.voxelsize.x*1e3
        dy = md.voxelsize.y*1e3
        dz = md.voxelsize.z*1e3

        P = arange(0,1.01,.01)

        interpModel = genWidefieldPSF(IntXVals, IntYVals, IntZVals, P,1e3, 0, 0, 0, 2*pi/525, 1.47, 10e3).astype('f')
        
        print 'foo'
        print interpModel.strides, interpModel.shape

        interpModel = interpModel/interpModel.max() #normalise to 1
        
        print 'bar'

genTheoreticalModel(MetaData.TIRFDefault)

#def setModel(mod, md):
#    global IntXVals, IntYVals, IntZVals, interpModel, dx, dy, dz
#
#    IntXVals = 1e3*md.voxelsize.x*mgrid[-(mod.shape[0]/2.):(mod.shape[0]/2.)]
#    IntYVals = 1e3*md.voxelsize.y*mgrid[-(mod.shape[1]/2.):(mod.shape[1]/2.)]
#    IntZVals = 1e3*md.voxelsize.z*mgrid[-(mod.shape[2]/2.):(mod.shape[2]/2.)]
#
#    dx = md.voxelsize.x*1e3
#    dy = md.voxelsize.y*1e3
#    dz = md.voxelsize.z*1e3
#
#    interpModel = mod
#
#    interpModel = interpModel/interpModel.max() #normalise to 1

def setModel(modName, md):
    global IntXVals, IntYVals, IntZVals, interpModel, dx, dy, dz

    
    mf = open(getFullExistingFilename(modName), 'rb')
    mod, voxelsize = cPickle.load(mf)
    mf.close()

    #if not voxelsize.x == md.voxelsize.x:
    #    raise RuntimeError("PSF and Image voxel sizes don't match")

    IntXVals = 1e3*voxelsize.x*mgrid[-(mod.shape[0]/2.):(mod.shape[0]/2.)]
    IntYVals = 1e3*voxelsize.y*mgrid[-(mod.shape[1]/2.):(mod.shape[1]/2.)]
    IntZVals = 1e3*voxelsize.z*mgrid[-(mod.shape[2]/2.):(mod.shape[2]/2.)]

    dx = voxelsize.x*1e3
    dy = voxelsize.y*1e3
    dz = voxelsize.z*1e3

    interpModel = mod

    interpModel = interpModel/interpModel.max() #normalise to 1

def interp(X, Y, Z):
    X = atleast_1d(X)
    Y = atleast_1d(Y)
    Z = atleast_1d(Z)

    ox = X[0]
    oy = Y[0]
    oz = Z[0]

    rx = (ox % dx)/dx
    ry = (oy % dy)/dy
    rz = (oz % dz)/dz

    fx = int(len(IntXVals)/2) + int(ox/dx)
    fy = int(len(IntYVals)/2) + int(oy/dy)
    fz = int(len(IntZVals)/2) + int(oz/dz)

    #print fx
    #print rx, ry, rz

    xl = len(X)
    yl = len(Y)
    zl = len(Z)

    #print xl

    m000 = interpModel[fx:(fx+xl),fy:(fy+yl),fz:(fz+zl)]
    m100 = interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),fz:(fz+zl)]
    m010 = interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),fz:(fz+zl)]
    m110 = interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),fz:(fz+zl)]

    m001 = interpModel[fx:(fx+xl),fy:(fy+yl),(fz+1):(fz+zl+1)]
    m101 = interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),(fz+1):(fz+zl+1)]
    m011 = interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),(fz+1):(fz+zl+1)]
    m111 = interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),(fz+1):(fz+zl+1)]

    #print m000.shape

#    m = scipy.sum([((1-rx)*(1-ry)*(1-rz))*m000, ((rx)*(1-ry)*(1-rz))*m100, ((1-rx)*(ry)*(1-rz))*m010, ((rx)*(ry)*(1-rz))*m110,
#        ((1-rx)*(1-ry)*(rz))*m001, ((rx)*(1-ry)*(rz))*m101, ((1-rx)*(ry)*(rz))*m011, ((rx)*(ry)*(rz))*m111], 0)

    m = ((1-rx)*(1-ry)*(1-rz))*m000 + ((rx)*(1-ry)*(1-rz))*m100 + ((1-rx)*(ry)*(1-rz))*m010 + ((rx)*(ry)*(1-rz))*m110+((1-rx)*(1-ry)*(rz))*m001+ ((rx)*(1-ry)*(rz))*m101+ ((1-rx)*(ry)*(rz))*m011+ ((rx)*(ry)*(rz))*m111
    #print m.shape
    return m

def interp2(X, Y, Z):
    X = atleast_1d(X)
    Y = atleast_1d(Y)
    Z = atleast_1d(Z)

    ox = X[0]
    oy = Y[0]
    oz = Z[0]

    rx = (ox % dx)/dx
    ry = (oy % dy)/dy
    rz = (oz % dz)/dz

    fx = int(len(IntXVals)/2) + int(ox/dx)
    fy = int(len(IntYVals)/2) + int(oy/dy)
    fz = int(len(IntZVals)/2) + int(oz/dz)

    #print fx
    #print rx, ry, rz

    xl = len(X)
    yl = len(Y)
    zl = len(Z)

    #print xl

    m000 = interpModel[fx:(fx+xl),fy:(fy+yl),fz:(fz+zl)]
    m100 = interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),fz:(fz+zl)]
    m010 = interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),fz:(fz+zl)]
    m110 = interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),fz:(fz+zl)]

    m001 = interpModel[fx:(fx+xl),fy:(fy+yl),(fz+1):(fz+zl+1)]
    m101 = interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),(fz+1):(fz+zl+1)]
    m011 = interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),(fz+1):(fz+zl+1)]
    m111 = interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),(fz+1):(fz+zl+1)]

    #print m000.shape

#    m = scipy.sum([((1-rx)*(1-ry)*(1-rz))*m000, ((rx)*(1-ry)*(1-rz))*m100, ((1-rx)*(ry)*(1-rz))*m010, ((rx)*(ry)*(1-rz))*m110,
#        ((1-rx)*(1-ry)*(rz))*m001, ((rx)*(1-ry)*(rz))*m101, ((1-rx)*(ry)*(rz))*m011, ((rx)*(ry)*(rz))*m111], 0)

    m = ((1-rx)*(1-ry)*(1-rz))*m000 + ((rx)*(1-ry)*(1-rz))*m100 + ((1-rx)*(ry)*(1-rz))*m010 + ((rx)*(ry)*(1-rz))*m110+((1-rx)*(1-ry)*(rz))*m001+ ((rx)*(1-ry)*(rz))*m101+ ((1-rx)*(ry)*(rz))*m011+ ((rx)*(ry)*(rz))*m111

    r000 = ((1-rx)*(1-ry)*(1-rz))
    r100 = ((rx)*(1-ry)*(1-rz))
    r010 = ((1-rx)*(ry)*(1-rz))
    r110 = ((rx)*(ry)*(1-rz))
    r001 = ((1-rx)*(1-ry)*(rz))
    r101 = ((1-rx)*(ry)*(rz))
    r011 = ((1-rx)*(ry)*(rz))
    r111 = ((rx)*(ry)*(rz))

    m = r000*m000
    m[:] = m[:] + r100*m100
    m[:] = m[:] + r010*m010
    m[:] = m[:] + r110*m110
    m[:] = m[:] + r001*m001
    m[:] = m[:] + r101*m101
    m[:] = m[:] + r011*m011
    m[:] = m[:] + r111*m111

    m = r000*m000 + r100*m100 + r010*m010 + r110*m110 + r001*m001 + r101*m101 + r011*m011 + r111*m111
    #print m.shape
    return m

def interp3(X, Y, Z):
    X = atleast_1d(X)
    Y = atleast_1d(Y)
    Z = atleast_1d(Z)

    ox = X[0]
    oy = Y[0]
    oz = Z[0]

    xl = len(X)
    yl = len(Y)
    zl = len(Z)
    
    return cInterp.Interpolate(interpModel, ox,oy,oz,xl,yl,dx,dy,dz)[:,:,None]

@fluor.registerIllumFcn
def PSFIllumFunction(fluors, position):
    xi = maximum(minimum(round_((fluors['x'] - position[0])/dx + interpModel.shape[0]/2).astype('i'), interpModel.shape[0]-1), 0)
    yi = maximum(minimum(round_((fluors['y'] - position[1])/dy + interpModel.shape[1]/2).astype('i'), interpModel.shape[1]-1), 0)
    zi = maximum(minimum(round_((fluors['z'] - position[2])/dz + interpModel.shape[2]/2).astype('i'), interpModel.shape[2]-1), 0)

    return interpModel[xi, yi, zi]


def simPalmIm(X,Y, z, fluors, intTime=.1, numSubSteps=10, roiSize=10, laserPowers = [.1,1]):
    im = zeros((len(X), len(Y)), 'f')

    if fluors == None:
        return im

    P = arange(0,1.01,.1)

    for f in fluors:
        A = array([f.illuminate(laserPowers,intTime/numSubSteps) for n  in range(numSubSteps)]).sum()
        if (A > 0):
            ix = abs(X - f.x).argmin()
            iy = abs(Y - f.y).argmin()

            imp =genWidefieldPSF(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)], z, P,A*1e3, f.x, f.y, f.z,depthInSample=0)
            im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp[:,:,0]

    return im


def simPalmImF(X,Y, z, fluors, intTime=.1, numSubSteps=10, roiSize=10, laserPowers = [.1,1]):
    im = zeros((len(X), len(Y)), 'f')
    
    if fluors == None:
        return im

    P = arange(0,1.01,.1)
    
    A = zeros(len(fluors.fl))

    #tLock.acquire()

    for n  in range(numSubSteps): 
        A += fluors.illuminate(laserPowers,intTime/numSubSteps)

    #tLock.release()

    flOn = where(A > 0)[0]
    
    #print flOn

    for i in flOn:
       ix = abs(X - fluors.fl['x'][i]).argmin()
       iy = abs(Y - fluors.fl['y'][i]).argmin()      
            
       imp =genWidefieldPSF(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)], z, P,A[i]*1e3, fluors.fl['x'][i], fluors.fl['y'][i], fluors.fl['z'][i],depthInSample=50e3)
       im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp[:,:,0] 

    return im


def simPalmImFI(X,Y, z, fluors, intTime=.1, numSubSteps=10, roiSize=15, laserPowers = [.1,1], position=[0,0,0], illuminationFunction='ConstIllum'):
    if interpModel == None:
        genTheoreticalModel(MetaData.TIRFDefault)
        
    im = zeros((len(X), len(Y)), 'f')

    if fluors == None:
        return im

    #P = arange(0,1.01,.1)

    A = zeros(len(fluors.fl))

    #tLock.acquire()

    for n  in range(numSubSteps):
        A += fluors.illuminate(laserPowers,intTime/numSubSteps, position=position, illuminationFunction=illuminationFunction)


    #print position
    #tLock.release()

    flOn = where(A > 0.1)[0]

    #print flOn
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    
    #print interpModel.shape, interpModel.strides


    for i in flOn:
       x = fluors.fl['x'][i] #+ position[0]
       y = fluors.fl['y'][i] #+ position[1]

       delX = abs(X - x)
       delY = abs(Y - y)

       ix = delX.argmin()
       iy = delY.argmin()

       s= min(roiSize, 20- roiSize)*dx
       if delX[ix] <  s and delY[iy] < s:
       #print ix, iy

           ix0 = max(ix - roiSize, 0)
           ix1 = min(ix + roiSize + 1, im.shape[0])
           iy0 = max(iy - roiSize, 0)
           iy1 = min(iy + roiSize + 1, im.shape[1])
           #imp =interp3(X[max(ix - roiSize, 0):(ix + roiSize + 1)] - x, Y[max(iy - roiSize, 0):(iy + roiSize + 1)] - y, z - fluors.fl['z'][i])* A[i]
           imp = cInterp.Interpolate(interpModel, X[ix0] - x, Y[iy0] - y, z - fluors.fl['z'][i], ix1-ix0, iy1-iy0,dx,dy,dz)* A[i]
           #print imp.shape
           #print imp.shape
           #if not imp.shape[2] == 0
           if imp.min() < 0 or isnan(A[i]):
               print ix0, ix1, iy0, iy1, (X[ix0] - x)/dx, (Y[iy0]-  y)/dx, A[i], imp.min()
           im[ix0:ix1, iy0:iy1] += imp[:,:,0]

    return im



def simPalmImFSpec(X,Y, z, fluors, intTime=.1, numSubSteps=10, roiSize=10, laserPowers = [.1,1], deltaY=64, deltaZ = 300):
    im = zeros((len(X), len(Y)), 'f')

    deltaY = (Y[1] - Y[0])*deltaY #convert to nm
    #print deltaY
    
    if fluors == None:
        return im

    P = arange(0,1.01,.1)
    
    A = zeros(len(fluors.fl))

    for n  in range(numSubSteps): 
        A += fluors.illuminate(laserPowers,intTime/numSubSteps)

    flOn = where(A > 0)[0]
    
    #print flOn

    for i in flOn:
       ix = abs(X - fluors.fl['x'][i]).argmin()
       iy = abs(Y - deltaY - fluors.fl['y'][i]).argmin()      
            
       imp =fluors.fl[i]['spec'][0]*genWidefieldPSF(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)], z, P,A[i]*1e3, fluors.fl['x'][i], fluors.fl['y'][i] + deltaY , fluors.fl['z'][i])
       im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp[:,:,0]


       ix = abs(X - fluors.fl['x'][i]).argmin()
       iy = abs(flipud(Y) - deltaY - fluors.fl['y'][i]).argmin()
       imp =fluors.fl[i]['spec'][1]*genWidefieldPSF(X[(ix - roiSize):(ix + roiSize + 1)], flipud(Y)[(iy - roiSize):(iy + roiSize + 1)], z, P,A[i]*1e3, fluors.fl['x'][i], fluors.fl['y'][i] + deltaY, fluors.fl['z'][i])
       im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp[:,:,0]

    return im

def simPalmImFSpecI(X,Y, z, fluors, intTime=.1, numSubSteps=10, roiSize=10, laserPowers = [.1,1], deltaY=64, deltaZ = 300):
    if interpModel == None:
        genTheoreticalModel(MetaData.TIRFDefault)
        
    im = zeros((len(X), len(Y)), 'f')

    deltaY = (Y[1] - Y[0])*deltaY #convert to nm
    #print deltaY

    if fluors == None:
        return im

    P = arange(0,1.01,.1)

    A = zeros(len(fluors.fl))

    for n  in range(numSubSteps):
        A += fluors.illuminate(laserPowers,intTime/numSubSteps)

    flOn = where(A > 0)[0]

    #print flOn

    for i in flOn:
       ix = abs(X - fluors.fl['x'][i]).argmin()
       iy = abs(Y - deltaY - fluors.fl['y'][i]).argmin()

       imp =fluors.fl[i]['spec'][0]*A[i]*1e3*interp3(X[(ix - roiSize):(ix + roiSize + 1)] - fluors.fl['x'][i], Y[(iy - roiSize):(iy + roiSize + 1)] - (fluors.fl['y'][i]+ deltaY), z - fluors.fl['z'][i])
       
       if not imp.shape[2] == 0:
           im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp[:,:,0]

       iy2 = abs(flipud(Y) - deltaY - fluors.fl['y'][i]).argmin()

       imp =fluors.fl[i]['spec'][1]*A[i]*1e3*interp3(X[(ix - roiSize):(ix + roiSize + 1)] - fluors.fl['x'][i], Y[(iy - roiSize):(iy + roiSize + 1)] - (fluors.fl['y'][i] + deltaY), z - fluors.fl['z'][i]+deltaZ)

       if not imp.shape[2] == 0:
           im[(ix - roiSize):(ix + roiSize + 1), (iy2 - roiSize):(iy2 + roiSize + 1)] += imp[:, ::-1, 0]
 

    return im

def simPalmImFBP(X,Y, z, fluors, intTime=.1, numSubSteps=10, roiSize=10, laserPowers = [.1,1], deltaY=64, deltaZ = 500):
    im = zeros((len(X), len(Y)), 'f')

    deltaY = (Y[1] - Y[0])*deltaY #convert to nm
    #print deltaY

    if fluors == None:
        return im

    P = arange(0,1.01,.1)

    A = zeros(len(fluors.fl))

    for n  in range(numSubSteps):
        A += fluors.illuminate(laserPowers,intTime/numSubSteps)

    flOn = where(A > 0)[0]

    #print flOn

    for i in flOn:
       ix = abs(X - fluors.fl['x'][i]).argmin()
       iy = abs(Y - deltaY - fluors.fl['y'][i]).argmin()

       imp =genWidefieldPSF(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)], z, P,A[i]*1e3, fluors.fl['x'][i], fluors.fl['y'][i] + deltaY , fluors.fl['z'][i])
       im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp[:,:,0]


       ix = abs(X - fluors.fl['x'][i]).argmin()
       iy = abs(flipud(Y) - deltaY - fluors.fl['y'][i]).argmin()
       imp =genWidefieldPSF(X[(ix - roiSize):(ix + roiSize + 1)], flipud(Y)[(iy - roiSize):(iy + roiSize + 1)], z + deltaZ, P,A[i]*1e3, fluors.fl['x'][i], fluors.fl['y'][i] + deltaY, fluors.fl['z'][i])
       im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp[:,:,0]

    return im
