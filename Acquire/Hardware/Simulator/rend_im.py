from PYME.PSFGen import *
from scipy import *
import fluor
from PYME.Analysis import MetaData

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

        interpModel = genWidefieldPSF(IntXVals, IntYVals, IntZVals, P,1e3, 0, 0, 0, 2*pi/525, 1.47, 10e3)

        interpModel = interpModel/interpModel.max() #normalise to 1

genTheoreticalModel(MetaData.TIRFDefault)

def setModel(mod, md):
    global IntXVals, IntYVals, IntZVals, interpModel, dx, dy, dz 

    IntXVals = 1e3*md.voxelsize.x*mgrid[-(mod.shape[0]/2.):(mod.shape[0]/2.)]
    IntYVals = 1e3*md.voxelsize.y*mgrid[-(mod.shape[1]/2.):(mod.shape[1]/2.)]
    IntZVals = 1e3*md.voxelsize.z*mgrid[-(mod.shape[2]/2.):(mod.shape[2]/2.)]

    dx = md.voxelsize.x*1e3
    dy = md.voxelsize.y*1e3
    dz = md.voxelsize.z*1e3

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


def simPalmImFI(X,Y, z, fluors, intTime=.1, numSubSteps=10, roiSize=20, laserPowers = [.1,1]):
    im = zeros((len(X), len(Y)), 'f')

    if fluors == None:
        return im

    #P = arange(0,1.01,.1)

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

       imp =interp(X[(ix - roiSize):(ix + roiSize + 1)] - fluors.fl['x'][i], Y[(iy - roiSize):(iy + roiSize + 1)] - fluors.fl['y'][i], z - fluors.fl['z'][i])* A[i]
       #print imp.shape
       if not imp.shape[2] == 0:
           im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp[:, :, 0]

    return im



def simPalmImFSpec(X,Y, z, fluors, intTime=.1, numSubSteps=10, roiSize=10, laserPowers = [.1,1], deltaY=64):
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
