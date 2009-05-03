from PYME.PSFGen import *
from scipy import *
import fluor

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
            
       imp =genWidefieldPSF(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)], z, P,A[i]*1e3, fluors.fl['x'][i], fluors.fl['y'][i], fluors.fl['z'][i],depthInSample=0)
       im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp[:,:,0] 

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
