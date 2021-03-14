# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:02:50 2014

@author: David Baddeley
"""

import numpy as np
# from pylab import fftn, ifftn, fftshift, ifftshift
from numpy.fft import fftn, ifftn, fftshift, ifftshift

import time
from scipy import ndimage
from PYME.Acquire import eventLog
#from PYME.gohlke import tifffile as tif

#import Pyro.core
#import Pyro.naming
import threading
from PYME.misc.computerName import GetComputerName

import logging
logger = logging.getLogger(__name__)

def correlateFrames(A, B):
    A = A.squeeze()/A.mean() - 1
    B = B.squeeze()/B.mean() - 1
    
    X, Y = np.mgrid[0.0:A.shape[0], 0.0:A.shape[1]]
    
    C = ifftshift(np.abs(ifftn(fftn(A)*ifftn(B))))
    
    Cm = C.max()    
    
    Cp = np.maximum(C - 0.5*Cm, 0)
    Cpsum = Cp.sum()
    
    x0 = (X*Cp).sum()/Cpsum
    y0 = (Y*Cp).sum()/Cpsum
    
    return x0 - A.shape[0]/2, y0 - A.shape[1]/2, Cm, Cpsum
    
    
def correlateAndCompareFrames(A, B):
    A = A.squeeze()/A.mean() - 1
    B = B.squeeze()/B.mean() - 1
    
    X, Y = np.mgrid[0.0:A.shape[0], 0.0:A.shape[1]]
    
    C = ifftshift(np.abs(ifftn(fftn(A)*ifftn(B))))
    
    Cm = C.max()    
    
    Cp = np.maximum(C - 0.5*Cm, 0)
    Cpsum = Cp.sum()
    
    x0 = (X*Cp).sum()/Cpsum
    y0 = (Y*Cp).sum()/Cpsum
    
    dx, dy = x0 - A.shape[0]/2, y0 - A.shape[1]/2
    
    As = ndimage.shift(A, [-dx, -dy])
    
    #print A.shape, As.shape
    
    return (As -B).mean(), dx, dy
    
    
class Correlator(object):
    def __init__(self, scope, piezo=None):
        self.scope = scope
        self.piezo = piezo
        
        self.focusTolerance = .05 #how far focus can drift before we correct
        self.deltaZ = 0.2 #z increment used for calibration
        self.stackHalfSize = 35
        self.NCalibStates = 2*self.stackHalfSize + 1
        self.calibState = 0

        self.tracking = False
        self.lockActive = False

        self.lockFocus = False
        self.logShifts = True
        
        self._last_target_z = -1
        #self.initialise()
#        self.buffer = []
        self.WantRecord = False
        self.minDelay = 10
        self.maxfac = 1.5e3
        self.Zfactor = 1.0
        
    def initialise(self):
        d = 1.0*self.scope.frameWrangler.currentFrame.squeeze()        
        
        self.X, self.Y = np.mgrid[0.0:d.shape[0], 0.0:d.shape[1]]
#        self.X -= d.shape[0]/2
#        self.Y -= d.shape[1]/2
        self.X -= np.ceil(d.shape[0]*0.5)
        self.Y -= np.ceil(d.shape[1]*0.5)
        
        #we want to discard edges after accounting for x-y drift
        self.mask = np.ones_like(d)
        self.mask[:10, :] = 0
        self.mask[-10:, :] = 0
        self.mask[:, :10] = 0
        self.mask[:,-10:] = 0
        
        self.calibState = 0 #completely uncalibrated
        
        self.corrRef = 0

        
        self.lockFocus = False
        self.lockActive = False
        self.logShifts = True
        self.lastAdjustment = 5 
        self.homePos = self.piezo.GetPos(0)
        
        
        self.history = []
        
        self.historyCorrections = []

        
    # def setRefA(self):
    #     d = 1.0*self.scope.frameWrangler.currentFrame.squeeze()
    #     self.refA = d/d.mean() - 1        
    #     self.FA = ifftn(self.refA)
    #     self.refA *= self.mask
        
    # def setRefB(self):
    #     d = 1.0*self.scope.frameWrangler.currentFrame.squeeze()
    #     self.refB = d/d.mean() - 1
    #     self.refB *= self.mask        
        
    # def setRefC(self):
    #     d = 1.0*self.scope.frameWrangler.currentFrame.squeeze()
    #     self.refC = d/d.mean() - 1
    #     self.refC *= self.mask
        
    #     self.dz = (self.refC - self.refB).ravel()
    #     self.dzn = 2./np.dot(self.dz, self.dz)
        
    def setRefN(self, N):
        d = 1.0*self.scope.frameWrangler.currentFrame.squeeze()
        ref = d/d.mean() - 1
        self.refImages[:,:,N] = ref        
        self.calFTs[:,:,N] = ifftn(ref)
        self.calImages[:,:,N] = ref*self.mask
        
    #def setRefD(self):
    #    self.refD = (1.0*self.d).squeeze()/self.d.mean() - 1 
    #    self.refD *= self.mask
        
        #self.dz = (self.refC - self.refA).ravel()

    def set_focus_tolerance(self, tolerance):
        """ Set the tolerance for locking position

        Parameters
        ----------

        tolerance : float
            The tolerance in um
        """

        self.focusTolerance = tolerance

    def get_focus_tolerance(self):
        return self.focusTolerance

    def set_focus_lock(self, lock=True):
        """ Set locking on or off

        Parameters
        ----------

        lock : bool
            whether the lock should be on
        """

        self.lockFocus = lock

    def get_focus_lock(self):
        return self.lockFocus

    def get_history(self, length=1000):
        try:
            return self.history[-length:]
        except AttributeError:
            return []

    def get_calibration_state(self):
        """ Returns the current calibration state as a tuple:

        (currentState, numStates)

        calibration is complete when currentState == numStates.
        """

        return self.calibState, self.NCalibStates

    def is_tracking(self):
        return self.tracking

    def get_offset(self):
        return self.piezo.GetOffset()

    def set_offset(self, offset):
        self.piezo.SetOffset(offset)
        
    def compare(self):
        d = 1.0*self.scope.frameWrangler.currentFrame.squeeze()
        dm = d/d.mean() - 1
        
        #where is the piezo suppposed to be
        #nomPos = self.piezo.GetPos(0)
        nomPos = self.piezo.GetTargetPos(0)
        
        #find closest calibration position
        posInd = np.argmin(np.abs(nomPos - self.calPositions))
        
        #dz = float('inf')
        #count = 0
        #while np.abs(dz) > 0.5*self.deltaZ and count < 1:
        #    count += 1
        
        #retrieve calibration information at this location        
        calPos = self.calPositions[posInd]
        FA = self.calFTs[:,:,posInd]
        refA = self.calImages[:,:,posInd] 

        ddz = self.dz[:,posInd]
        dzn = self.dzn[posInd]
        
        #what is the offset between our target position and the calibration position         
        posDelta = nomPos - calPos
        
        #print('%s' % [nomPos, posInd, calPos, posDelta])
        
        #find x-y drift
        C = ifftshift(np.abs(ifftn(fftn(dm)*FA)))
        
        Cm = C.max()    
        
        Cp = np.maximum(C - 0.5*Cm, 0)
        Cpsum = Cp.sum()
        
        dx = (self.X*Cp).sum()/Cpsum
        dy = (self.Y*Cp).sum()/Cpsum
        
        ds = ndimage.shift(dm, [-dx, -dy])*self.mask
        
        #print A.shape, As.shape
        
        self.ds_A = (ds - refA)
        
        #calculate z offset between actual position and calibration position
        dz = self.deltaZ*np.dot(self.ds_A.ravel(), ddz)*dzn
        
        #posInd += np.round(dz / self.deltaZ)
        #posInd = int(np.clip(posInd, 0, self.NCalibStates))
            
#            print count, dz
        
        #add the offset back to determine how far we are from the target position
        dz = dz - posDelta
        
#        if 1000*np.abs((dz + posDelta))>200 and self.WantRecord:
            #dz = np.median(self.buffer)
#            tif.imsave('C:\\Users\\Lab-test\\Desktop\\peakimage.tif', d)
            # np.savetxt('C:\\Users\\Lab-test\\Desktop\\parameter.txt', self.buffer[-1])
            #np.savetxt('C:\\Users\\Lab-test\\Desktop\\posDelta.txt', posDelta)
#            self.WantRecord = False

        
        #return dx, dy, dz + posDelta, Cm, dz, nomPos, posInd, calPos, posDelta
        return dx, dy, dz, Cm, dz, nomPos, posInd, calPos, posDelta
        
    
    def tick(self, **kwargs):
        targetZ = self.piezo.GetTargetPos(0)
        
        if not 'mask' in dir(self) or not self.scope.frameWrangler.currentFrame.shape[:2] == self.mask.shape[:2]:
            self.initialise()
            
        #called on a new frame becoming available
        if self.calibState == 0:
            #print "cal init"
            #redefine our positions for the calibration
            self.homePos = self.piezo.GetPos(0)
            self.calPositions = self.homePos + self.deltaZ*np.arange(-float(self.stackHalfSize), float(self.stackHalfSize + 1))
            self.NCalibStates = len(self.calPositions)
            
            self.refImages = np.zeros(self.mask.shape[:2] + (self.NCalibStates,))
            self.calImages = np.zeros(self.mask.shape[:2] + (self.NCalibStates,))
            self.calFTs = np.zeros(self.mask.shape[:2] + (self.NCalibStates,), dtype='complex64')
            
            self.piezo.MoveTo(0, self.calPositions[0])
            
            #self.piezo.SetOffset(0)
            self.calibState += .5
        elif self.calibState < self.NCalibStates:
            # print "cal proceed"
            if (self.calibState % 1) == 0:
                #full step - record current image and move on to next position
                self.setRefN(int(self.calibState - 1))
                self.piezo.MoveTo(0, self.calPositions[int(self.calibState)])
            
			
            #increment our calibration state
            self.calibState += 0.5
            
        elif (self.calibState == self.NCalibStates):
            # print "cal finishing"
            self.setRefN(int(self.calibState - 1))
            
            #perform final bit of calibration - calcuate gradient between steps
            #self.dz = (self.refC - self.refB).ravel()
            #self.dzn = 2./np.dot(self.dz, self.dz)
            self.dz = np.gradient(self.calImages)[2].reshape(-1, self.NCalibStates)
            self.dzn = np.hstack([1./np.dot(self.dz[:,i], self.dz[:,i]) for i in range(self.NCalibStates)])
            
            self.piezo.MoveTo(0, self.homePos)
            
            #reset our history log
            self.history = []
            self.historyCorrections = []
            
            self.calibState += 1
            
        elif (self.calibState > self.NCalibStates) and np.allclose(self._last_target_z, targetZ):
            # print "fully calibrated"
            dx, dy, dz, cCoeff, dzcorr, nomPos, posInd, calPos, posDelta = self.compare()
            
            self.corrRef = max(self.corrRef, cCoeff)
            
            #print dx, dy, dz
            
            #FIXME: logging shouldn't call piezo.GetOffset() etc ... for performance reasons
            self.history.append((time.time(), dx, dy, dz, cCoeff, self.corrRef, self.piezo.GetOffset(), self.piezo.GetPos(0)))
            eventLog.logEvent('PYME2ShiftMeasure', '%3.4f, %3.4f, %3.4f' % (dx, dy, dz))
            
            self.lockActive = self.lockFocus and (cCoeff > .5*self.corrRef)
            if self.lockActive:
                if abs(self.piezo.GetOffset()) > 20.0:
                    self.lockFocus = False
                    logger.info("focus lock released")
                if abs(dz) > self.focusTolerance and self.lastAdjustment >= self.minDelay:
                    zcorr = self.piezo.GetOffset() - dz
                    if zcorr < - self.maxfac*self.focusTolerance:
                        zcorr = - self.maxfac*self.focusTolerance
                    if zcorr >  self.maxfac*self.focusTolerance:
                        zcorr = self.maxfac*self.focusTolerance
                    self.piezo.SetOffset(zcorr)
                    
                    #FIXME: this shouldn't be needed as it is logged during LogShifts anyway
                    self.piezo.LogFocusCorrection(zcorr) #inject offset changing into 'Events'
                    eventLog.logEvent('PYME2UpdateOffset', '%3.4f' % (zcorr))
                    
                    self.historyCorrections.append((time.time(), dz))
                    self.lastAdjustment = 0
                else:
                    self.lastAdjustment += 1
            
            if self.logShifts:
                self.piezo.LogShifts(dx, dy, dz, self.lockActive)
        
        self._last_target_z = targetZ                    
            
    def reCalibrate(self):
        self.calibState = 0
        self.corrRef = 0
        self.lockActive = False
        
    def register(self):
        #self.scope.frameWrangler.WantFrameGroupNotification.append(self.tick)
        self.scope.frameWrangler.onFrameGroup.connect(self.tick)
        self.tracking = True
        
    def deregister(self):
        #self.scope.frameWrangler.WantFrameGroupNotification.remove(self.tick)
        self.scope.frameWrangler.onFrameGroup.disconnect(self.tick)
        self.tracking = False
    
    # def setRefs(self, piezo):
    #     time.sleep(0.5)
    #     p = piezo.GetPos()
    #     self.setRefA()
    #     piezo.MoveTo(0, p -.2)
    #     time.sleep(0.5)
    #     self.setRefB()
    #     piezo.MoveTo(0,p +.2)
    #     time.sleep(0.5)
    #     self.setRefC()
    #     piezo.MoveTo(0, p)


def correlator(scope, piezo=None):
    # API compatible constructor for Py2
    import Pyro.core
    class klass(Pyro.core.ObjBase):
        def __init__(self, scope, piezo=None):
            Pyro.core.ObjBase.__init__(self)
            Correlator.__init__(self, scope, piezo=piezo)

    return klass


class ServerThread(threading.Thread):
    def __init__(self, driftTracker):
        threading.Thread.__init__(self)
        import Pyro.core
        import Pyro.naming

        import socket
        ip_addr = socket.gethostbyname(socket.gethostname())
        
        compName = GetComputerName()
        
        Pyro.core.initServer()

        pname = "%s.DriftTracker" % compName
        
        try:
            from PYME.misc import pyme_zeroconf 
            ns = pyme_zeroconf.getNS()
        except:
            ns=Pyro.naming.NameServerLocator().getNS()

            if not compName in [n[0] for n in ns.list('')]:
                ns.createGroup(compName)

            #get rid of any previous instance
            try:
                ns.unregister(pname)
            except Pyro.errors.NamingError:
                pass        
        
        self.daemon=Pyro.core.Daemon(host = ip_addr)
        self.daemon.useNameServer(ns)
        
        self.driftCorr = piezoOffsetProxy(driftTracker)
        
        #pname = "%s.Piezo" % compName
        
        
        
        uri=self.daemon.connect(self.driftCorr,pname)
        
    def run(self):
        #print 'foo'
        #try:
        self.daemon.requestLoop()
        #finally:
        #    daemon.shutdown(True)
        
    def cleanup(self):
        logger.info('Shutting down drift tracking Server')
        self.daemon.shutdown(True)
    
def getClient(compName = GetComputerName()):
    try:
        from PYME.misc import pyme_zeroconf 
        ns = pyme_zeroconf.getNS()
        URI = ns.resolve('%s.DriftTracker' % compName)
    except:
        URI ='PYRONAME://%s.DriftTracker'%compName

    return Pyro.core.getProxyForURI(URI)
