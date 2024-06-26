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
    
from PYME.contrib import dispatch
class StandardFrameSource(object):
    '''This is a simple source which emits frames once per polling interval of the frameWrangler 
    (i.e. corresponding to the onFrameGroup signal of the frameWrangler).
    
    The intention is to reproduce the historical behaviour of the drift tracking code, whilst
    abstracting some of the detailed knowledge of frame handling out of the actual tracking code. 
    
    '''
    def __init__(self, frameWrangler):
        self._fw = frameWrangler
        self._on_frame = dispatch.Signal(['frameData'])
        self._fw.onFrameGroup.connect(self.tick)


    def tick(self, *args, **kwargs):
        self._on_frame.send(sender=self, frameData=self._fw.currentFrame)

    @property
    def shape(self):
        return self._fw.currentFrame.shape
    
    def connect(self, callback):
        self._on_frame.connect(callback)

    def disconnect(self, callback):
        self._on_frame.disconnect(callback)

class OIDICFrameSource(StandardFrameSource):
    """ Emit frames from the camera to the tracking code only for a single OIDIC orientation.

        Currently a straw man / skeleton pending details of OIDIC code.

        TODO - should this reside here, or with the other OIDIC code (which I believe to be in a separate repo)?
    
    """

    def __init__(self, frameWrangler, oidic_controller, oidic_orientation=0):
        #super().__init__(frameWrangler)
        self._fw = frameWrangler
        self._on_frame = dispatch.Signal(['frameData'])
        
        # connect to onFrame rather than onFrameGroup so we get the
        # current frame which as opposed to an older one.
        # this is important as the OIDIC orientation is expected to change between frames.
        # NOTE: we still assume that this connection happens before any of the acquisition
        # classes connect to onFrame, so that we get the frame data (and microscope state)
        # before the acquisition classes have had a chance to modify the state (e.g. by changing the DIC orientation).
        # TODO: This should be pretty safe, as the FrameSource is usually created in the init script, 
        # but can we make this robust agains the assumed ordering (execution of handlers in order of addition is an undocumented feature of dispatch)?
        self._fw.onFrame.connect(self.tick)

        self._oidic = oidic_controller

        self._last_tick_time = 0
        # throttle to ensure 50 ms delay between completion of computation on one frame and start of computation on the next.
        # the 50ms is emperical, but should (hopefully) be long enough to let anything else which is hooked to onFrame run. 
        self._tick_throttle = 0.05 
        #self._target_orientation = oidic_orientation

    def tick(self, frameData, **kwargs):
        # Because we are connected to onFrame rather than onFrameGroup (which is inherently throttled by the GUI loop),
        # we need to throttle the signal ourselves to avoid overwhelming things
        # when the frameWrangler is running at high speed (i.e. drift computation time
        # is slower than or on the order of the camera integration time). 
        if (time.time() - self._last_tick_time) < self._tick_throttle:
            return
        
        #self._target_orientation = self._oidic.home_channel()
        #if self._oidic.current_channel == self._target_orientation:
        if self._oidic.current_channel == self._oidic.home_channel():
            # send with the frame data of the frame which triggered the signal, rather than frameWrangler.currentFrame, which lags.
            self._on_frame.send(sender=self, frameData=frameData)
            self._last_tick_time = time.time() 
        else:
            # clobber all frames coming from camera when not in the correct DIC orientation
            pass
class Correlator(object):
    def __init__(self, scope, piezo=None, frame_source=None, sub_roi=None, focusTolerance=.05, deltaZ=0.2, stackHalfSize=35):
        self.piezo = piezo

        if frame_source is None:
            self.frame_source = StandardFrameSource(scope.frameWrangler)
        else:
            self.frame_source = frame_source

        self.sub_roi = sub_roi

        self.focusTolerance = focusTolerance #how far focus can drift before we correct
        self.deltaZ = deltaZ #z increment used for calibration
        self.stackHalfSize = stackHalfSize
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
        
    def _initialise(self, frame_data):
        d = 1.0*frame_data.squeeze()        
        
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

        
        
    def _setRefN(self, frame_data, N):
        d = 1.0*frame_data.squeeze()
        ref = d/d.mean() - 1
        self.refImages[:,:,N] = ref        
        self.calFTs[:,:,N] = ifftn(ref)
        self.calImages[:,:,N] = ref*self.mask
        
    def set_subroi(self, bounds):
        """ Set the position of the roi to crop

        Parameters
        ----------

        position : tuple
            The pixel position (x0, x1, y0, y1) in int
        """

        self.sub_roi = bounds
        self.reCalibrate()

    def _crop_frame(self, frame_data):
        if self.sub_roi is None:
            return frame_data.squeeze()   # we may as well do the squeeze here to avoid lots of squeezes elsewhere
        else:
            x0, x1, y0, y1 = self.sub_roi
            return frame_data.squeeze()[x0:x1, y0:y1]


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


    def set_delta_Z(self, delta):
        """ Set the Z increment for calibration stack

        Parameters
        ----------

        delta : float
            The delta in um. This should be the distance over which changes in PSF intensity with depth 
            can be approximated as being linear, with an upper bound of the Nyquist sampling in Z. 
            At Nyquist sampling, the linearity assumption is already getting a bit tenuous. Default = 0.2 um, 
            which is approximately Nyquist sampled at 1.4NA.
        """

        self.deltaZ = delta

    def get_delta_Z(self):
        return self.deltaZ


    def set_stack_halfsize(self, halfsize):
        """ Set the calibration stack half size

        This dictates the maximum size of z-stack you can record whilst retaining focus lock. The resulting 
        calibration range can be calculated as deltaZ*(2*halfsize), and should extend about 1 micron above 
        and below the size of the the largest z-stack to ensure that lock can be maintained at the edges of 
        the stack. The default of 35 gives about 12 um of axial range.

        Parameters
        ----------

        halfsize : int
        """

        self.stackHalfSize = halfsize

    def get_stack_halfsize(self):
        return self.stackHalfSize
    

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
        
    def compare(self, frame_data):
        d = 1.0*frame_data.squeeze()
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
        
    
    def tick(self, frameData = None, **kwargs):
        if frameData is None:
            raise ValueError('frameData must be specified')
        else:
            frameData = self._crop_frame(frameData)
        
        targetZ = self.piezo.GetTargetPos(0)
        
        #if not 'mask' in dir(self) or not self.frame_source.shape[:2] == self.mask.shape[:2]:
        if not 'mask' in dir(self) or not frameData.shape[:2] == self.mask.shape[:2]:
            self._initialise(frameData)
            
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
                self._setRefN(frameData, int(self.calibState - 1))
                self.piezo.MoveTo(0, self.calPositions[int(self.calibState)])
            
			
            #increment our calibration state
            self.calibState += 0.5
            
        elif (self.calibState == self.NCalibStates):
            # print "cal finishing"
            self._setRefN(frameData, int(self.calibState - 1))
            
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
            dx, dy, dz, cCoeff, dzcorr, nomPos, posInd, calPos, posDelta = self.compare(frameData)
            
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
        self.frame_source.connect(self.tick)
        self.tracking = True
        
    def deregister(self):
        self.frame_source.disconnect(self.tick)
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
