 #!/usr/bin/python

##################
# previewaquisator.py
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
"""Implementation of the FrameWrangler class (previously PreviewAcquisator)
which manages the flow of information from a camera into the rest of the program,
letting dependant parts of the program know when new data arrives via signals.

As hardware control in PYMEAcquire is largely driven by and synchronized with 
camera frames, this also encompasses the 'heart' or core of the program which 
everything else responds to."""

import numpy as np

import ctypes
import sys

import logging
logger=logging.getLogger(__name__)

if sys.platform == 'win32':
    memcpy = ctypes.cdll.msvcrt.memcpy
elif sys.platform == 'darwin':
    memcpy = ctypes.CDLL('libSystem.dylib').memcpy
else: #linux
    memcpy = ctypes.CDLL('libc.so.6').memcpy

import time
import traceback

from PYME.contrib import dispatch
import warnings

from PYME.Acquire import eventLog
import threading
#sfrom PYME.ui import mytimer

class FrameWrangler(object):
    """
    Grabs frames from the camera buffers

    Notes
    -----
    dispatch Signals are used to allow other files to listen to key events 
    happinging within the acquisition.

    Attributes
    ----------
    onFrame : dispatch.Signal
        Called once per new-frame appearing in our buffer; used to pass
        frame data to e.g. spoolers. Note that while onFrame gets called once
        per new frame, the new frames are only checked for once per polling
        cycle, meaning that this event will be fired N-times for each 
        `onFrameGroup` event.
    onFrameGroup : dispatch.Signal
        Called on each new frame group (once per polling interval) - use for 
        updateing GUIs etc.
    onStop : dispatch.Signal
        Called when acquisition stops.

    """
    def __init__(self, _cam, _ds = None, event_loop=None):
        #wx.EvtHandler.__init__(self)
        #self.timer = wx.Timer(self)
        #self.Bind(wx.EVT_TIMER, self.Notify)
        
        if event_loop is None:
            from PYME.ui import mytimer
            self._event_loop = mytimer
        else:
            self._event_loop = event_loop
        
        self.timer = self._event_loop.SingleTargetTimer(self.Notify)
        
        self.currentFrame = _ds
        self.cam = _cam

        self.aqOn = False

        #list of functions to call to see if we ought to wait on any hardware
        self.HardwareChecks = []
        
        #should we start a new exposure on the next timer check?
        self.needExposureStart = False
        
        #did the buffer overflow?
        self.bufferOverflowed = False

        self.tLastFrame=0
        self.tThisFrame=0
        self.nFrames = 0
        self.n_frames_in_group=0
        self.tl=0
        
        self.inNotify = False
        self._notify_lock = threading.Lock()
        
        #Signals
        ##########################
        # these allow other files to listen to key events happingin within the acquisiotn        
        #new style signals - these will replace the WantFrameNotification etc ...
        #which are currently being kept for backwards compatibility
        
        self.onFrame = dispatch.Signal(['frameData']) #called each time a new frame appears in our buffer
        self.onFrameGroup = dispatch.Signal() #called on each new frame group (once per polling interval) - use for updateing GUIs etc.
        self.onStop = dispatch.Signal()
        self.onStart = dispatch.Signal()

        # should the thread which polls the camera still be running?
        self._poll_camera = True

        self._current_frame_lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._polling_interval = 0.01  # time between calls to poll the camera [s]
        self._poll_thread = threading.Thread(target=self._poll_loop)
        self._poll_thread.start()
        
    def __del__(self):
        self.destroy()
        
    def destroy(self):
        self._poll_camera = False
        

    def Prepare(self, keepds=True):
        """Prepare for acquisition by allocating the buffer which will store the
        data we recieve. The buffer stores a single frame, and all frames pass
        through this buffer. The current state of the buffer is accessible via
        the currentFrame variable.

        Parameters
        ----------
        keepds: Whether or not to keep the previously allocated array        
        
        """
        
        #what byte-order does the camera use for its frames?
        try:
            if not (self.cam.order == self.order):
                keepds = False
                self.order = self.cam.order
        except AttributeError:
            # not all cameras expose the order property, use Fortran ordering
            # by default (i.e. x then y)
            # note: This has a lot to do with me liking to use the 0th 
            # array index as x, which might (in retrospect) have been a bad design choice
            self.order = 'F'
            self.cam.order = 'F'
            keepds=False
        
        if (self.currentFrame is None) or(self.currentFrame.shape[0] != self.cam.GetPicWidth()) or (self.currentFrame.shape[1] != self.cam.GetPicHeight()):
            keepds = False



        if not keepds:
            self.currentFrame = np.zeros([self.cam.GetPicWidth(), self.cam.GetPicHeight(), 
                                1], dtype = 'uint16', order = self.order)
            
        self._cf = self.currentFrame
   

            
    def getFrame(self, colours=None):
        """Ask the camera to put a frame into our buffer"""
        #logger.debug('acquire _current_frame_lock in getFrame()')
        with self._current_frame_lock:
            self._cf = np.empty([1, self.cam.GetPicWidth(), self.cam.GetPicHeight(),
	                                ], dtype = 'uint16', order = self.order)
	        
            if getattr(self.cam, 'numpy_frames', False):
                cs = self._cf[0,:,:] #self.currentFrame[:,:,0]
            else:
                cs = self._cf.ctypes.data
	            
            #Get camera to insert data into our array (results passed back "by reference")
            #this is a kludge/artifact of an old call into c-code
            #in this context cs is a pointer to the memory we want the frame to go into
            #for newer cameras, we pass a numpy array object, and the camera code
            #copies the data into that array.
            self.cam.ExtractColor(cs,0)

        #logger.debug('release _current_frame_lock in getFrame()')
        return self._cf

    def purge(self):
        """purge (and discard) all remaining frames in the camera buffer"""
        while(self.cam.ExpReady()):
            self.curMemChn = 0
            self.getFrame(self.BW)
            self.curMemChn = 0



    def onExpReady(self): 
        """ There is an exposure waiting in the Camera,
        pull it down into our local storage and notify any listeners
        """
        
        #determine if we are running in contiuous or single shot mode.
        #previous cameras only supported single shot mode, and don't implement
        #the contMode attribute, so catch this and assign to single shot
        try:
            contMode = self.cam.contMode
        except AttributeError:
            contMode = False
        
        # Pull the existing data from the camera
        try:
            d = self.getFrame()

            #notify anyone who cares that we've just got a new frame
            ### NEW: now send a copy so that receivers don't need to copy it. This results in a) more predictable behaviour
            # and b) sets the stage for passing raw frames to spoolers without any copying
            #d = self.dsa.copy()
            #d = np.empty_like(self.dsa)
            #ctypes.cdll.msvcrt.memcpy(d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.dsa.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), d.nbytes)
            self.onFrame.send(sender=self, frameData=d)
        except:
            import traceback
            traceback.print_exc()

        finally:       
            if not contMode:
                #flag the need to start a new exposure
                #the exposure will be started in the main Notify method after 
                #any hardware movements etc that are triggered by the onFrame 
                #signal complete 
                self.needExposureStart = True

    def _poll_loop(self):
        """
        This loop runs in a background thread to continuously poll the camera 
        and deal with frames as they arrive. `FrameWrangler._polling_interval`
        is used to set the delay between loops.
        """
        while (self._poll_camera):
            if (not self.cam.CamReady()):# and self.piezoReady())):
                # Stop the aquisition if there is a hardware error
                self._event_loop.call_in_main_thread(self.stop)
            else:
                #is there a picture waiting for us?
                #if so do the relevant processing
                #otherwise do nothing ...
        
                #nFrames = 0 #number of frames grabbed this pass
        
                #bufferOverflowed = False
        
                while (self.cam.ExpReady()): #changed to deal with multiple frames being ready
                    if 'GetNumImsBuffered' in dir(self.cam):
                        bufferOverflowing = self.cam.GetNumImsBuffered() >= (self.cam.GetBufferSize() - 2)
                    else:
                        bufferOverflowing = False

                    if bufferOverflowing:
                        with self._notify_lock:
                            #acquire lock before flagging the buffer as overflowed so that we can be sure that StopAq
                            # gets called before StartAq in Notify()
                            
                            self.bufferOverflowed = True
                            logger.warning('Camera buffer overflowing - purging buffer')
                            eventLog.logEvent('Camera Buffer Overflow')
                            #stop the aquisition - we're going to restart after we're read out to purge the buffer
                            #doing it this way _should_ stop the black frames which I guess are being caused by the reading the frame which is
                            #currently being written to
                            self._event_loop.call_in_main_thread(self.cam.StopAq)
                        #self.needExposureStart = True
            
                    self.onExpReady()
                    self.n_frames_in_group += 1
        
            time.sleep(self._polling_interval)



    def getReqMemChans(self, colours):
        """  Use this function to calc how may channels to allocate when creating a new data stack """
        return 1


    def Notify(self, event=None):
        """Callback which is called regularly by a system timer to poll the
        camera"""
        
        #check to see if we are already running
        if self.inNotify:
            logger.debug('Already in notify, skip for now')
            return
            
        with self._notify_lock:
            #lock to prevent _poll_loop setting an overflowed flag while we're in here.
            try:
                self.inNotify = True
                self.te = time.clock()
                #print self.te - self.tl
                self.tl = self.te
                
                if (not self.cam.CamReady()):# and self.piezoReady())):
                    # Stop the aquisition if there is a hardware error
                    self.stop()
                    return
    
                if getattr(self.cam, 'hardware_overflowed', False):
                    self.cam.StopAq()
                    self.bufferOverflowed = True
    
    
                #is there a picture waiting for us?
                #if so do the relevant processing
                #otherwise do nothing ...
                
                #nFrames = 0 #number of frames grabbed this pass
                
                #bufferOverflowed = False
    
                # while(self.cam.ExpReady()): #changed to deal with multiple frames being ready
                #     if 'GetNumImsBuffered' in dir(self.cam):
                #         bufferOverflowing  = self.cam.GetNumImsBuffered() >= (self.cam.GetBufferSize() - 1)
                #     else:
                #         bufferOverflowing = False
                #     if bufferOverflowing:
                #         bufferOverflowed = True
                #         print('Warning: Camera buffer overflowing - purging buffer')
                #         eventLog.logEvent('Camera Buffer Overflow')
                #         #stop the aquisition - we're going to restart after we're read out to purge the buffer
                #         #doing it this way _should_ stop the black frames which I guess are being caused by the reading the frame which is
                #         #currently being written to
                #         self.cam.StopAq()
                #         #self.needExposureStart = True
                #
                #     self.onExpReady()
                #     nFrames += 1
                    #te= time.clock()
                    
                    #If we can't deal with the data fast enough (e.g. due to file i/o limitations) this can turn into an infinite loop -
                    #avoid this by bailing out with a warning if nFrames exceeds a certain value. This will probably lead to buffer overflows
                    #and loss of data, but is arguably better than an unresponsive app.
                    #This value is (currently) chosen fairly arbitrarily, taking the following facts into account:
                    #the buffer has enough storage for ~3s when running flat out,
                    #we're polling at ~5hz, and we should be able to get more frames than would be expected during the polling intervall to
                    #allow us to catch up following glitches of one form or another, although not too many more.
                if ('GetNumImsBuffered' in dir(self.cam)) and (self.n_frames_in_group > self.cam.GetBufferSize()/2):
                    logger.warning(('Not keeping up with camera, giving up with %d frames still in buffer' % self.cam.GetNumImsBuffered()))
                 
                # just copy data to the current frame once per frame group - individual frames don't get copied
                # directly calling memcpy is a bit of a cheat, but is significantly faster than the alternatives
                with self._current_frame_lock:
                    memcpy(self.currentFrame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                           self._cf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),self.currentFrame.nbytes)
                
    
                if self.bufferOverflowed:
                    logger.debug('Setting needStartExposure flag')
                    self.needExposureStart = True
    
                # See if we need to restart the exposure. This will happen if
                # a) we are in single shot mode
                # or b) the camera buffer overflowed
                if self.needExposureStart and self.checkHardware():
                    self.needExposureStart = False
                    self.bufferOverflowed = False
                    self.cam.StartExposure() #restart aquisition - this should purge buffer
                    
    
                if self.n_frames_in_group > 0:
                    #we got some frames, record timing info and let any listeners know
                    self.n_Frames += self.n_frames_in_group
                    
                    self.tLastFrame = self.tThisFrame
                    self.nFrames = self.n_frames_in_group
                    self.n_frames_in_group = 0
                    self.tThisFrame = time.clock()
                     
                    self.onFrameGroup.send(self)
               
            except:
                traceback.print_exc()
            finally:
                self.inNotify = False
                
                #restart the time so we get called again
                self.timer.start(self.tiint)
            
            
    @property
    def dsa(self):
        """Whatever frame is currently passing through the acquisition queue
        
        NB: this new code should use the more informatively named currentFrame
        property instead
        """
        warnings.warn('.dsa is deprecated, use .currentFrame instead', DeprecationWarning)
        return self.currentFrame

    def checkHardware(self):
        """Check to see if our hardware is ready for us to take the next frame
        
        NB: This is largely legacy code, as the camera is usually used in 
        free-running mode."""
        for callback in self.HardwareChecks:
            if not callback():
                logger.debug('Waiting for hardware')
                return False

        return True

    def stop(self):
        "Stop sequence aquisition"

        self.timer.stop()
        self.aqOn = False

        #logger.debug('acquire _current_frame_lock in stop()')
        with self._current_frame_lock:
            try: #deal with Andor without breaking sensicam
                self.cam.StopAq()
            except AttributeError:
                pass
        #logger.debug('release _current_frame_lock in stop()')
                
        self.onStop.send_robust(self)

    def start(self, tiint = 100):
        "Start aquisition"
        self.tiint = tiint

        self.zPos = 0

        #clear saturation intervened flag TODO - this probably doesn't belong here
        self.cam.saturationIntervened = False
        
        self.onStart.send_robust(self)

        #logger.debug('acquire _current_frame_lock in start')
        with self._current_frame_lock:
            self.cam.StartExposure()
        #logger.debug('release _current_frame_lock in start')

        self.aqOn = True

        self.t_start = time.time()
        self.n_Frames = 0

        #start our timer, this will call Notify
        self.timer.start(self.tiint)
        return True


    def isRunning(self):
        return self.aqOn

    def getFPS(self):
        t = time.time()
        dt = t - self.t_start

        fps = 1.0*self.n_Frames/(1e-5 + dt)
        
        return fps

