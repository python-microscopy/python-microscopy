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

import wx
#from PYME.cSMI import CDataStack, CDataStack_AsArray

import numpy as np

import time
import traceback

import dispatch

from PYME.Acquire import eventLog

#class dsFake(object):
#    def __init__(self, width, height, length, nChans):
#        self.width = width
#        self.height = height
#        self.length = length
#        self.nChans = nChans
        
        


class PreviewAquisator(wx.EvtHandler):
#    BW = 1
#    RED = 2
#    GREEN1 = 4
#    GREEN2 = 8
#    BLUE = 16

    def __init__(self, _chans, _cam, _shutters, _ds = None):
        wx.EvtHandler.__init__(self)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.Notify)
        
        self.chans = _chans
        #self.hwChans = _chans.hw
        #self.numHWChans = len(_chans.hw)
        #self.cols =  _chans.cols
        self.dsa = _ds
        self.cam = _cam
        self.shutters = _shutters
        self.loopnuf = 0
        self.aqOn = False
        #lists of functions to call on a new frame, and when the aquisition ends
        #self.WantFrameNotification = []
        #self.WantStopNotification = []
        #self.WantStartNotification = []
        #list of functions to call to see if we ought to wait on any hardware
        self.HardwareChecks = []
        #should we start a new exposure on the next timer check?
        self.needExposureStart = False

        self.tLastFrame=0
        self.tThisFrame=0
        self.nFrames = 0
        self.tl=0
        
        self.inNotify = False
        
        self.zPos = 0
        
        #will be needed to allow the display load to be minimised by, e.g. only updating display once per poll rather than once per frame
        #self.WantFrameGroupNotification = [] 
        
        
        #Signals
        ##########################
        # these allow other files to listen to key events happingin within the acquisiotn        
        #new style signals - these will replace the WantFrameNotification etc ...
        #which are currently being kept for backwards compatibility
        
        self.onFrame = dispatch.Signal(['frameData']) #called each time a new frame appears in our buffer
        self.onFrameGroup = dispatch.Signal() #called on each new frame group (once per polling interval) - use for updateing GUIs etc.
        self.onStop = dispatch.Signal()
        self.onStart = dispatch.Signal()
        

    def Prepare(self, keepds=False):
        """Prepare for acquisition by allocating the buffer which will store the
        data we recieve. The buffer stores a single frame, and all frames pass
        through this buffer. The current state of the buffer is accessible via
        the currentFrame variable.

        Parameters
        ----------
        keepds: Whether or not to keep the previously allocated array        
        
        """
        self.looppos=0
        self.curMemChn=0
        
        self.hwChans = self.chans.hw
        self.numHWChans = len(self.chans.hw)
        self.cols =  self.chans.cols
        
        order = 'F'
        if 'order' in dir(self.cam):
            order = self.cam.order  
        
        if (self.dsa == None or keepds == False):
            self.dsa = None
            #self.ds = CDataStack(self.cam.GetPicWidth(), self.cam.GetPicHeight(), 
            #    self.GetSeqLength(),self.getReqMemChans(self.cols))
            #self.dsa = CDataStack_AsArray(self.ds, 0)
            self.dsa = np.zeros([self.cam.GetPicWidth(), self.cam.GetPicHeight(), 
                                 self.GetSeqLength()], dtype = 'uint16', order = order)
   

            
    def getFrame(self, colours=None):
        """Ask the camera to put a frame into our buffer"""
        #print self.zPos
        if ('numpy_frames' in dir(self.cam)):
            cs = self.dsa[:,:,self.zPos]
        else:
            cs = self.dsa[:,:,self.zPos].ctypes.data
            
        #Get camera to insert data into our array (results passed back "by reference")
        #this is a kludge/artifact of an old call into c-code
        #in this context cs is a pointer to the memory we want the frame to go into
        #for newer cameras, we pass a numpy array object, and the camera code
        #copies the data into that array.
        self.cam.ExtractColor(cs,0)

    def purge(self):
        """purge (and discard) all remaining frames in the camera buffer"""
        while(self.cam.ExpReady()):
            self.curMemChn = 0
            self.getFrame(self.BW)
            self.curMemChn = 0



    def onExpReady(self): 
        """ There is an exposure waiting in the Camera,
            looppos inticates which hardware (shutter) channel we're currently on """
        
        self.loopnuf = self.loopnuf + 1

        #If this was the last set of shutter combinations, move us to the position
        # for the next slice.
        if (self.looppos == (self.numHWChans - 1)): 
            self.doPiezoStep()
        
        # Set the shutters for the next exposure
        #self.shutters.setShutterStates(self.hwChans[(self.looppos + 1)%self.numHWChans])
        self.shutters.setShutterStates(0)

        #self.Wait(15) #give shutters a chance to close - should fix hardware

        self.looppos = self.looppos + 1
	
        contMode = False
        if ('contMode' in dir(self.cam)): #hack for continous acquisition - do not want to/can't keep setting iTime
	        contMode =self.cam.contMode

        if ('itimes' in dir(self.chans) and not contMode): #maintain compatibility with old versions
            self.cam.SetIntegTime(self.chans.itimes[self.looppos%self.numHWChans])
            self.cam.SetCOC()
        
        self.shutters.setShutterStates(self.hwChans[(self.looppos)%self.numHWChans])
        #self.Wait(15)

        #self.cam.StartExposure()

        
        # Pull the existing data from the camera
        try:
            self.getFrame(self.cols[self.looppos-1])
        except:
            traceback.print_exc()
        finally:
        
            if not contMode:
                #flag the need to start a new exposure
                #print 'se'
                self.needExposureStart = True
                #self.cam.StartExposure()


            if (self.looppos >= self.numHWChans):
                self.looppos = 0
                self.curMemChn = 0

                #for a in self.WantFrameNotification:
                #    a(self)
                    
                #print 'onFrame'
                self.onFrame.send(sender=self, frameData=self.dsa)

            # If we're at the end of the Data Stack, then stop
            # Note that in normal sequence aquisition this is the line which determines how long to 
            # record for - in this class (ie the live preview) getNextDsSlice is defined such that 
            # it doesn't move through the stack, and always returns true, such that the aquisition 
            # continues for ever unless we stop it some other way (ie by clicking "Stop Live Preview")
            # in CRealAquisator it's overridden to behave in the right way.
            if not (self.getNextDsSlice()):
                 self.stop()



    def getReqMemChans(self, colours):
        """  Use this function to calc how may channels to allocate when creating a new data stack """
        return 1

#        t = 0
#        for c in colours:
#            if(c & self.BW): 
#                t = t + 1
#            if(c & self.RED):
#                t = t + 1
#            if(c & self.GREEN1):
#                t = t + 1
#            if(c & self.GREEN2):
#                t = t + 1
#            if(c & self.BLUE):
#                t = t + 1
#
#        return t


    def Notify(self, event=None):
        """Callback which is called regularly by a system timer to poll the
        camera"""
        
        #check to see if we are already running
        if self.inNotify:
            print('Already in notify, skip for now')
            return
            
        try:            
            self.inNotify = True
            "Should be called on each timer tick"
            self.te = time.clock()
            #print self.te - self.tl
            self.tl = self.te
            #print "Notify"
    
            #self.loopnuf = self.loopnuf + 1
            
            if (True): #check that we are aquiring
            
    	
                if(not (self.cam.CamReady() and self.piezoReady())):
                    # Stop the aquisition if there is a hardware error
                    self.stop()
                    return
    
                #is there a picture waiting for us?
                #if so do the relevant processing
                #otherwise do nothing ...
            	
                nFrames = 0 #number of frames grabbed this pass
                
                bufferOverflowed = False
    
                while(self.cam.ExpReady()): #changed to deal with multiple frames being ready
                    if 'GetNumImsBuffered' in dir(self.cam):
                        bufferOverflowing  = self.cam.GetNumImsBuffered() >= (self.cam.GetBufferSize() - 1)
                    else:
                        bufferOverflowing = False
                    if bufferOverflowing:
                        bufferOverflowed = True
                        print('Warning: Camera buffer overflowing - purging buffer')
                        eventLog.logEvent('Camera Buffer Overflow')
                        #stop the aquisition - we're going to restart after we're read out to purge the buffer
                        #doing it this way _should_ stop the black frames which I guess are being caused by the reading the frame which is
                        #currently being written to
                        self.cam.StopAq()
     
                    self.onExpReady()
                    nFrames += 1
                    #te= time.clock()
                    
                    #If we can't deal with the data fast enough (e.g. due to file i/o limitations) this can turn into an infinite loop -
                    #avoid this by bailing out with a warning if nFrames exceeds a certain value. This will probably lead to buffer overflows
                    #and loss of data, but is arguably better than an unresponsive app.
                    #This value is (currently) chosen fairly arbitrarily, taking the following facts into account: 
                    #the buffer has enough storage for ~3s when running flat out,
                    #we're polling at ~5hz, and we should be able to get more frames than would be expected during the polling intervall to
                    #allow us to catch up following glitches of one form or another, although not too many more.
                    if ('GetNumImsBuffered' in dir(self.cam)) and (nFrames > self.cam.GetBufferSize()/2):
                        print(('Warning: not keeping up with camera, giving up with %d frames still in buffer' % self.cam.GetNumImsBuffered()))
                        break
    
                if bufferOverflowed:
                    print('nse')
                    self.needExposureStart = True
    
                if self.needExposureStart and self.checkHardware():
                    self.needExposureStart = False
                    self.cam.StartExposure() #restart aquisition - this should purge buffer
                    
    
                #if 'nQueued' in dir(self.cam):
                #    print '\n', nFrames, self.cam.nQueued, self.cam.nFull , self.cam.doPoll
                if nFrames > 0:
                    self.n_Frames += nFrames
                    
                    self.tLastFrame = self.tThisFrame
                    self.nFrames = nFrames
                    self.tThisFrame = time.clock()
                     
                    self.onFrameGroup.send_robust(self)
            else:
                 self._stop()
        except:
            traceback.print_exc()
        finally:     
            self.inNotify = False
            #self.timer.StartOnce(self.tiint)
            self.timer.Start(self.tiint, wx.TIMER_ONE_SHOT)
            
            
    @property
    def currentFrame(self):
        """Whatever frame is currently passing through the acquisition queue
        
        NB: this is an attempt to give a more user friendly name to  .dsa       
        """
        return self.dsa

    def checkHardware(self):
        """Check to see if our hardware is ready for us to take the next frame
        
        NB: This is largely legacy code, as the camera is usually used in 
        free-running mode."""
        for callback in self.HardwareChecks:
            if not callback():
                print 'Waiting for hardware'
                return False

        return True

    def stop(self):
        "Stop sequence aquisition"

        self.timer.Stop()

        self.aqOn = False

        if 'StopAq' in dir(self.cam): #deal with Andor without breaking sensicam
            self.cam.StopAq()

        self.shutters.closeShutters(self.shutters.ALL)

        self.zPos = 0

        self.piezoGoHome()

        self.doStopLog()

                
        self.onStop.send_robust(self)

    def start(self, tiint = 100):
        "Start aquisition"
        self.tiint = tiint

        self.looppos = 0
        #self.ds.setZPos(0) #go to start of data stack
        self.zPos = 0
        
        #set the shutters up for the first frame
        self.shutters.setShutterStates(self.hwChans[self.looppos])

        #clear saturation intervened flag
        self.cam.saturationIntervened = False

        #move piezo to starting position
        self.setPiezoStartPos()

        self.doStartLog()
        
        self.onStart.send_robust(self)

        if ('itimes' in dir(self.chans)): #maintain compatibility with old versions
            self.cam.SetIntegTime(self.chans.itimes[self.looppos])
            self.cam.SetCOC()

        iErr = self.cam.StartExposure()
        self.cam.DisplayError(iErr)
	
        if (iErr < 0):
            self.stop()
            return False

        self.aqOn = True

        self.t_old = time.time()
        self.n_Frames = 0

        self.timer.Start(self.tiint, wx.TIMER_ONE_SHOT)
        return True


#    def Wait(self,iTime):
#        """ Dirty delay routine - blocks until given no of milliseconds has elapsed\n 
#            Probably best not to use with a delay of more than about a second or windows\n
#            could rightly assume that the programme is <not responding> """
#        time.sleep(iTime/1000)
#        #FirstTime = time.clock()
#        #dc = 0
#        #while(time.clock() < (FirstTime + iTime/1000)):
#        #    dc = dc + 1

    def isRunning(self):
        return self.aqOn

    def getFPS(self):
        t = time.time()
        dt = t - self.t_old

        fps = 1.0*self.n_Frames/(1e-5 + dt)
        self.t_old = t
        self.n_Frames = 0
        
        return fps

        #return 1.0*self.nFrames/(1e-5 + self.tThisFrame - self.tLastFrame)

    #place holders ... for overridden class which actually knows 
    #about the piezo
    def doPiezoStep(self):
        pass

    def piezoReady(self):
        return True

    def setPiezoStartPos(self):
        pass

    def getNextDsSlice(self):
        return True

    def _stop(self):
        self.stop()

    def doStartLog(self):
        pass

    def doStopLog(self):
        pass

    def GetSeqLength(self):
        return 1

    def piezoGoHome(self):
        pass
