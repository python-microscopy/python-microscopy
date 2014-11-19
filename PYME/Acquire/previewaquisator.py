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
from PYME.cSMI import CDataStack, CDataStack_AsArray

import time
import traceback

from PYME.Acquire import eventLog

class PreviewAquisator(wx.Timer):
    BW = 1
    RED = 2
    GREEN1 = 4
    GREEN2 = 8
    BLUE = 16

    def __init__(self, _chans, _cam, _shutters, _ds = None):
        wx.Timer.__init__(self)
        
        self.chans = _chans
        #self.hwChans = _chans.hw
        #self.numHWChans = len(_chans.hw)
        #self.cols =  _chans.cols
        self.ds = _ds
        self.cam = _cam
        self.shutters = _shutters
        self.loopnuf = 0
        self.aqOn = False
        #lists of functions to call on a new frame, and when the aquisition ends
        self.WantFrameNotification = []
        self.WantStopNotification = []
        self.WantStartNotification = []
        #list of functions to call to see if we ought to wait on any hardware
        self.HardwareChecks = []
        #should we start a new exposure on the next timer check?
        self.needExposureStart = False

        self.tLastFrame=0
        self.tThisFrame=0
        self.nFrames = 0
        self.tl=0
        
        self.inNotify = False
        
        #will be needed to allow the display load to be minimised by, e.g. only updating display once per poll rather than once per frame
        self.WantFrameGroupNotification = [] 

    def Prepare(self, keepds=False):
        self.looppos=0
        self.curMemChn=0
        
        self.hwChans = self.chans.hw
        self.numHWChans = len(self.chans.hw)
        self.cols =  self.chans.cols
   
        if (self.ds == None or keepds == False):
            self.ds = None
            self.ds = CDataStack(self.cam.GetPicWidth(), self.cam.GetPicHeight(), 
                self.GetSeqLength(),self.getReqMemChans(self.cols))
            self.dsa = CDataStack_AsArray(self.ds, 0)

            i = 0
            for j in range(len(self.cols)):
                a = self.chans.names[j]
                c = self.chans.cols[j]
                #for (a,c) in (self.chans.names, self.chans.cols):

                if(c & self.BW):
                    self.ds.setChannelName(i, (a + "_BW").encode())
                    i = i + 1
                if(c & self.RED):
                    self.ds.setChannelName(i, (a + "_R").encode())
                    i = i + 1
                if(c & self.GREEN1):
                    self.ds.setChannelName(i, (a + "_G1").encode())
                    i = i + 1
                if(c & self.GREEN2):
                    self.ds.setChannelName(i, (a + "_G2").encode())
                    i = i + 1
                if(c & self.BLUE):
                    self.ds.setChannelName(i, (a + "_B").encode())
                    i = i + 1


        
        #Check to see if the DataStack is big enough!
        if (self.ds.getNumChannels() < self.getReqMemChans(self.cols)):
            raise RuntimeError("Not enough channels in Data Stack")

        self.shutters.closeShutters(self.shutters.ALL)

    def getFrame(self, colours):
        """ Get a frame from the camera and extract the channels we want,
            putting them into ds. """
        if ('numpy_frames' in dir(self.cam)):
            getChanSlice = lambda ds,chan: CDataStack_AsArray(ds, chan)[:,:,ds.getZPos()]
        else:
            getChanSlice = lambda ds,chan: ds.getCurrentChannelSlice(chan)

        if(colours & self.BW):
            cs = getChanSlice(self.ds,self.curMemChn)
            self.cam.ExtractColor(cs,0)
            self.curMemChn = self.curMemChn + 1	
        if(colours & self.RED):
            self.cam.ExtractColor(getChanSlice(self.ds,self.curMemChn),1)
            self.curMemChn = self.curMemChn + 1
        if(colours & self.GREEN1):
            self.cam.ExtractColor(getChanSlice(self.ds,self.curMemChn),2)
            self.curMemChn = self.curMemChn + 1
        if(colours & self.GREEN2):
            self.cam.ExtractColor(getChanSlice(self.ds,self.curMemChn),4)
            self.curMemChn = self.curMemChn + 1
        if(colours & self.BLUE):
            self.cam.ExtractColor(getChanSlice(self.ds,self.curMemChn),3)
            self.curMemChn = self.curMemChn + 1

    def purge(self):
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
                self.needExposureStart = True
                #self.cam.StartExposure()


            if (self.looppos >= self.numHWChans):
                self.looppos = 0
                self.curMemChn = 0

                for a in self.WantFrameNotification:
                    a(self)

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

        t = 0
        for c in colours:
            if(c & self.BW): 
                t = t + 1
            if(c & self.RED):
                t = t + 1
            if(c & self.GREEN1):
                t = t + 1
            if(c & self.GREEN2):
                t = t + 1
            if(c & self.BLUE):
                t = t + 1

        return t


    def Notify(self):
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
                    if ('GetNumImsBuffered' in dir(self.cam)) and (nFrames > self.cam.GetBufferSize()/4):
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
    
                    for a in self.WantFrameGroupNotification:
                    	a(self)
            else:
                 self._stop()
        except:
            traceback.print_exc()
        finally:     
            self.inNotify = False

    def checkHardware(self):
        for callback in self.HardwareChecks:
            if not callback():
                print 'Waiting for hardware'
                return False

        return True

    def stop(self):
        "Stop sequence aquisition"

        wx.Timer.Stop(self)

        self.aqOn = False

        if 'StopAq' in dir(self.cam): #deal with Andor without breaking sensicam
            self.cam.StopAq()

        self.shutters.closeShutters(self.shutters.ALL)
        #self.cam.StopLifePreview()
        self.ds.setZPos(0)

        self.piezoGoHome()

        self.doStopLog()

        for a in self.WantStopNotification:
                a(self)

    def start(self, tiint = 100):
        "Start aquisition"

        self.looppos = 0
        self.ds.setZPos(0) #go to start of data stack
        
        #set the shutters up for the first frame
        self.shutters.setShutterStates(self.hwChans[self.looppos])

        #clear saturation intervened flag
        self.cam.saturationIntervened = False

        #move piezo to starting position
        self.setPiezoStartPos()

        self.doStartLog()
        
        for cb in self.WantStartNotification:
            cb(self)

        #self.Wait(1000)  # Warten, so dass Piezotisch wieder in Ruhe

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

        wx.Timer.Start(self,tiint)
        return True


    def Wait(self,iTime):
        """ Dirty delay routine - blocks until given no of milliseconds has elapsed\n 
            Probably best not to use with a delay of more than about a second or windows\n
            could rightly assume that the programme is <not responding> """
        time.sleep(iTime/1000)
        #FirstTime = time.clock()
        #dc = 0
        #while(time.clock() < (FirstTime + iTime/1000)):
        #    dc = dc + 1

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
