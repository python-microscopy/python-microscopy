import wx
import example
import time

class PreviewAquisator(wx.Timer):
    BW = 1
    RED = 2
    GREEN1 = 4
    GREEN2 = 8
    BLUE = 16

    def __init__(self, _chans, _cam, _ds = None):
        wx.Timer.__init__(self)
        
        self.chans = _chans
        #self.hwChans = _chans.hw
        #self.numHWChans = len(_chans.hw)
        #self.cols =  _chans.cols
        self.ds = _ds
        self.cam = _cam
        self.loopnuf = 0
        self.aqOn = False
        #lists of functions to call on a new frame, and when the aquisition ends
        self.WantFrameNotification = []
        self.WantStopNotification = []

    def Prepare(self, keepds=False):
        self.looppos=0
        self.curMemChn=0
        
        self.hwChans = self.chans.hw
        self.numHWChans = len(self.chans.hw)
        self.cols =  self.chans.cols
   
        if (self.ds == None or keepds == False):
            self.ds = None
            self.ds = example.CDataStack(self.cam.GetPicWidth(), self.cam.GetPicHeight(), 
                self.GetSeqLength(),self.getReqMemChans(self.cols))

            i = 0
            for j in range(len(self.cols)):
                a = self.chans.names[j]
                c = self.chans.cols[j]
                #for (a,c) in (self.chans.names, self.chans.cols):

                if(c & self.BW):
                    self.ds.setChannelName(i, a + "_BW")
                    i = i + 1
                if(c & self.RED):
                    self.ds.setChannelName(i, a + "_R")
                    i = i + 1
                if(c & self.GREEN1):
                    self.ds.setChannelName(i, a + "_G1")
                    i = i + 1
                if(c & self.GREEN2):
                    self.ds.setChannelName(i, a + "_G2")
                    i = i + 1
                if(c & self.BLUE):
                    self.ds.setChannelName(i, a + "_B")
                    i = i + 1


        
        #Check to see if the DataStack is big enough!
        if (self.ds.getNumChannels() < self.getReqMemChans(self.cols)):
            raise Exception, "Not enough channels in Data Stack"

        #example.CShutterControl.closeShutters(example.CShutterControl.ALL)

    def getFrame(self, colours):
        """ Get a frame from the camera and extract the channels we want,
            putting them into ds. """

        if(colours & self.BW):
            self.cam.ExtractColor(self.ds.getCurrentChannelSlice(self.curMemChn),0)
            self.curMemChn = self.curMemChn + 1	
        if(colours & self.RED):
            self.cam.ExtractColor(self.ds.getCurrentChannelSlice(self.curMemChn),1)
            self.curMemChn = self.curMemChn + 1
        if(colours & self.GREEN1):
            self.cam.ExtractColor(self.ds.getCurrentChannelSlice(self.curMemChn),2)
            self.curMemChn = self.curMemChn + 1
        if(colours & self.GREEN2):
            self.cam.ExtractColor(self.ds.getCurrentChannelSlice(self.curMemChn),4)
            self.curMemChn = self.curMemChn + 1
        if(colours & self.BLUE):
            self.cam.ExtractColor(self.ds.getCurrentChannelSlice(self.curMemChn),3)
            self.curMemChn = self.curMemChn + 1


    def onExpReady(self): 
        """ There is an exposure waiting in the Camera,
            looppos inticates which hardware (shutter) channel we're currently on """
        
        self.loopnuf = self.loopnuf + 1

        #If this was the last set of shutter combinations, move us to the position
        # for the next slice.
        #if (self.looppos == (self.numHWChans - 1)): 
        #    self.doPiezoStep()
        
        # Set the shutters for the next exposure
        #example.CShutterControl.setShutterStates(self.hwChans[(self.looppos + 1)%self.numHWChans])
        #example.CShutterControl.setShutterStates(0)

        #self.Wait(15) #give shutters a chance to close - should fix hardware

        self.looppos = self.looppos + 1
        
        if ('itimes' in dir(self.chans)): #maintain compatibility with old versions
            self.cam.SetIntegTime(self.chans.itimes[self.looppos%self.numHWChans])
            self.cam.SetCOC()
        
        #example.CShutterControl.setShutterStates(self.hwChans[(self.looppos)%self.numHWChans])
        #self.Wait(15)

        self.cam.StartExposure()

        # Pull the existing data from the camera
        self.getFrame(self.cols[self.looppos-1])

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
        "Should be called on each timer tick"

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
            if(self.cam.ExpReady()):
                self.onExpReady()
        else:
             self._stop()

    def stop(self):
        "Stop sequence aquisition"

        wx.Timer.Stop(self)

        self.aqOn = False

        example.CShutterControl.closeShutters(example.CShutterControl.ALL)
        #self.cam.StopLifePreview()
        self.ds.setZPos(0)

        #self.piezoGoHome()

        self.doStopLog()

        for a in self.WantStopNotification:
                a(self)

    def start(self, tiint = 20):
        "Start aquisition"

        self.looppos = 0
        self.ds.setZPos(0) #go to start of data stack
        
        #set the shutters up for the first frame
        example.CShutterControl.setShutterStates(self.hwChans[self.looppos]) 

        #move piezo to starting position
        #self.setPiezoStartPos()

        self.doStartLog()

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
