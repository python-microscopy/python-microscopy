import wx
import example
import time

class PreviewAquisator(wx.Timer)
    def __init__(_hwChans, _cols, _ds, _cam):
        wx.Timer.__init__()
        
        self.hwChans = _hwChans
        self.numHWChans = len(hwChans)
        self.cols =  _cols
        self.ds = _ds
        self.cam = _cam
        self.pView = _pView
	
        self.looppos=0
        self.curMemChn=0
	
        #Check to see if the DataStack is big enough!
        if (self.ds->getNumChannels() < getReqMemChans(len(self.hwChans), cols)):
            raise Exception, "Not enough channels in Data Stack"
	
        example.CShutterControl.closeShutters(example.CShutterControl.ALL)

    def getFrame(self, colours):
        """ Get a frame from the camera and extract the channels we want,
            putting them into ds. """

        if(colours & BW):
            self.cam.ExtractColor(self.ds.getCurrentChannelSlice(self.curMemChn),0)
            self.curMemChn = self.curMemChn + 1	
        elif(colours & RED):
            self.cam.ExtractColor(self.ds.getCurrentChannelSlice(self.curMemChn),1)
            self.curMemChn = self.curMemChn + 1
        elif(colours & GREEN1):
            self.cam.ExtractColor(self.ds.getCurrentChannelSlice(self.curMemChn),2)
            self.curMemChn = self.curMemChn + 1
        elif(colours & GREEN2):
            self.cam.ExtractColor(self.ds.getCurrentChannelSlice(self.curMemChn),4)
            self.curMemChn = self.curMemChn + 1
        if(colours & BLUE):
            self.cam.ExtractColor(ds.getCurrentChannelSlice(self.curMemChn),3)
            self.curMemChn = self.curMemChn + 1


    def onExpReady(void): 
        """ There is an exposure waiting in the Camera,
            looppos inticates which hardware (shutter) channel we're currently on """
	
        #If this was the last set of shutter combinations, move us to the positition
        # for the next slice.
        if (self.looppos == (self.numHWChans - 1)): 
            self.doPiezoStep()

        # Set the shutters for the next exposure
        example.CShutterControl.setShutterStates(self.hwChans[(self.looppos + 1)%self.numHWChans])
	
        # Pull the existing data from the camera
        self.getFrame(self.cols[self.looppos])

        self.looppos = self.looppos + 1
	
        if (self.looppos >= self.numHWChans):
            self.looppos = 0
            self.curMemChn = 0

            # If we're at the end of the Data Stack, then stop
            # Note that in normal sequence aquisition this is the line which determines how long to 
            # record for - in this class (ie the live preview) getNextDsSlice is defined such that 
            # it doesn't move through the stack, and always returns true, such that the aquisition 
            # continues for ever unless we stop it some other way (ie by clicking "Stop Live Preview")
            # in CRealAquisator it's overridden to behave in the right way.
            if (!self.getNextDsSlice()):
                 self.stop()

            #pView->UpdateView(6);



    def getReqMemChans(self, colours):
        """  Use this function to calc how may channels to allocate when creating a new data stack """

        t = 0
        for(c in colours):
            if(c & BW): 
                t = t + 1
            if(c & RED):
                t = t + 1
            if(c & GREEN1):
                t = t + 1
            if(c & GREEN2):
                t = t + 1
            if(c & BLUE):
                t = t + 1

        return t


    def onClock(self):
        "Should be called on each timer tick"
        
        if (self.aqOn): #check that we are aquiring
	
            if(!self.cam.CamReady() and !self.piezoReady())
                # Stop the aquisition if there is a hardware error
                self.stop()
                return
		
            #is there a picture waiting for us?
            #if so do the relevant processing
            #otherwise do nothing ...
            if(self.cam.ExpReady()):
                self.onExpReady()
         else:
             _stop()


    def stop(self):
        "Stop sequence aquisition"

        self.aqOn = false
	
        example.CShutterControl.closeShutters(example.CShutterControl.ALL)
        self.cam.StopLifePreview()
        self.ds.setZPos(0)

        self.piezoGoHome()

        self.doStopLog()

    def start(self):
        "Start aquisition"

        self.looppos = 0
        self.ds.setZPos(0) #go to start of data stack
        
        #set the shutters up for the first frame
        example.CShutterControl.setShutterStates(self.hwChans[self.looppos]) 

        #move piezo to starting position
        self.setPiezoStartPos()

        self.doStartLog()
	
        self.Wait(1000)  # Warten, so dass Piezotisch wieder in Ruhe
	
        iErr = self.cam.StartLifePreview()
	
        if (iErr < 0):
            stop()
            return false

        self.aqOn = true
        return true

    def Wait(self,iTime):
        """ Dirty delay routine - blocks until given no of milliseconds has elapsed\n 
            Probably best not to use with a delay of more than about a second or windows\n
            could rightly assume that the programme is <not responding> """
        FirstTime = time.clock()
        while(time.clock() < (FirstTime + iTime/1000)):
            pass

    def isRunning(self):
        return self.aqOn

    #place holders ... for overridden class which actually knows 
    #about the piezo
    def doPiezoStep(self):
        pass

    def piezoReady(self):
        return true

    def setPiezoStartPos(self):
        pass

    def getNextDsSlice(self):
        return true

    def _stop(self):
        self.stop()

    def doStartLog(self):
        pass

    def doStopLog(self):
        pass
