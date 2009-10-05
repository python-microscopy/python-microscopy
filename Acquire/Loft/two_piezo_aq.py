#!/usr/bin/python

##################
# two_piezo_aq.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
import example
import time
import math

class TwoPiezoAquisator(wx.Timer):
    BW = 1
    RED = 2
    GREEN1 = 4
    GREEN2 = 8
    BLUE = 16
    PHASE = 1
    OBJECT = 0
    CENTRE_AND_LENGTH = 0
    START_AND_END = 1
    FORWARDS = 1
    BACKWARDS = -1

    def __init__(self, _chans, _cam, _piezos, _ds = None, _log = None):
        wx.Timer.__init__(self)

        self.chans = _chans        

        self.piezos = _piezos
        self.log = _log
        
        self.ScanChan  = [0,2]
        self.StartMode = [self.CENTRE_AND_LENGTH, self.CENTRE_AND_LENGTH]
        self.SeqLength = [30, 100]  
        self.StepSize  = [0.1, 0.1]
        self.startPos = [0,0]
        self.endPos = [0,0]
        self.prevPos = [0,0]
        
        self.direction = [self.FORWARDS, self.FORWARDS]

        
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
        
        #self.hwChans = self.chans.hw
        self.numHWChans = self.GetSeqLength(1)
        #self.cols =  self.chans.cols
   
        if (self.ds == None or keepds == False):
            self.ds = None
            self.ds = example.CDataStack(self.cam.GetPicWidth(), self.cam.GetPicHeight(), 
                self.GetSeqLength(0),self.GetSeqLength(1))

            i = 0
            for j in range(self.GetSeqLength(1)):
                self.ds.setChannelName(j,'%s' % (j,))

        
        #Check to see if the DataStack is big enough!
        #if (self.ds.getNumChannels() < self.getReqMemChans(self.cols)):
        #    raise Exception, "Not enough channels in Data Stack"

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
        if (self.looppos == (self.numHWChans - 1)): 
            self.doPiezoStep(0)
            self.piezoGoHome(1)
            self.Wait(15)
            self.setPiezoStartPos(1)
        
        # Set the shutters for the next exposure
        #example.CShutterControl.setShutterStates(self.hwChans[(self.looppos + 1)%self.numHWChans])
        #example.CShutterControl.setShutterStates(0)

        self.doPiezoStep(1)        

        #self.Wait(15) #give shutters a chance to close - should fix hardware

        self.looppos = self.looppos + 1
        
        #if ('itimes' in dir(self.chans)): #maintain compatibility with old versions
        #    self.cam.SetIntegTime(self.chans.itimes[self.looppos%self.numHWChans])
        #    self.cam.SetCOC()
        
        #example.CShutterControl.setShutterStates(self.hwChans[(self.looppos)%self.numHWChans])
        #self.Wait(15)

        self.cam.StartExposure()

        # Pull the existing data from the camera
        self.getFrame(self.BW)

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
        
	
            if(not (self.cam.CamReady() and self.piezoReady(0) and self.piezoReady(1))):
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

        #example.CShutterControl.closeShutters(example.CShutterControl.ALL)
        #self.cam.StopLifePreview()
        self.ds.setZPos(0)

        self.piezoGoHome(0)
        self.piezoGoHome(1)

        #self.doStopLog()

        for a in self.WantStopNotification:
                a(self)

    def start(self, tiint = 20):
        "Start aquisition"

        self.looppos = 0
        self.ds.setZPos(0) #go to start of data stack
        
        #set the shutters up for the first frame
        #example.CShutterControl.setShutterStates(self.hwChans[self.looppos]) 

        #move piezo to starting position
        self.setPiezoStartPos(0)
        self.setPiezoStartPos(1)

        #self.doStartLog()

        self.Wait(1000)  # Warten, so dass Piezotisch wieder in Ruhe

        #if ('itimes' in dir(self.chans)): #maintain compatibility with old versions
        #    self.cam.SetIntegTime(self.chans.itimes[self.looppos])
        #    self.cam.SetCOC()

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
    #def doPiezoStep(self):
    #    pass

    #def piezoReady(self):
    #    return True

    #def setPiezoStartPos(self):
    #    pass

    #def getNextDsSlice(self):
    #    return True

    #def _stop(self):
    #    self.stop()

    #def doStartLog(self):
    #    pass

    #def doStopLog(self):
    #    pass

    #def GetSeqLength(self):
    #    return 1

    #def piezoGoHome(self):
    #    pass


    def doPiezoStep(self, pz):
        if(self.GetStepSize(pz) > 0):
            curr_pos = self.piezos[self.GetScanChannel(pz)][0].GetPos(self.piezos[self.GetScanChannel(pz)][1]);
            print curr_pos
            self.piezos[self.GetScanChannel(pz)][0].MoveTo(self.piezos[self.GetScanChannel(pz)][1], curr_pos + 
                self.GetDirection(pz)*self.GetStepSize(pz), False)


    def piezoReady(self, pz):
        return not ((self.GetStepSize(pz) > 0) and
            (self.piezos[self.GetScanChannel(pz)][0].GetControlReady() == False))


    def setPiezoStartPos(self, pz):
        " Used internally to move the piezo at the beginning of the aquisition "
        #self.curpos = self.piezos[self.GetScanChannel()][0].GetPos(self.piezos[self.GetScanChannel()][1])

        self.SetPrevPos(pz,self._CurPos(pz)) # aktuelle Position der Piezotische merken

        self.piezos[self.GetScanChannel(pz)][0].MoveTo(self.piezos[self.GetScanChannel(pz)][1], self.GetStartPos(pz), False)
            

    def getNextDsSlice(self):
        return self.ds.nextZ()

    def GetScanChannel(self, pz):
        return self.ScanChan[pz]

    def piezoGoHome(self, pz):
        self.piezos[self.GetScanChannel(pz)][0].MoveTo(self.piezos[self.GetScanChannel(pz)][1],self.GetPrevPos(pz),False)

    def SetScanChannel(self, pz,iMode):
        self.ScanChan[pz] = iMode

    def SetSeqLength(self, pz,iLength):
        self.SeqLength[pz] = iLength

    def GetSeqLength(self, pz):
        if (self.StartMode[pz] == 0):
            return self.SeqLength[pz]
        else:
            return int(math.ceil(abs(self.GetEndPos(pz) - self.GetStartPos(pz))/self.GetStepSize(pz)))

    def SetStartMode(self, pz, iMode):
        self.StartMode[pz] = iMode

    def GetStartMode(self, pz):
        return self.StartMode[pz]

    def SetStepSize(self, pz, fSize):
        self.StepSize[pz] = fSize

    def GetStepSize(self, pz):
        return self.StepSize[pz]

    def SetStartPos(self, pz, sPos):
        self.startPos[pz] = sPos

    def GetStartPos(self, pz):
        if (self.GetStartMode(pz) == 0):
            return self._CurPos(pz) - (self.GetStepSize(pz)*self.GetSeqLength(pz)*self.GetDirection(pz)/2)
        else: 
            if not ("startPos" in dir(self)):
                raise Exception, "Please call SetStartPos first !!"
            return self.startPos[pz]

    def SetEndPos(self, pz,  ePos):
        self.endPos[pz] = ePos

    def GetEndPos(self, pz):
        if (self.GetStartMode(pz) == 0):
            return self._CurPos(pz) + (self.GetStepSize(pz)*self.GetSeqLength(pz)*self.GetDirection(pz)/2)
        else: 
            if not ("endPos" in dir(self)):
                raise Exception, "Please call SetEndPos first !!"
            return self.endPos[pz]

    def SetPrevPos(self, pz, sPos):
        self.prevPos[pz] = sPos

    def GetPrevPos(self, pz):
        if not ("prevPos" in dir(self)):
            raise Exception, "Please call SetPrevPos first !!"
        return self.prevPos[pz]

    def SetDirection(self, pz,  dir):
        " Fowards = 1, backwards = -1 "
        self.direction[pz] = dir

    def GetDirection(self, pz):
        if (self.GetStartMode(pz) == 0):
            if (self.direction[pz] > 0.1):
                return 1
            else:
                return -1
        else:
            if ((self.GetEndPos(pz) - self.GetStartPos(pz)) > 0.1):
                return 1
            else:
                return -1
            
    def _CurPos(self, pz):
        return self.piezos[self.GetScanChannel(pz)][0].GetPos(self.piezos[self.GetScanChannel(pz)][1])
            
    def Verify(self):
        if (self.GetStartPos() < self.piezos[self.GetScanChannel()][0].GetMin(self.piezos[self.GetScanChannel()][1])):
            return (False, 'StartPos', 'StartPos is smaller than piezo minimum',self.piezos[self.GetScanChannel()][0].GetMin(self.piezos[self.GetScanChannel()][1])) 
        
        if (self.GetStartPos() > self.piezos[self.GetScanChannel()][0].GetMax(self.piezos[self.GetScanChannel()][1])):
            return (False, 'StartPos', 'StartPos is larger than piezo maximum',self.piezos[self.GetScanChannel()][0].GetMax(self.piezos[self.GetScanChannel()][1])) 

        if (self.GetEndPos() < self.piezos[self.GetScanChannel()][0].GetMin(self.piezos[self.GetScanChannel()][1])):
            return (False, 'EndPos', 'EndPos is smaller than piezo minimum',self.piezos[self.GetScanChannel()][0].GetMin(self.piezos[self.GetScanChannel()][1])) 
        
        if (self.GetEndPos() > self.piezos[self.GetScanChannel()][0].GetMax(self.piezos[self.GetScanChannel()][1])):
            return (False, 'EndPos', 'EndPos is larger than piezo maximum',self.piezos[self.GetScanChannel()][0].GetMax(self.piezos[self.GetScanChannel()][1])) 
        
        if (self.GetEndPos() < self.GetStartPos()):
            return (False, 'EndPos', 'EndPos is before Startpos',self.GetStartPos()) 

        #stepsize limits are at present arbitrary == not really restricted to sensible values
        if (self.GetStepSize() < 0.001):
            return (False, 'StepSize', 'StepSize is smaller than piezo minimum',0.001) 
        
        if (self.GetStartPos() > 50):
            return (False, 'StepSize', 'StepSize is larger than piezo maximum',100)
        
        return (True,) 

    def doStartLog(self):
        if not 'GENERAL' in self.log.keys():
            self.log['GENERAL'] = {}
        if not 'PIEZOS' in self.log.keys():
            self.log['PIEZOS'] = {}
        if not 'CAMERA' in self.log.keys():
            self.log['CAMERA'] = {}
            
        self.log['PIEZOS']['Piezo'] = self.piezos[self.GetScanChannel()][2]
        self.log['PIEZOS']['Stepsize'] = self.GetStepSize()
        self.log['PIEZOS']['StartPos'] = self.GetStartPos()
        for pz in self.piezos:
            self.log['PIEZOS']['%s_Pos' % pz[2]] = pz[0].GetPos(pz[1])
            
        #self.log['STEPPER']['XPos'] = m_pDoc->Step.GetNullX(),
        #m_pDoc->Step.GetPosX(), m_pDoc->Step.GetNullY(), m_pDoc->Step.GetPosY(),
        #m_pDoc->Step.GetNullZ(), m_pDoc->Step.GetPosZ());
        self.cam.GetStatus()
        self.log['CAMERA']['Binning'] = self.cam.GetHorizBin()
        #m_pDoc->LogData.SetIntegTime(m_pDoc->Camera.GetIntegTime());
        self.log['GENERAL']['Width'] = self.cam.GetPicWidth()
        self.log['GENERAL']['Height'] = self.cam.GetPicHeight()
        self.log['CAMERA']['ROIPosX'] = self.cam.GetROIX1()
        self.log['CAMERA']['ROIPosY'] = self.cam.GetROIY1()
        self.log['CAMERA']['ROIWidth'] = self.cam.GetROIX2() - self.cam.GetROIX1()
        self.log['CAMERA']['ROIHeight'] = self.cam.GetROIY2() - self.cam.GetROIY1()
        self.log['CAMERA']['StartCCDTemp'] = self.cam.GetCCDTemp()
        self.log['CAMERA']['StartElectrTemp'] = self.cam.GetElectrTemp()
        
        self.log['GENERAL']['NumChannels'] = self.ds.getNumChannels()
        self.log['GENERAL']['NumHWChans'] = self.numHWChans
        
        for ind in range(self.numHWChans):
            self.log['SHUTTER_%d' % ind] = {}
            self.log['SHUTTER_%d' % ind]['Name'] = self.chans.names[ind]
            self.log['SHUTTER_%d' % ind]['IntegrationTime'] = self.chans.itimes[ind]
            self.log['SHUTTER_%d' % ind]['Mask'] = self.hwChans[ind]
            
            s = ''
            bef = 0
            if (self.cols[ind] & self.BW):
                s = s + 'BW'
                bef = 1
            if (self.cols[ind] & self.RED):
                if bef:
                    s = s + ' '
                s = s + 'R'
                bef = 1
            if (self.cols[ind] & self.GREEN1):
                if bef:
                    s = s + ' '
                s = s + 'G1'
                bef = 1
            if (self.cols[ind] & self.GREEN2):
                if bef:
                    s = s + ' '
                s = s + 'G2'
                bef = 1
            if (self.cols[ind] & self.BLUE):
                if bef:
                    s = s + ' '
                s = s + 'B'
                #bef = 1
            self.log['SHUTTER_%d' % ind]['Colours'] = s
        
        dt = datetime.datetime.now()
        
        self.log['GENERAL']['Date'] = '%d/%d/%d' % (dt.day, dt.month, dt.year)
        self.log['GENERAL']['StartTime'] = '%d:%d:%d' % (dt.hour, dt.minute, dt.second)

        #m_pDoc->LogData.SaveSeqROIMode(m_pDoc->Camera.GetROIMode());
        #pass

    def doStopLog(self):
        self.log['GENERAL']['Depth'] = self.ds.getDepth()
        self.log['PIEZOS']['EndPos'] = self.GetEndPos()
        self.cam.GetStatus()
        self.log['CAMERA']['EndCCDTemp'] = self.cam.GetCCDTemp()
        self.log['CAMERA']['EndElectrTemp'] = self.cam.GetElectrTemp()
        
        dt = datetime.datetime.now()
        self.log['GENERAL']['EndTime'] = '%d:%d:%d' % (dt.hour, dt.minute, dt.second)

  
       