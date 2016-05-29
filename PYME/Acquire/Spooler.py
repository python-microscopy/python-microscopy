#!/usr/bin/python

##################
# Spooler.py
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

import os
#import logparser
import datetime

from PYME.io import MetaDataHandler


try:
    from PYME.Acquire import sampleInformation
except:
    sampleInformation= None

import time

global timeFcn
timeFcn = time.time

import dispatch

from PYME.Acquire import eventLog
from PYME.Acquire import protocol as p

class EventLogger:
    '''Event logging backend base class'''
    def __init__(self, scope, hdf5File):
        self.scope = scope      

    def logEvent(self, eventName, eventDescr = ''):
        '''Log an event. Should be overriden in derived classes.
        
        .. note:: In addition to the name and description, timing information is recorded
        for each event.
          
        Parameters
        ----------
        eventName : string
            short event name - < 32 chars and should be shared by events of the
            same type.
        eventDescr : string
            description of the event - additional, even specific information
            packaged as a string (<255 chars). This is commonly used to store 
            parameters - e.g. z positions, and should be both human readable and 
            easily parsed.
        
        '''
        pass


class Spooler:
    '''Spooler base class'''
    def __init__(self, filename, frameSource, protocol = p.NullProtocol, 
                 guiUpdateCallback=None, fakeCamCycleTime=None, maxFrames = p.maxint, **kwargs):
        '''Create a new spooler.
        
        Parameters
        ----------
        scope : PYME.Acquire.microscope.microscope object
            The microscope providing the data
        filename : string
            The file into which to spool
        frameSource : dispatch.Signal object
            A source of frames we can subscribe to. It should implement a "connect"
            method allowing us to register a callback and then call the callback with
            the frame data in a "frameData" kwarg.
        protocol : PYME.Acquire.protocol.TaskListProtocol object
            The acquisition protocol
        guiUpdateCallback : function
            a function to call when the spooling GUI needs updating
            
        '''
        global timeFcn
        #self.scope = scope
        self.filename=filename
        self.frameSource = frameSource
        self.guiUpdateCallback = guiUpdateCallback
        self.protocol = protocol
        
        self.maxFrames = maxFrames
        
        self.onSpoolStop = dispatch.Signal()
    
        #if we've got a fake camera - the cycle time will be wrong - fake our time sig to make up for this
        #if scope.cam.__class__.__name__ == 'FakeCamera':
        #    timeFcn = self.fakeTime
            
        if not fakeCamCycleTime == None:
            self.fakeCamCycleTime = fakeCamCycleTime
            timeFcn = self.fakeTime

       

    def StartSpool(self):
        self.watchingFrames = True
        eventLog.WantEventNotification.append(self.evtLogger)

        self.imNum = 0
   
        self.doStartLog()

        self.protocol.Init(self)
   
        self.frameSource.connect(self.OnFrame)
        self.spoolOn = True
       
    def StopSpool(self):
        #try:
        self.frameSource.disconnect(self.OnFrame)
        
        #there is a race condition on disconnect - ignore any additional frames
        self.watchingFrames = False 
        
        #except:
        #    pass

        try:
            self.protocol.OnFinish()#this may still cause events
            self.FlushBuffer()
            self.doStopLog()
        except:
            import traceback
            traceback.print_exc()
            
        try:
            eventLog.WantEventNotification.remove(self.evtLogger)
        except ValueError:
            pass
        
        self.spoolOn = False
        
        self.onSpoolStop.send(self)

    def OnFrame(self, **kwargs):
        '''Callback which should be called on every frame'''
        if not self.watchingFrames:
            #we have allready disconnected - ignore any new frames
            return
            
        self.imNum += 1
        if not self.guiUpdateCallback is None:
            self.guiUpdateCallback()
            
        self.protocol.OnFrame(self.imNum)

        if self.imNum == 2 and sampleInformation and sampleInformation.currentSlide[0]: #have first frame and should thus have an imageID
            sampleInformation.createImage(self.md, sampleInformation.currentSlide[0])
            
        if self.imNum >= self.maxFrames:
            self.StopSpool()
            

    def doStartLog(self):
        '''Record pertinant information to metadata at start of acquisition.
        
        Loops through all registered sources of start metadata and adds their entries.
        
        See Also
        --------
        PYME.io.MetaDataHandler
        '''
        dt = datetime.datetime.now()
        
        self.dtStart = dt
        
        self.tStart = time.time()
          
        mdt = MetaDataHandler.NestedClassMDHandler()
          
        mdt.setEntry('StartTime', self.tStart)

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(mdt)
            
        self.md.copyEntriesFrom(mdt)
       

    def doStopLog(self):
        '''Record information to metadata at end of acquisition'''
        self.md.setEntry('EndTime', time.time())
        
        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStopMetadata:
           mdgen(self.md)

    def fakeTime(self):
        '''Generate a fake timestamp for use with the simulator where the camera 
        cycle time does not match the actual time elapsed to generate the frame'''
        #return self.tStart + self.imNum*self.scope.cam.GetIntegTime()
        return self.tStart + self.imNum*self.fakeCamCycleTime

    def FlushBuffer(self):
        pass
        
        
    def __del__(self):
        if self.spoolOn:
            self.StopSpool()
