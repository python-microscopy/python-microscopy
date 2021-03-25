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

from PYME.IO import MetaDataHandler

try:
    from PYME.Acquire import sampleInformation
except:
    sampleInformation= None

import time

global timeFcn
timeFcn = time.time

from PYME.contrib import dispatch
import uuid

from PYME.Acquire import eventLog
from PYME.Acquire import protocol as p

import logging
logger = logging.getLogger(__name__)

class EventLogger:
    """Event logging backend base class"""
    def __init__(self, scope, hdf5File):
        self.scope = scope      

    def logEvent(self, eventName, eventDescr = '', timestamp=None):
        """Log an event. Should be overriden in derived classes.
        
        .. note::
        
          In addition to the name and description, timing information is recorded for each event.
          
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
        
        """
        pass


class Spooler:
    """Spooler base class"""
    def __init__(self, filename, frameSource, protocol = p.NullProtocol, 
                 guiUpdateCallback=None, fakeCamCycleTime=None, maxFrames = p.maxint, **kwargs):
        """Create a new spooler.
        
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
            
        """
        global timeFcn
        #self.scope = scope
        self.filename=filename
        self.frameSource = frameSource
        self.guiUpdateCallback = guiUpdateCallback
        self.protocol = protocol
        
        self.maxFrames = maxFrames
        
        stack_settings = kwargs.get('stack_settings', None)
        if stack_settings:
            # only record stack settings if provided (letting protocol fall through to global stack settings,
            # if not provided / None)
            self.stack_settings = stack_settings
        
        self.onSpoolStop = dispatch.Signal()
    
        #if we've got a fake camera - the cycle time will be wrong - fake our time sig to make up for this
        #if scope.cam.__class__.__name__ == 'FakeCamera':
        #    timeFcn = self.fakeTime

        self._last_gui_update = 0
        self.spoolOn = False
        self.imNum = 0
        
        self.spool_complete = False
        
        self._spooler_uuid = uuid.uuid4()
            
        if not fakeCamCycleTime is None:
            self.fakeCamCycleTime = fakeCamCycleTime
            timeFcn = self.fakeTime

       

    def StartSpool(self):
        """ Perform protocol 'frame -1' tasks, log start metadata, then connect
        to the frame source.
        """
        self.watchingFrames = True
        eventLog.WantEventNotification.append(self.evtLogger)

        self.imNum = 0
        
        # set tStart here for simulator so that events in init phase get time stamps. Real start time is set below
        # **after** protocol.Init() call
        self.tStart = time.time()

        self.protocol.Init(self)
        
        # record start time when we start receiving frames.
        self.tStart = time.time()
        self._collect_start_metadata()
        self.frameSource.connect(self.OnFrame, dispatch_uid=self._spooler_uuid)
        
        self.spoolOn = True
       
    def StopSpool(self):
        #try:
        logger.debug('Disconnecting from frame source')
        self.frameSource.disconnect(self.OnFrame, dispatch_uid=self._spooler_uuid)
        logger.debug('Frame source should be disconnected')
        
        #there is a race condition on disconnect - ignore any additional frames
        self.watchingFrames = False 
        
        #except:
        #    pass

        try:
            self.protocol.OnFinish()#this may still cause events
            self.FlushBuffer()
            self._collect_stop_metadata()
        except:
            import traceback
            traceback.print_exc()
            
        try:
            eventLog.WantEventNotification.remove(self.evtLogger)
        except ValueError:
            pass
        
        self.spoolOn = False
        if not self.guiUpdateCallback is None:
            self.guiUpdateCallback()
        
        self.finalise()
        self.onSpoolStop.send(self)
        self.spool_complete = True
        
    def finalise(self):
        """
        Over-ride in derived classes to do any spooler specific tidy up - e.g. sending events to server

        """
        pass
        
    def abort(self):
        """
        Tidy up if something goes horribly wrong. Disconnects frame source and event logger  and then calls cleanup()

        """
        #there is a race condition on disconnect - ignore any additional frames
        self.watchingFrames = False
        
        try:
            logger.debug('Disconnecting from frame source')
            self.frameSource.disconnect(self.OnFrame, dispatch_uid=self._spooler_uuid)
            logger.debug('Frame source should be disconnected')
        except:
            logger.exception('Error disconnecting frame source')


        try:
            eventLog.WantEventNotification.remove(self.evtLogger)
        except ValueError:
            pass

        self.spoolOn = False
        self.onSpoolStop.send(self)
        

    def OnFrame(self, **kwargs):
        """Callback which should be called on every frame"""
        if not self.watchingFrames:
            #we have allready disconnected - ignore any new frames
            return

        t = time.time()
            
        self.imNum += 1
        if not self.guiUpdateCallback is None:
            if (t > (self._last_gui_update +.1)):
                self._last_gui_update = t
                self.guiUpdateCallback()
            
        try:
            import wx #FIXME - shouldn't do this here
            wx.CallAfter(self.protocol.OnFrame, self.imNum)
            #FIXME - The GUI logic shouldn't be here (really needs to change at the level of the protocol and/or general structure of PYMEAcquire
        except (ImportError, AssertionError):  # handle if spooler doesn't have a GUI
            self.protocol.OnFrame(self.imNum) #FIXME - This will most likely fail for anything but a NullProtocol

        if self.imNum == 2 and sampleInformation and sampleInformation.currentSlide[0]: #have first frame and should thus have an imageID
            sampleInformation.createImage(self.md, sampleInformation.currentSlide[0])
            
        if self.imNum >= self.maxFrames:
            self.StopSpool()
            

    def _collect_start_metadata(self):
        """Record pertinant information to metadata at start of acquisition.
        
        Loops through all registered sources of start metadata and adds their entries.
        
        See Also
        --------
        PYME.IO.MetaDataHandler
        """
        dt = datetime.datetime.now()
        
        self.dtStart = dt
        
        #self.tStart = time.time()
        
        # create an in-memory metadata handler and populate this prior to copying data over to the spooler
        # metadata handler. This significantly improves performance if the spooler metadata handler has high latency
        # (as is the case for both the HDFMetaDataHandler and, especially, the QueueMetaDataHandler).
        mdt = MetaDataHandler.NestedClassMDHandler()
        mdt.setEntry('StartTime', self.tStart)

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(mdt)
            
        self.md.copyEntriesFrom(mdt)
       

    def _collect_stop_metadata(self):
        """Record information to metadata at end of acquisition"""
        self.md.setEntry('EndTime', time.time())
        
        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStopMetadata:
           mdgen(self.md)

    def fakeTime(self):
        """Generate a fake timestamp for use with the simulator where the camera
        cycle time does not match the actual time elapsed to generate the frame"""
        #return self.tStart + self.imNum*self.scope.cam.GetIntegTime()
        return self.tStart + self.imNum*self.fakeCamCycleTime

    def FlushBuffer(self):
        pass
    
    def status(self):
        return {'spooling' : self.spoolOn,
                'frames_spooled' : self.imNum}
    
    def cleanup(self):
        """ over-ride to do any cleanup"""
        pass
    
    def finished(self):
        """ over-ride in derived classes to indicate when buffers flushed"""
        return True
    
    def get_n_frames(self):
        return self.imNum
        
    def __del__(self):
        if self.spoolOn:
            self.StopSpool()
