# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:42:16 2016

@author: david
"""


#import datetime
from PYME.Acquire import HDFSpooler
from PYME.Acquire import QueueSpooler, HTTPSpooler
# the following is now done by registering the suitable handler with
#   MetaDataHandler.provideStartMetadata from the init file when
#   loading the sampleinfo interface, see Acquire/Scripts/init.py
#try:
#     from PYME.Acquire import sampleInformationDjangoDirect as sampleInformation
#     sampInf = True
# except:
#     print('Could not connect to the sample information database')
#     sampInf = False
#import win32api
from PYME.IO.FileUtils import nameUtils
from PYME.IO.FileUtils.nameUtils import numToAlpha, getRelFilename, genHDFDataFilepath
#from PYME.IO.FileUtils.freeSpace import get_free_space


#import PYME.Acquire.Protocols
import PYME.Acquire.protocol as prot
from PYME.Acquire.ui import preflight

import os
import sys
#import glob

import subprocess

import dispatch

class SpoolController(object):
    def __init__(self, scope, defDir=genHDFDataFilepath(), defSeries='%(day)d_%(month)d_series'):
        """Initialise the spooling controller.
        
        Parameters
        ----------
        scope : microscope instance
            The currently active microscope class (see microscope.py)
        defDir : string pattern
            The default directory to save data to. Any keys of the form `%(<key>)` 
            will be substituted using the values defined in `PYME.fileUtils.nameUtils.dateDict` 
        defSeries : string pattern
            This specifies a pattern for file naming. Keys will be substituted as for `defDir`
            
        """
        self.scope = scope
        self.spoolType = 'Queue'
        
        #dtn = datetime.datetime.now()
        
        #dateDict = {'username' : win32api.GetUserName(), 'day' : dtn.day, 'month' : dtn.month, 'year':dtn.year}
        
        self.dirname = defDir % nameUtils.dateDict
        self.seriesStub = defSeries % nameUtils.dateDict

        self.seriesCounter = 0
        self.seriesName = self._GenSeriesName()

        self.protocol = prot.NullProtocol
        self.protocolZ = prot.NullZProtocol
        
        self.onSpoolProgress = dispatch.Signal()
        self.onSpoolStart = dispatch.Signal()
        self.onSpoolStop = dispatch.Signal()

        
        #if we've had to quit for whatever reason start where we left off
        #while os.path.exists(os.path.join(self.dirname, self.seriesName + '.h5')):
        while self._checkOutputExists(self.seriesName):
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
            
        
        

    def _GenSeriesName(self):
        return self.seriesStub + '_' + numToAlpha(self.seriesCounter)
       
    def _checkOutputExists(self, fn):
        if self.spoolType == 'HTTP':
            #special case for HTTP spooling
            return HTTPSpooler.exists(getRelFilename(self.dirname + fn + '.h5'))
            #return (fn + '.h5/') in HTTPSpooler.clusterIO.listdir(self.dirname)
        else:
            return (fn + '.h5') in os.listdir(self.dirname)
            
    def SetSpoolDir(self, dirname):
        """Set the directory we're spooling into"""
        self.dirname = dirname + os.sep

        #if we've had to quit for whatever reason start where we left off
        while self._checkOutputExists(self.seriesName):
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
            
    def _ProgressUpate(self, **kwargs):
        self.onSpoolProgress.send(self)


    def StartSpooling(self, fn=None, stack=False, compLevel = 2, zDwellTime = None, doPreflightCheck=True, maxFrames = sys.maxsize,
                      compressionSettings=HTTPSpooler.defaultCompSettings):
        """Start spooling
        """

        if fn in ['', None]: #sanity checking
            fn = self.seriesName
            #raise RuntimeError('No output file specified')
            #return #bail
        
        if not (self.spoolType == 'Cluster' or os.path.exists(self.dirname)):
            os.makedirs(self.dirname)

        if not self.dirname[-1] == os.sep:
            self.dirname += os.sep

        if self._checkOutputExists(fn): #check to see if data with the same name exists
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
            
            raise IOError('Output file already exists')
            return #bail
            

        if stack:
            protocol = self.protocolZ
            if not zDwellTime is None:
                protocol.dwellTime = zDwellTime
            print(protocol)
        else:
            protocol = self.protocol

        if doPreflightCheck and not preflight.ShowPreflightResults(None, self.protocol.PreflightCheck()):
            return #bail if we failed the pre flight check, and the user didn't choose to continue
            
          
        #fix timing when using fake camera
        if self.scope.cam.__class__.__name__ == 'FakeCamera':
            fakeCycleTime = self.scope.cam.GetIntegTime()
        else:
            fakeCycleTime = None
            
        frameShape = (self.scope.cam.GetPicWidth(), self.scope.cam.GetPicHeight())
        
        if self.spoolType == 'Queue':
            self.queueName = getRelFilename(self.dirname + fn + '.h5')
            self.spooler = QueueSpooler.Spooler(self.queueName, self.scope.frameWrangler.onFrame, 
                                                frameShape = frameShape, protocol=protocol, 
                                                guiUpdateCallback=self._ProgressUpate, complevel=compLevel, 
                                                fakeCamCycleTime=fakeCycleTime, maxFrames=maxFrames)
        elif self.spoolType == 'Cluster':
            #self.queueName = self.dirname + fn + '.h5'
            self.queueName = getRelFilename(self.dirname + fn + '.pcs')
            self.spooler = HTTPSpooler.Spooler(self.queueName, self.scope.frameWrangler.onFrame, 
                                               frameShape = frameShape, protocol=protocol, 
                                               guiUpdateCallback=self._ProgressUpate, complevel=compLevel, 
                                               fakeCamCycleTime=fakeCycleTime, maxFrames=maxFrames,
                                               compressionSettings=compressionSettings)
           
        else:
            self.spooler = HDFSpooler.Spooler(self.dirname + fn + '.h5', self.scope.frameWrangler.onFrame, 
                                              frameShape = frameShape, protocol=protocol, 
                                              guiUpdateCallback=self._ProgressUpate, complevel=compLevel, 
                                              fakeCamCycleTime=fakeCycleTime, maxFrames=maxFrames)

       
        # if sampInf and False: # with proper use of provideStartMetadata this should not be necessary?
        #     try:
        #         sampleInformation.getSampleData(self, self.spooler.md)
        #     except:
        #         #the connection to the database will timeout if not present
        #         #FIXME: catch the right exception (or delegate handling to sampleInformation module)
        #         pass
            
        self.spooler.onSpoolStop.connect(self.SpoolStopped)
        self.spooler.StartSpool()
        
        self.onSpoolStart.send(self)
        
        #return a function which can be called to indicate if we are done
        return lambda : not self.spooler.spoolOn

        

    def StopSpooling(self):
        """GUI callback to stop spooling."""
        self.spooler.StopSpool()
        
    def SpoolStopped(self, **kwargs):
        self.seriesCounter +=1
        self.seriesName = self._GenSeriesName()
        
        self.onSpoolStop.send(self)
        
    @property
    def autostart_analysis(self):
        if 'analysisSettings' in dir(self.scope):
            return self.scope.analysisSettings.propagateToAcquisisitonMetadata
        else:
            return False
        

    def LaunchAnalysis(self):
        """Launch analysis
        """
        
        dh5view_cmd = 'dh5view'
        if sys.platform == 'win32':
            dh5view_cmd = 'dh5view.cmd'
            
        if self.autostart_analysis:
            dh5view_cmd += ' -g'
        
        if isinstance(self.spooler, QueueSpooler.Spooler): #queue or not
            subprocess.Popen('%s -q %s QUEUE://%s' % (dh5view_cmd, self.spooler.tq.URI, self.queueName), shell=True)
        elif isinstance(self.spooler, HTTPSpooler.Spooler): #queue or not
            subprocess.Popen('%s %s' % (dh5view_cmd, self.spooler.getURL()), shell=True) 
        


    def SetProtocol(self, protocolName=None, reloadProtocol=True):
        """Set the current protocol .
        
        See also: PYME.Acquire.Protocols."""

        if (protocolName is None) or (protocolName == '<None>'):
            self.protocol = prot.NullProtocol
            self.protocolZ = prot.NullZProtocol
        else:
            #pmod = __import__('PYME.Acquire.Protocols.' + protocolName.split('.')[0],fromlist=['PYME', 'Acquire','Protocols'])
            
            #if reloadProtocol:
            #    reload(pmod) #force module to be reloaded so that changes in the protocol will be recognised
            pmod = prot.get_protocol(protocol_name=protocolName, reloadProtocol=reloadProtocol)

            self.protocol = pmod.PROTOCOL
            self.protocol.filename = protocolName
            
            self.protocolZ = pmod.PROTOCOL_STACK
            self.protocolZ.filename = protocolName
            
    def SetSpoolMethod(self, method):
        """Set the spooling method
        
        Parameters
        ----------
        
        method : string
            One of 'File', 'Queue', or 'HTTP'
        """
        self.spoolType = method
        
        #if we've had to quit for whatever reason start where we left off
        while self._checkOutputExists(self.seriesName):
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()

