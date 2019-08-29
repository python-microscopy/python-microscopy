# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:42:16 2016

@author: david
"""


#import datetime
#from PYME.Acquire import HDFSpooler, QueueSpooler
from PYME.Acquire import HTTPSpooler
# TODO: change to use a metadata handler / provideStartMetadata hook
#   MetaDataHandler.provideStartMetadata from the init file when
#   loading the sampleinfo interface, see Acquire/Scripts/init.py
try:
    from PYME.Acquire import sampleInformation
    sampInf = True
except:
    print('Could not connect to the sample information database')
    sampInf = False
#import win32api
from PYME.IO.FileUtils import nameUtils
from PYME.IO.FileUtils.nameUtils import numToAlpha, getRelFilename, genHDFDataFilepath
#from PYME.IO.FileUtils.freeSpace import get_free_space


#import PYME.Acquire.Protocols
import PYME.Acquire.protocol as prot
from PYME.Acquire.ui import preflight
from PYME import config

import os
import sys
#import glob

import subprocess

import dispatch

import logging
logger = logging.getLogger(__name__)

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
        
        self._user_dir = None
        self._base_dir = nameUtils.get_local_data_directory()
        self._subdir = nameUtils.get_spool_subdir()
        self.seriesStub = defSeries % nameUtils.dateDict

        self.seriesCounter = 0
        self._series_name = None

        self.protocol = prot.NullProtocol
        self.protocolZ = prot.NullZProtocol
        
        self.onSpoolProgress = dispatch.Signal()
        self.onSpoolStart = dispatch.Signal()
        self.onSpoolStop = dispatch.Signal()
        
    @property
    def _sep(self):
        if self.spoolType == 'Cluster':
            return '/'
        else:
            return os.sep
        
    @property
    def dirname(self):
        if not self._user_dir is None:
            return self._user_dir
        
        if self.spoolType == 'Cluster':
            return '/'.join(self._subdir)
        else:
            return os.sep.join([self._base_dir, ] + self._subdir)
        
    @property
    def seriesName(self):
        # make this a property so that we can defer evaluation to first use
        # this lets us set 'acquire-spool_subdirectories' in the init.py for a specific microscope
        if self._series_name is None:
            #if we've had to quit for whatever reason start where we left off
            #while os.path.exists(os.path.join(self.dirname, self.seriesName + '.h5')):
            self._series_name = self._GenSeriesName()
            self._update_series_counter()
        
        return self._series_name
    
    @seriesName.setter
    def seriesName(self, val):
        self._series_name = val
    

    def _GenSeriesName(self):
        if config.get('acquire-spool_subdirectories', False):
            # High-throughput performance optimization
            # If true, add a layer of directories to limit the number of series saved in a single directory
            return '%03d%s%s_%05d' % (int(self.seriesCounter/100), self._sep, self.seriesStub, self.seriesCounter)
        else:
            return self.seriesStub + '_' + numToAlpha(self.seriesCounter)
       
    def _checkOutputExists(self, fn):
        if self.spoolType == 'Cluster':
            from PYME.Acquire import HTTPSpooler
            # special case for HTTP spooling.  Make sure 000\series.pcs -> 000/series.pcs
            pyme_cluster = self.dirname + '/' + fn.replace('\\', '/')
            logger.debug('Looking for %s (.pcs or .h5) on cluster' % pyme_cluster)
            return HTTPSpooler.exists(pyme_cluster + '.pcs') or HTTPSpooler.exists(pyme_cluster + '.h5')
            #return (fn + '.h5/') in HTTPSpooler.clusterIO.listdir(self.dirname)
        else:
            local_h5 = os.sep.join([self.dirname, fn + '.h5'])
            logger.debug('Looking for %s on local machine' % local_h5)
            return os.path.exists(local_h5)
        
    def get_free_space(self):
        """
        Get available space in the target spool directory
        
        Returns
        -------
        
        free space in GB

        """
        if self.spoolType == 'Cluster':
            logger.warn('Cluster free space calculation not yet implemented, using fake value')
            return 100
        else:
            from PYME.IO.FileUtils.freeSpace import get_free_space
            return get_free_space(self.dirname)/1e9
        
    def _update_series_counter(self):
        logger.debug('Updating series counter')
        while self._checkOutputExists(self.seriesName):
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
            
    def SetSpoolDir(self, dirname):
        """Set the directory we're spooling into"""
        self._dirname = dirname + os.sep
        #if we've had to quit for whatever reason start where we left off
        self._update_series_counter()
            
    def _ProgressUpate(self, **kwargs):
        self.onSpoolProgress.send(self)
        
    def _get_queue_name(self, fn, pcs=False):
        if pcs:
            ext = '.pcs'
        else:
            ext = '.h5'
            
        return self._sep.join([self.dirname.rstrip(self._sep), fn + ext])


    def StartSpooling(self, fn=None, stack=False, compLevel = 2, zDwellTime = None, doPreflightCheck=True, maxFrames = sys.maxsize,
                      compressionSettings=HTTPSpooler.defaultCompSettings, cluster_h5 = False):
        """Start spooling
        """

        if fn in ['', None]:
            fn = self.seriesName
        
        #make directories as needed
        if not (self.spoolType == 'Cluster'):
            dirname = os.path.split(self._get_queue_name(fn))[0]
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        if self._checkOutputExists(fn): #check to see if data with the same name exists
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
            
            raise IOError('Output file already exists')

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
            from PYME.Acquire import QueueSpooler
            self.queueName = getRelFilename(self._get_queue_name(fn))
            self.spooler = QueueSpooler.Spooler(self.queueName, self.scope.frameWrangler.onFrame, 
                                                frameShape = frameShape, protocol=protocol, 
                                                guiUpdateCallback=self._ProgressUpate, complevel=compLevel, 
                                                fakeCamCycleTime=fakeCycleTime, maxFrames=maxFrames)
        elif self.spoolType == 'Cluster':
            from PYME.Acquire import HTTPSpooler
            self.queueName = self._get_queue_name(fn, pcs=(not cluster_h5))
            self.spooler = HTTPSpooler.Spooler(self.queueName, self.scope.frameWrangler.onFrame, 
                                               frameShape = frameShape, protocol=protocol, 
                                               guiUpdateCallback=self._ProgressUpate, complevel=compLevel, 
                                               fakeCamCycleTime=fakeCycleTime, maxFrames=maxFrames,
                                               compressionSettings=compressionSettings, aggregate_h5=cluster_h5)
           
        else:
            from PYME.Acquire import HDFSpooler
            self.spooler = HDFSpooler.Spooler(self._get_queue_name(fn), self.scope.frameWrangler.onFrame,
                                              frameShape = frameShape, protocol=protocol, 
                                              guiUpdateCallback=self._ProgressUpate, complevel=compLevel, 
                                              fakeCamCycleTime=fakeCycleTime, maxFrames=maxFrames)

        #TODO - sample info is probably better handled with a metadata hook
        #if sampInf:
        #    try:
        #        sampleInformation.getSampleData(self, self.spooler.md)
        #    except:
        #        #the connection to the database will timeout if not present
        #        #FIXME: catch the right exception (or delegate handling to sampleInformation module)
        #        pass
            
        self.spooler.onSpoolStop.connect(self.SpoolStopped)
        self.spooler.StartSpool()
        
        self.onSpoolStart.send(self)
        
        #return a function which can be called to indicate if we are done
        return lambda : not self.spooler.spoolOn

    @property
    def rel_dirname(self):
        return self._sep.join(self._subdir)

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
        from PYME.Acquire import QueueSpooler, HTTPSpooler
        
        dh5view_cmd = 'dh5view'
        if sys.platform == 'win32':
            dh5view_cmd = 'dh5view.exe'
            
        if self.autostart_analysis:
            dh5view_cmd += ' -g'
        
        if isinstance(self.spooler, QueueSpooler.Spooler): #queue or not
            subprocess.Popen('%s -q %s QUEUE://%s' % (dh5view_cmd, self.spooler.tq.URI, self.queueName), shell=True)
        elif isinstance(self.spooler, HTTPSpooler.Spooler): #queue or not
            if self.autostart_analysis:
                self.launch_cluster_analysis()
            else:
                subprocess.Popen('%s %s' % (dh5view_cmd, self.spooler.getURL()), shell=True)
     
    def launch_cluster_analysis(self):
        from PYME.cluster import HTTPRulePusher
        
        seriesName = self.spooler.getURL()
        try:
            HTTPRulePusher.launch_localize(self.scope.analysisSettings.analysisMDH, seriesName)
        except:
            logger.exception('Error launching analysis for %s' % seriesName)


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
        self._update_series_counter()

