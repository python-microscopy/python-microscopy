# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:42:16 2016

@author: david
"""


#import datetime
from PYME.IO import acquisition_backends
# TODO: change to use a metadata handler / provideStartMetadata hook
# MetaDataHandler.provideStartMetadata from the init file when
# loading the sampleinfo interface, see Acquire/Scripts/init.py
try:
    from PYME.Acquire import sampleInformation
    sampInf = True
except:
    print('Could not connect to the sample information database')
    sampInf = False
#import win32api
from PYME.IO.FileUtils import nameUtils
from PYME.IO.FileUtils.nameUtils import numToAlpha, getRelFilename, genHDFDataFilepath
from PYME.IO import unifiedIO, MetaDataHandler

from PYME.Acquire.protocol_acquisition import ProtocolAcquisition
from PYME.Acquire.xyztc import XYZTCAcquisition, ZStackAcquisition


#import PYME.Acquire.Protocols
import PYME.Acquire.protocol as prot
from PYME.Acquire.ui import preflight
from PYME.Acquire import stackSettings
from PYME import config

from PYME.misc import hybrid_ns

import os
import sys
import json

import subprocess
import threading
try:
    import queue
except ImportError:
    # py2, remove this when we can
    import Queue as queue # type: ignore

from PYME.contrib import dispatch

import logging
logger = logging.getLogger(__name__)

from PYME.util import webframework

class ProtocolAcquisitionSettings(object):
    '''
    Manages settings which are specific to protocol-based acquisitions
    '''
    def __init__(self) -> None:
        self.protocol = prot.NullProtocol
        self.protocolZ = prot.NullZProtocol
        self.z_stepped = False  # z-step during acquisition
        self.z_dwell = 100 # time to spend at each z level (if z_stepped == True)

    def set_protocol(self, protocolName=None, reloadProtocol=True):
        """
        Set the current protocol.
        
        Parameters
        ----------
        protocolName: str
            path to protocol file including extension
        reloadProtocol : bool
            currently ignored; protocol module is reinitialized regardless.
        
        See also: PYME.Acquire.Protocols.
        """

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
            self.z_dwell = self.protocolZ.dwellTime

    def get_protocol_for_acquisition(self, settings={}):
        stack = settings.get('z_stepped', self.z_stepped)
        stack_settings = settings.get('stack_settings', None)
        
        # try stack settings for z_dwell, then aq settings.
        # precedence is settings > stack_settings > self.z_dwell
        # The reasoning for allowing the dwell time to be set in either the spooling or stack settings is to allow
        # API users to choose which is most coherent for their use case (it would seem logical to put dwell time with
        # the other stack settings, but this becomes problematic when sharing stack settings across modalities - e.g.
        # PALM/STORM and widefield stacks which are likely to share most of the stack settings but have greatly different
        # z dwell times). PYMEAcquire specifies it in the spooling/series settings by default to allow shared usage
        # between modalities.
        if stack_settings:
            if isinstance(stack_settings, dict):
                z_dwell = stack_settings.get('DwellFrames', self.z_dwell)
            else:
                # have a StackSettings object
                # TODO - fix this to be a bit more sane and not use private attributes etc ...
                z_dwell = stack_settings._dwell_frames
                # z_dwell defaults to -1  (with a meaning of ignore) in StackSettings objects if not value is not
                # explicitly provided. In this case, use our internal value instead. The reason for the 'ignore'
                # special value is to allow the same StackSettings object to be used for widefield stacks and
                # localization series (where sharing everything except dwell time makes sense).
                if z_dwell < 1:
                    z_dwell = self.z_dwell
        else:
            z_dwell = self.z_dwell
        
        z_dwell = settings.get('z_dwell', z_dwell)
        protocol_name = settings.get('protocol_name', None)

        if protocol_name is None:
            protocol, protocol_z = self.protocol, self.protocolZ
        else:
            pmod = prot.get_protocol(protocol_name)
            protocol, protocol_z = pmod.PROTOCOL, pmod.PROTOCOL_STACK

        if stack:
            protocol = protocol_z
            protocol.dwellTime = z_dwell
            #print(protocol)
        else:
            protocol = protocol

        return protocol

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
        
        if int(sys.version[0]) < 3:
            #default to Queue for Py2
            self.spoolType = 'Queue'
        else:
            #else default to file
            self.spoolType = 'File'
        
        self._base_dir = nameUtils.get_local_data_directory()
        self._dirname = os.sep.join([self._base_dir, ] + nameUtils.get_spool_subdir())
        self._cluster_dirname = self.get_cluster_dirname(self._dirname)

        self.seriesStub = defSeries % nameUtils.dateDict

        self.seriesCounter = 0
        self._series_name = None

        self.acquisition_types = {
                                  'ZStackAcquisition': ZStackAcquisition,
                                  'ProtocolAcquisition': ProtocolAcquisition,
                                 }
        
        
        self.acquisition_type='ProtocolAcquisition'
        self.protocol_settings = ProtocolAcquisitionSettings()
        
        self.onSpoolProgress = dispatch.Signal()
        self.onSpoolStart = dispatch.Signal()
        self.on_stop = dispatch.Signal()

        self.analysis_mode = 'interactive' # 'interactive' or 'rule-based'
        self.analysis_rule_name = 'default'
        self.analysis_launch_mode = 'triggered' # 'triggered' or 'series-end
        

        self._analysis_launchers = queue.Queue(3)
        
        self._status_changed_condition = threading.Condition()
        
        #settings which were managed by GUI
        self.hdf_compression_level = 2 # zlib compression level that pytables should use (spool to file and queue)

        self.cluster_h5 = False # spool to h5 on cluster (cluster of one)
        self.pzf_compression_settings=acquisition_backends.ClusterBackend.default_compression_settings # only for cluster spooling

        #check to see if we have a cluster
        self._N_data_servers = len(hybrid_ns.getNS('_pyme-http').get_advertised_services())
        if self._N_data_servers > 0:
            # switch to cluster as spool method if available.
            self.SetSpoolMethod('Cluster')
            
        if self._N_data_servers  == 1:
            self.cluster_h5 = True # we have a cluster of one
            
    @property
    def available_spool_methods(self):
        if False:#self.acquisition_type == 'ProtocolAcquision':
            return ['File', 'Cluster']
        else:
            return ['File', 'Cluster', 'Memory']
        
    def get_info(self):
        info =  {'settings' : self.get_settings(),
                 'series_name' : self.seriesName,
                 'available_spool_methods' : self.available_spool_methods
                }
        
        try:
            info['status'] = self.spooler.status()
        except AttributeError:
            info['status'] = {'spooling':False}
        
        return info
    
    def register_acquisition_type(self, name, cls):
        self.acquisition_types[name] = cls
    
    def update_settings(self, settings):
        """
        Sets the state of the `SpoolController` by calling set-methods or by 
        setting attributes.

        Parameters
        ----------
        settings : dict
            keys should be `SpoolController` attributes or properties with 
            setters. Not all keys must be present, and example keys include:
                method : str
                    One of 'File', 'Cluster', or 'Queue'(py2 only)
                hdf_compression_level: int
                    zlib compression level that pytables should use (spool to 
                    file and queue)
                z_stepped : bool
                    toggle z-stepping during acquisition
                z_dwell : int
                    number of frames to acquire at each z level (predicated on
                    `SpoolController.z_stepped` being True)
                cluster_h5 : bool
                    Toggle spooling to single h5 file on cluster rather than pzf
                    file per frame. Only applicable to 'Cluster' `method` and
                    preferred for PYMEClusterOfOne.
                pzf_compression_settings : dict
                    Compression settings relevant for 'Cluster' `method` if 
                    `cluster_h5` is False. See acquisition_backends.ClusterBackend.default_compression_settings.
                protocol_name : str
                    Note that passing the protocol name will force a (re)load of
                    the protocol file (even if it is already selected).
            Notable keys which are not supported through this method include 
            'series_name', 'seriesName' and 'dirname'.
        
        
        """
        method = settings.pop('method', None)
        if method and method != self.spoolType:
            self.SetSpoolMethod(method)
        
        protocol_name = settings.pop('protocol_name', None)
        if protocol_name:
            self.protocol_settings.SetProtocol(protocol_name)

            
        pzf_settings = settings.pop('pzf_compression_settings', None)
        if pzf_settings:
            self.pzf_compression_settings = dict(pzf_settings)
        
        for k, v in settings.items():
            if k in ['z_stepped', 'z_dwell']:
                # these settings belong to the protocol
                # TODO - rename to protocol.xx???
                setattr(self.protocol_settings, k, v)
            else:
                setattr(self, k, v)

        with self._status_changed_condition:
            self._status_changed_condition.notify_all()
            
            
        
    @property
    def _sep(self):
        if self.spoolType == 'Cluster':
            return '/'
        else:
            return os.sep
        
    @property
    def dirname(self):
        return self.get_dirname()
    
    def get_dirname(self, subdirectory=None, spoolType=None):
        """ Get the current directory name, including any subdirectories from
        chunking or additional spec.

        Parameters
        ----------
        subdirectory : str, optional
            Directory within current set directory to spool this series. The
            directory will be created if it doesn't already exist.

        spoolType : str, optional
            Used  when using get_dirname externally (e.g. in tiling) to over-ride the spool type.

        Returns
        -------
        str
            spool directory name
        """
        
        if spoolType is None:
            spoolType = self.spoolType

        dir = self._dirname if spoolType.lower() != 'cluster' else self._cluster_dirname

        if subdirectory != None:
            dir = dir + self._sep + subdirectory.replace(os.sep, self._sep)
        
        if config.get('acquire-spool_subdirectories', False):
            # limit single directory size for (cluster) IO performance
            subdir = '%03d' % int(self.seriesCounter/100)
            dir = dir + self._sep + subdir

        return dir

    def get_cluster_dirname(self, dirname):
        # Typically we'll be below the base directory, which we want to remove
        dir = dirname.replace(self._base_dir + os.sep, '')
        # if we weren't below PYMEData dir, which probably isn't great, at least drop any windows nonsense
        dir = dir.split(':')[-1]
        return unifiedIO.verbose_fix_name(dir.replace(os.sep, '/'))
        
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
        return self.seriesStub + '_' + numToAlpha(self.seriesCounter)
       
    def _checkOutputExists(self, fn):
        if self.spoolType == 'Cluster':
            #FIXME - remove dependance on HTTPSpooler
            from PYME.IO import HTTPSpooler_v2 as HTTPSpooler
            # special case for HTTP spooling.  Make sure 000\series.pcs -> 000/series.pcs
            pyme_cluster = self.dirname + '/' + fn.replace('\\', '/')
            logger.debug('Looking for %s (.pcs or .h5) on cluster' % pyme_cluster)
            return HTTPSpooler.exists(pyme_cluster + '.pcs') or HTTPSpooler.exists(pyme_cluster + '.h5') or HTTPSpooler.exists(pyme_cluster + '.tiles')
            #return (fn + '.h5/') in HTTPSpooler.clusterIO.listdir(self.dirname)
        else:
            local_stub = os.sep.join([self.dirname, fn])
            local_h5 = local_stub + '.h5'
            logger.debug('Looking for %s on local machine' % local_stub)
            return os.path.exists(local_h5) or os.path.exists(local_stub + '.pcs') or os.path.exists(local_stub + '.tiles')
        
    def get_free_space(self):
        """
        Get available space in the target spool directory
        
        Returns
        -------
        
        free space in GB

        """
        if self.spoolType == 'Cluster':
            from PYME.cluster import status
            nodes = status.get_polled_status()
            free_storage = sum([n['Disk']['free'] for n in nodes])
            return free_storage / 1e9
        else:
            from PYME.IO.FileUtils.freeSpace import get_free_space
            # avoid dirname property here so we can differ building
            # 'acquire-spool_subdirectories' to `start_spooling`
            return get_free_space(self._dirname)/1e9
        
    def _update_series_counter(self):
        logger.debug('Updating series counter')
        self.seriesCounter = 0
        self.seriesName = self._GenSeriesName()
        while self._checkOutputExists(self.seriesName):
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
            
    def SetSpoolDir(self, dirname):
        """Set the directory we're spooling into"""
        logger.info('Setting spool dir: %s' % dirname)
        self._dirname = dirname
        self._cluster_dirname = self.get_cluster_dirname(dirname)
        #if we've had to quit for whatever reason start where we left off
        self._update_series_counter()
            
    def _ProgressUpate(self, **kwargs):
        with self._status_changed_condition:
            self._status_changed_condition.notify_all()
            
        self.onSpoolProgress.send(self)

    @property
    def acquisition_cls(self):
        return self.acquisition_types[self.acquisition_type]
        
    def _get_queue_name(self, fn, pcs=False, subdirectory=None):
        """ Get fully resolved uri to spool to

        Parameters
        ----------
        fn : str
            file stub of the series
        pcs : bool, optional
            sets extension to PYME Cluster Series (pcs) if spooling series as
            plurality of pcs files, by default False (spooling to h5 file)
        subdirectory : str, optional
            Directory within current set directory to spool this series. The
            directory will be created if it doesn't already exist.

        Returns
        -------
        str
            fully resolved uri to spool to
        """
        if pcs:
            ext = '.pcs'
        else:
            ext = '.h5'

        # allow acquisition types (e.g. tiling) to specify their own extension
        ext = getattr(self.acquisition_cls, 'FILE_EXTENSION', ext)
        
        return self._sep.join([self.get_dirname(subdirectory), fn + ext])


    def start_spooling(self, fn=None, settings={}, preflight_mode='interactive'):
        """

        Parameters
        ----------
        fn : str, optional
            fn can be hardcoded here, otherwise differs to the seriesName
            property which will create one if need-be.
        settings : dict
            keys should be `SpoolController` attributes or properties with
            setters. Not all keys must be present, and example keys include:
                method : str
                    One of 'File', 'Cluster', or 'Queue'(py2 only)
                hdf_compression_level: int
                    zlib compression level that pytables should use (spool to
                    file and queue)
                z_stepped : bool
                    toggle z-stepping during acquisition
                z_dwell : int
                    number of frames to acquire at each z level (predicated on
                    `SpoolController.z_stepped` being True)
                cluster_h5 : bool
                    Toggle spooling to single h5 file on cluster rather than pzf
                    file per frame. Only applicable to 'Cluster' `method` and
                    preferred for PYMEClusterOfOne.
                pzf_compression_settings : dict
                    Compression settings relevant for 'Cluster' `method` if
                    `cluster_h5` is False. See acquisition_backends.ClusterBackend.defaultCompSettings.
                protocol_name : str
                    Note that passing the protocol name will force a (re)load of
                    the protocol file (even if it is already selected).
                max_frames : int, optional
                    point at which to end the series automatically, by default
                    sys.maxsize
                subdirectory : str, optional
                    Directory within current set directory to spool this series. The
                    directory will be created if it doesn't already exist.
                extra_metadata : dict, optional
                    metadata to supplement this series for entries known prior to
                    acquisition which do not have handlers to hook start metadata
        preflight_mode : str (default='interactive')
            What to do when the preflight check fails. Options are 'interactive', 'warn', 'abort' and 'skip' which will
            display a dialog and prompt the user, log a warning and continue, and log an error and abort, or skip completely.
            The former is suitable for interactive acquisition, whereas one of the latter modes is likely better for automated spooling
            via the action manager.

        """
        
        # these settings were managed by the GUI, but are now managed by the 
        # controller, still allow them to be passed in, but default to internals

        acquisition_type = settings.get('acquisition_type', self.acquisition_type)
        
        fn = self.seriesName if fn in ['', None] else fn
        
        #compLevel = settings.get('hdf_compression_level', self.hdf_compression_level)
        #pzf_compression_settings = settings.get('pzf_compression_settings', self.pzf_compression_settings)
        cluster_h5 = settings.get('cluster_h5', self.cluster_h5)
        
        subdirectory  = settings.get('subdirectory', None)

        # make directories as needed, makedirs(dir, exist_ok=True) once py2 support is dropped
        if (self.spoolType != 'Cluster') and (not os.path.exists(self.get_dirname(subdirectory))):
                os.makedirs(self.get_dirname(subdirectory))

        if self._checkOutputExists(fn): #check to see if data with the same name exists
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
            
            raise IOError('A series with the same name already exists')
            
        # update launch analysis settings
        self.analysis_mode = settings.get('analysis_mode', self.analysis_mode)
        self.analysis_rule_name = settings.get('analysis_rule_name', self.analysis_rule_name)
        self.analysis_launch_mode = settings.get('analysis_launch_mode', self.analysis_launch_mode)  
        
        if self.spoolType == 'Cluster':
            self.queueName = self._get_queue_name(fn, pcs=(not cluster_h5), 
                                                  subdirectory=subdirectory)
        else:
            self.queueName = self._get_queue_name(fn, subdirectory=subdirectory)


        from PYME.IO import acquisition_backends
        backends = {'File': acquisition_backends.HDFBackend,
                    'Cluster': acquisition_backends.ClusterBackend, 
                    'Memory': acquisition_backends.MemoryBackend}
        
        backend_kwargs = {}
        if self.spoolType == 'Cluster':
            backend_kwargs['cluster_h5'] = settings.get('cluster_h5', self.cluster_h5)
            backend_kwargs['compression_settings'] = settings.get('pzf_compression_settings', self.pzf_compression_settings)
        elif self.spoolType == 'File':
            backend_kwargs['complevel'] = settings.get('hdf_compression_level', self.hdf_compression_level)
        
        # put preflight mode into settings so we can pass it to the protocol acquisition
        settings['preflight_mode'] = preflight_mode

        
        try:
            self.spooler = self.acquisition_cls.from_spool_settings(self.scope, settings, backend=backends[self.spoolType], backend_kwargs=backend_kwargs, series_name=self.queueName, spool_controller=self)
        except KeyError:
            raise RuntimeError('Unknown acquisition type %s' % acquisition_type)
        
        self.spooler.on_progress.connect(self._ProgressUpate)
           
        
        extra_metadata = settings.get('extra_metadata')
        if extra_metadata is not None:
            self.spooler.md.mergeEntriesFrom(MetaDataHandler.DictMDHandler(extra_metadata))

        # NOTE - stopping and starting the framewrangler has moved to the spooler .start() method
        #self.scope.frameWrangler.stop()
        
        try:
            self.spooler.on_stop.connect(self.SpoolStopped)
            self.spooler.start()
        except:
            self.spooler.abort()
            raise

        # restart frame wrangler
        #self.scope.frameWrangler.Prepare()
        #self.scope.frameWrangler.start()
        
        self.onSpoolStart.send(self)

        if self.spoolType == 'Memory':
            # open a viewer window for the data that is being acquired
            self._display_image()
        
        #return a function which can be called to indicate if we are done
        return lambda : self.spooler.spool_complete
    
    def estimate_spool_time(self, settings={}, **kwargs):
        """
        Estimate the time to spool a series based on the current settings

        used by queued actions to set timeouts etc ... if in doubt, we should 
        overestimate.
        
        Returns
        -------
        float
            estimated time in seconds

        FIXME - these are extremely rough estimates
        FIXME - defer to acquisition type
        """
        
        acquisition_type = settings.get('acquisition_type', self.acquisition_type)
        
        if acquisition_type == 'ProtocolAcquisition':
            #FIXME - this is a very rough estimate
            n_frames = settings.get('max_frames', 100000)

            try:
                return  1.25 * n_frames / self.scope.cam.GetFPS()  # per series
            except NotImplementedError:
                # specifically the simulated camera here, which has a non-predictable frame rate
                # use a conservative default of 10 s/frame (should not matter as simulation will generally not be doing 10s of thousands of series)
                return 10*n_frames

        
        else:
            # 30 minutes for all other acquisition types
            # TODO - does this need to be longer for tiling??
            return 30*60  
    
    def get_settings(self, method_only=False):
        """Get the current settings for the spool controller
        
        Used when adding actions to the action manager - this should freeze
        the relevant settings for the acquisition type and method.
        """
        settings = {'method' : self.spoolType,
        }

        if self.spoolType == 'File':
            settings['hdf_compression_level'] = self.hdf_compression_level

        if self.spoolType == 'Cluster':
            settings['cluster_h5'] = self.cluster_h5
            settings['pzf_compression_settings'] = self.pzf_compression_settings

        if method_only:
            return settings
        
        else:
            settings['acquisition_type'] = self.acquisition_type
            
            settings['analysis_mode'] = self.analysis_mode
            settings['analysis_launch_mode'] = self.analysis_launch_mode
            settings['analysis_rule_name'] = self.analysis_rule_name
            
            settings.update(self.acquisition_types[self.acquisition_type].get_frozen_settings(self.scope, self)) 
            
            return settings

    
    def _display_image(self):
        ''' Display the image in a viewer (for memory backend)
        '''
        try:
            assert isinstance(self.spooler.storage, acquisition_backends.MemoryBackend)
            
            import wx
            if not wx.App.IsMainLoopRunning():
                logger.debug("No wx app, can't show image")
                return
            
            from PYME.DSView import ViewIm3D
            self._view = ViewIm3D(self.spooler.storage.image)
            self.scope.frameWrangler.onFrameGroup.connect(self._update_display)
        except:
            logger.exception('Error displaying image')

    def _update_display(self, *args, **kwargs):
        ''' Update the displayed image (for memory backend)
        '''
        import wx
        wx.CallAfter(self._view.view.Redraw)

    def _unlink_display(self, *args, **kwargs):
        ''' Unlink the display from the frameWrangler (for memory backend)
        '''
        try:
            self.scope.frameWrangler.onFrameGroup.disconnect(self._update_display)
        except:
            pass

    @property
    def display_dirname(self):
        """ 
        Returns a relative directory name for display in user interfaces

        Returns
        -------
        dirname : str
            current spool directory, relative to local PYMEData directory 
            (ideally)
        """
        dirname = self.dirname
        if self.spoolType == 'Cluster':
            return dirname
        else:
            return dirname.replace(self._base_dir + os.sep, '')

    def StopSpooling(self, **kwargs):
        """GUI callback to stop spooling."""
        self.spooler.stop()
        
    def SpoolStopped(self, **kwargs):
        self.seriesCounter +=1
        self.seriesName = self._GenSeriesName()

        logger.info('Spooling stopped')
        
        self.on_stop.send(self)

        if self.analysis_launch_mode == 'series-end':
            self.LaunchAnalysis()

        try:
            self.spooler.on_progress.disconnect(self._ProgressUpate)
            self._ProgressUpate()
            self._unlink_display()
        except AttributeError:
            pass

        
    @property
    def autostart_analysis(self):
        if 'analysisSettings' in dir(self.scope):
            return self.scope.analysisSettings.propagateToAcquisisitonMetadata
        else:
            return False
        

    def LaunchAnalysis(self):
        from warnings import warn
        warn('LaunchAnalysis is deprecated, use launch_analysis instead', DeprecationWarning)
        self.launch_analysis()
    
    def launch_analysis(self):
        """Launch analysis
        """
        import posixpath

        try:
            if not self.spoolType == 'Cluster':
                from PYME import warnings
                warnings.warn('Analysis is only supported for cluster spooling', category=RuntimeWarning)
        
            if self.analysis_mode == 'interactive':
                subprocess.Popen('%s %s' % (self.dh5view_cmd, self.spooler.getURL()), shell=True)

            elif self.analysis_mode == 'rule-based':
                seriesName = self.spooler.getURL()

                try:
                    # we have the chained analysis module loaded
                    rule_factory = self.scope.analysis_rules[self.analysis_rule_name].rule_factories[0]

                    context = {
                        'seriesName': seriesName,
                        'inputs': {'input': seriesName}, # needed for recipes
                        'output_dir':  posixpath.split(seriesName)[0],
                        'spooler': self.spooler.storage, # for SpoolLocalLocalization rule completeness check
                    }

                    rule =  rule_factory.get_rule(context=context)
                    # launch analysis in a separate thread    

                except AttributeError:
                    # we don't have the chained analysis module loaded
                    from PYME.cluster import rules
                    import warnings
                    warnings.warn('using legacy automated localisation rule - please add the chained analysis module to your init and use this instead', category=RuntimeWarning)
                    rule = rules.LocalisationRule(seriesName=seriesName, analysisMetadata=self.scope.analysisSettings.analysisMDH)

                except KeyError:
                    raise RuntimeError('Analysis rule %s not found' % self.analysis_rule_name)
                
                t = threading.Thread(target=rule.push)
                t.start()
                # keep track of a couple launching threads to make sure they have ample time to finish before joining
                if self._analysis_launchers.full():
                    self._analysis_launchers.get().join()
                self._analysis_launchers.put(t)

                self._rule_outputs = rule.output_files
        except:
            logger.exception('Error launching analysis')

    # TODO - do these belong here?
    @property
    def pymevis_cmd(self):
        if sys.platform == 'win32':
            return 'PYMEVis.exe'
        else:
            return 'PYMEVis'
        
    @property
    def pymeimage_cmd(self):
        if sys.platform == 'win32':
            return 'PYMEImage.exe'
        else:
            return 'PYMEImage'
    
    dh5view_cmd = pymeimage_cmd
    
    def open_analysis(self):
        """Open the currenly running analysis in PYMEVis"""
        import subprocess
        
        output = self._rule_outputs.get('results')
        
        # get the URL
        if output.endswith('.h5r'):
            uri = output + '?live'
            subprocess.Popen('%s %s' % (self.pymevis_cmd, uri), shell=True)
        elif output.endswith('.h5') or output.endswith('.tif'):
            uri = output
            subprocess.Popen('%s %s' % (self.pymeimage_cmd, uri), shell=True)

    def open_view(self):
        if hasattr(self.spooler, '_launch_viewer'):
            # TODO - make less special case - maybe defer normal launch to the spooler as well
            self.spooler._launch_viewer()
        else:
            subprocess.Popen('%s %s' % (self.pymeimage_cmd, self.spooler.getURL()), shell=True)


            
     
    # def launch_cluster_analysis(self):
    #     from PYME.cluster import rules
        
    #     seriesName = self.spooler.getURL()
    #     try:
    #         #HTTPRulePusher.launch_localize(self.scope.analysisSettings.analysisMDH, seriesName)
    #         rules.LocalisationRule(seriesName=seriesName, analysisMetadata=self.scope.analysisSettings.analysisMDH).push()
    #     except:
    #         logger.exception('Error launching analysis for %s' % seriesName)

            
    def SetSpoolMethod(self, method):
        """Set the spooling method
        
        Parameters
        ----------
        
        method : string
            One of 'File', 'Queue', or 'Cluster'
        """
        self.spoolType = method
        self._update_series_counter()

    def __del__(self):
        # make sure our analysis launchers have a chance to finish their job before exiting
        while not self._analysis_launchers.empty():
            self._analysis_launchers.get().join()
            


class SpoolControllerWrapper(object):
    def __init__(self, spool_controller):
        self.spool_controller = spool_controller # type: SpoolController

    @webframework.register_endpoint('/info', output_is_json=False)
    def info(self):
        return self.spool_controller.get_info()

    @webframework.register_endpoint('/info_longpoll', output_is_json=False)
    def info_longpoll(self):
        with self.spool_controller._status_changed_condition:
            self.spool_controller._status_changed_condition.wait()
            return self.spool_controller.get_info()

    @webframework.register_endpoint('/settings', output_is_json=False)
    def settings(self, body):
        import json
        try:
            self.spool_controller.update_settings(json.loads(body))
            return 'OK'
        except:
            logger.exception('Error setting spool controller settings')
            return 'Failure'

    @webframework.register_endpoint('/stop_spooling', output_is_json=False)
    def stop_spooling(self):
        self.spool_controller.StopSpooling()
        return 'OK'

    @webframework.register_endpoint('/start_spooling', output_is_json=False)
    def start_spooling(self, body, filename=None, preflight_mode='abort'):
        """
        See also SpoolController.start_spooling()

        Parameters
        ----------
        filename : str, optional
            fn can be hardcoded here, otherwise differs to the seriesName
            property which will create one if need-be.
        preflight_mode : str, default == 'abort'
             One of 'warn', 'abort', 'skip, or 'interactive'. Note that 'interactive' requires an active wx.App
             
        The majority of parameters are passed in the request body, which should be a json-formatted dictionary with the
        the following keys (see also `settings` parameter to `SpoolController.start_spooling`
        
        z_stepped : bool, optional
            toggle z-stepping during acquisition. By default None, which differs
            to current `SpoolController` state.
        hdf_comp_level : int, optional
            zlib compression level for pytables. Not relevant for `Cluster`
            spool method unless `cluster_h5` is True. By default None, which 
            differs to current `SpoolController` state.
        z_dwell : int, optional
            frames per z-step. By default None, which differs to current 
            `SpoolController` state.
        max_frames : int, optional
            point at which to end the series automatically, by default 
            sys.maxsize
        pzf_compression_settings : dict, optional
            Compression settings relevant for 'Cluster' `method` if `cluster_h5`
            is False. See acquisition_backends.ClusterBackend.defaultCompSettings. By default None, 
            which defers to current `SpoolController` state.
        cluster_h5 : bool, optional
            Toggle spooling to single h5 file on cluster rather than pzf file 
            per frame. Only applicable to 'Cluster' `method` and preferred for 
            PYMEClusterOfOne. By default None, which differs to current 
            `SpoolController` state.
        protocol : str, optional
            path to acquisition protocol. By default None which differs to 
            current `SpoolController` state.
        subdirectory : str, optional
            Directory within current set directory to spool this series. The
            directory will be created if it doesn't already exist.
        extra_metadata : dict, optional
            metadata to supplement this series for entries known prior to
            acquisition which do not have handlers to hook start metadata
        stack_settings : dict, optional
            The stack settings. See PYME.Acquire.stackSettings.StackSettings. By default the global StackSettings instance
            is used.
             
        """
        import json
        # FIXME - do some sanity checks on filename (this can't be as simple as urlescaping, as we need to support
        # URIs as well as filenames). In practice this is best dealt with by enforcing authentication and only using on a 
        # trusted network.
        if len(body) > 0:
            # have settings in message body
            self.spool_controller.start_spooling(filename, settings=json.loads(body), preflight_mode=preflight_mode)
        else:
            self.spool_controller.start_spooling(filename, preflight_mode=preflight_mode)
        return 'OK'
