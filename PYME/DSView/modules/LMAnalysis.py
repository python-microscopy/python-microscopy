#!/usr/bin/python
##################
# LMAnalysis.py
#
# Copyright David Baddeley, 2011
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
import posixpath
import PYME.localization.FitFactories
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
from PYME.contrib import dispatch
import numpy as np
import wx
import wx.lib.agw.aui as aui
from PYME.DSView import fitInfo
#from PYME.DSView.OverlaysPanel import OverlayPanel
from PYME.IO import MetaDataHandler
from PYME.IO import tabular
from PYME.IO.FileUtils import fileID
from PYME.IO.FileUtils.nameUtils import genResultFileName, genClusterResultFileName
from PYME.LMVis import pipeline
from PYME.LMVis import progGraph as progGraph
from PYME.localization import MetaDataEdit as mde
from PYME.localization import remFitBuf
from PYME.ui.mytimer import mytimer
import sys

import logging
logger = logging.getLogger(__name__)

if sys.version_info.major >= 3:
    # always use new style on py3
    NEW_STYLE_DISTRIBUTION = True
else:
    # use new style on py2 if we detect a running ruleserver
    # TODO - distinguish between local and remote ruleservers
    try:
        from PYME.cluster import distribution
        
        #test for a running task distributor
        distribURIs = distribution.getDistributorInfo()
        assert(distribURIs) # fail if we have an empty dictionary
        NEW_STYLE_DISTRIBUTION = True
    except:
        logger.exception('Did not find ruleserver, falling back to Pyro task queue')
        NEW_STYLE_DISTRIBUTION = False

debug = True

MAP_WILDCARDS = 'TIFF|*.tif;|All files|*.*'

def debugPrint(msg):
    if debug:
        print(msg)

def _verifyResultsFilename(resultsFilename):
    if os.path.exists(resultsFilename):
        di, fn = os.path.split(resultsFilename)
        i = 1
        stub = os.path.splitext(fn)[0]
        while os.path.exists(os.path.join(di, stub + '_%d.h5r' % i)):
            i += 1
        fdialog = wx.FileDialog(None, 'Analysis file already exists, please select a new filename',
                    wildcard='H5R files|*.h5r', defaultDir=di, defaultFile=stub + '_%d.h5r' % i, style=wx.FD_SAVE)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            resultsFilename = fdialog.GetPath()
        else:
            raise RuntimeError('Invalid results file - not running')
            
    return resultsFilename


def _verifyClusterResultsFilename(resultsFilename):
    from PYME.IO import clusterIO, unifiedIO
    
    resultsFilename = unifiedIO.verbose_fix_name(resultsFilename) # fix any spaces in the input filename
    
    if clusterIO.exists(resultsFilename):
        di, fn = posixpath.split(resultsFilename)
        i = 1
        stub = posixpath.splitext(fn)[0]
        while clusterIO.exists(posixpath.join(di, stub + '_%d.h5r' % i)):
            i += 1

        fdialog = wx.TextEntryDialog(None, 'Analysis file already exists, please select a new filename')#,
        fdialog.SetValue(stub + '_%d.h5r' % i)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            resultsFilename = posixpath.join(di, str(fdialog.GetValue()))
        else:
            raise RuntimeError('Invalid results file - not running')

    return resultsFilename


class AnalysisSettingsView(object):
    FINDING_PARAMS = [mde.FloatParam('Analysis.DetectionThreshold', 'Thresh:', 1.0),
                      mde.IntParam('Analysis.DebounceRadius', 'Debounce rad:', 4),
    ]
    
    DEFAULT_PARAMS = [mde.IntParam('Analysis.StartAt', 'Start at:', default=30),
                      mde.ParamGroup('Background Subtraction', [
                          mde.RangeParam('Analysis.BGRange', 'Background:', default=(-30,0)),
                          mde.BoolParam('Analysis.subtractBackground', 'Subtract background in fit', default=True),
                          mde.BoolFloatParam('Analysis.PCTBackground' , 'Use percentile for background', default=False, helpText='', ondefault=0.25, offvalue=0),
                          ], folded=False),
                      mde.ParamGroup('sCMOS camera noise maps',
                             [mde.FilenameParam('Camera.VarianceMapID', 'Variance Map:',
                                                prompt='Please select variance map to use ...',
                                                wildcard=MAP_WILDCARDS, filename=''),
                              mde.FilenameParam('Camera.DarkMapID',
                                                'Dark Map:', prompt='Please select dark map to use ...',
                                                wildcard=MAP_WILDCARDS, filename=''),
                              mde.FilenameParam('Camera.FlatfieldMapID', 'Flatfield Map:',
                                                prompt='Please select flatfield map to use ...',
                                                wildcard=MAP_WILDCARDS, filename=''),
                             ]),
                      mde.ParamGroup('Fiducial based drift tracking',
                             [mde.BoolParam('Analysis.TrackFiducials', 'Track Fiducials', default=False),
                              mde.FloatParam('Analysis.FiducialThreshold', 'Fiducial Threshold', default=1.8),
                              mde.IntParam('Analysis.FiducialROISize', 'Fiducial ROI', default=11),
                              mde.FloatParam('Analysis.FiducialSize', 'Fiducial Diameter [nm]', default=1000.),
                             ]),
    ]
    
    def __init__(self, dsviewer, analysisController, lmanal=None):
        self.foldAnalPanes = False

        self.analysisController = analysisController
        self.analysisMDH = analysisController.analysisMDH
        self.lmanal = lmanal

        dsviewer.paneHooks.append(self.GenPointFindingPanel)
        dsviewer.paneHooks.append(self.GenAnalysisPanel)
        
        self.analysisController.onMetaDataChange.connect(self.on_metadata_change)

    def _populateStdOptionsPanel(self, pan, vsizer):
        for param in self.DEFAULT_PARAMS:
            pg = param.createGUI(pan, self.analysisMDH, syncMdh=True, 
                                 mdhChangedSignal = self.analysisController.onMetaDataChange)
            vsizer.Add(pg, 0,wx.BOTTOM|wx.EXPAND, 5)
        vsizer.Fit(pan)
            
    def _populateFindOptionsPanel(self, pan, vsizer):
        for param in self.FINDING_PARAMS:
            pg = param.createGUI(pan, self.analysisMDH, syncMdh=True, 
                                 mdhChangedSignal = self.analysisController.onMetaDataChange)
            vsizer.Add(pg, 0,wx.BOTTOM|wx.EXPAND, 5)    
        
    def _populateCustomAnalysisPanel(self, pan, vsizer):
        try:
            fitMod = self.fitFactories[self.cFitType.GetSelection()]
            fm = __import__('PYME.localization.FitFactories.' + fitMod, fromlist=['PYME', 'localization', 'FitFactories'])
            
            #vsizer = wx.BoxSizer(wx.VERTICAL)
            for param in fm.PARAMETERS:
                pg = param.createGUI(pan, self.analysisMDH, syncMdh=True, 
                                 mdhChangedSignal = self.analysisController.onMetaDataChange)
                vsizer.Add(pg, 0,wx.BOTTOM|wx.EXPAND, 5)
            vsizer.Fit(pan)
                
        except AttributeError:
            pass
        
    def on_metadata_change(self, *args, **kwargs):
        fit_module = self.analysisMDH.getOrDefault('Analysis.FitModule', 'LatGaussFitFR')
        if not fit_module == self.fitFactories[self.cFitType.GetSelection()]:
            self.cFitType.SetSelection(self.fitFactories.index(fit_module))
            
        self.OnFitModuleChanged(None)
        
    def OnFitModuleChanged(self, event):
        self.customOptionsSizer.Clear(True)
        self._populateCustomAnalysisPanel(self.customOptionsPan, self.customOptionsSizer)
        self.customOptionsPan.GetTopLevelParent().OnSize(None)
        self.analysisMDH['Analysis.FitModule'] = self.fitFactories[self.cFitType.GetSelection()]

    def GenAnalysisPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Analysis", pinned = not(self.foldAnalPanes))

        #############################
        #std options
        pan = wx.Panel(item, -1)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        pan.SetSizer(vsizer)
        
        self._populateStdOptionsPanel(pan, vsizer)
                
        item.AddNewElement(pan)

        
        #######################
        #Fit factory selection
        pan = wx.Panel(item, -1)

        #find out what fit factories we have
        self.fitFactories = PYME.localization.FitFactories.resFitFactories
        print((self.fitFactories))

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Type:'), 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.cFitType = wx.Choice(pan, -1, choices = ['{:<35} \t- {:} '.format(f, PYME.localization.FitFactories.useFor[f]) for f in self.fitFactories], size=(110, -1))
        
        if 'Analysis.FitModule' in self.analysisMDH.getEntryNames():
            #has already been analysed - most likely to want the same method again
            try:
                self.cFitType.SetSelection(self.fitFactories.index(self.analysisMDH['Analysis.FitModule']))
                #self.tThreshold.SetValue('%s' % self.image.mdh.getOrDefault('Analysis.DetectionThreshold', 1))
            except ValueError:
                self.cFitType.SetSelection(self.fitFactories.index('LatGaussFitFR'))
                
        #elif 'Camera.ROIPosY' in self.image.mdh.getEntryNames() and (self.image.mdh.getEntry('Camera.ROIHeight') + 1 + 2*(self.image.mdh.getEntry('Camera.ROIPosY')-1)) == 512:
        #    #we have a symetrical ROI about the centre - most likely want to analyse using splitter
        #    self.cFitType.SetSelection(self.fitFactories.index('SplitterFitQR'))
        #    self.tThreshold.SetValue('0.5')
        else:
            self.cFitType.SetSelection(self.fitFactories.index('LatGaussFitFR'))
            self.analysisMDH['Analysis.FitModule'] = 'LatGaussFitFR'
            
        self.cFitType.Bind(wx.EVT_CHOICE, self.OnFitModuleChanged)

        hsizer.Add(self.cFitType, 1,wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)

        pan.SetSizer(vsizer)
        vsizer.Fit(pan)

        item.AddNewElement(pan)


        ########################################        
        #custom  (fit factory dependant) options        
        self.customOptionsPan = wx.Panel(item, -1)
        self.customOptionsSizer = wx.BoxSizer(wx.VERTICAL)
        self.customOptionsPan.SetSizer(self.customOptionsSizer)

        self._populateCustomAnalysisPanel(self.customOptionsPan, self.customOptionsSizer)        

        item.AddNewElement(self.customOptionsPan)
        
        ######################
        #Go
        if self.lmanal:
            self.bGo = wx.Button(item, -1, 'Go')

            self.bGo.Bind(wx.EVT_BUTTON, lambda e : self.lmanal.OnGo(e))
            item.AddNewElement(self.bGo)
        _pnl.AddPane(item)

        self.analysisPanel = item

    def GenPointFindingPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Point Finding", pinned = not(self.foldAnalPanes))

        pan = wx.Panel(item, -1)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        #load point finding settings
        self._populateFindOptionsPanel(pan, vsizer)
        
        if not self.lmanal is None:
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            #bTest = wx.Button(pan, -1, 'Test', style=wx.BU_EXACTFIT)
            #bTest.Bind(wx.EVT_BUTTON, lambda e : self.lmanal.Test())
            #hsizer.Add(bTest, 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
            
            bTestF = wx.Button(pan, -1, 'Test This Frame', style=wx.BU_EXACTFIT)
            bTestF.Bind(wx.EVT_BUTTON, lambda e : self.lmanal.OnTestFrame(e))
            hsizer.Add(bTestF, 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
            
            vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)
        
        pan.SetSizerAndFit(vsizer)

        item.AddNewElement(pan)
        _pnl.AddPane(item)
        



        
class AnalysisController(object):
    def __init__(self, imageMdh=None, tq = None):
        self.analysisMDH = MetaDataHandler.CopyOnWriteMDHandler(imageMdh) #MetaDataHandler.NestedClassMDHandler(imageMdh)
        self.onImagesPushed = dispatch.Signal()
        self.onMetaDataChange = dispatch.Signal()

        self.tq = tq
        
        if self.tq is None:
            #try and find a task queue
            try:
                self._checkTQ()
            except:
                pass


    def pushImages(self, image):
        self._checkTQ()
        if debug:
            print('TQ checked')

        if image.dataSource.moduleName == 'HDFDataSource':
            return self.pushImagesHDF(image)
        elif image.dataSource.moduleName == 'TQDataSource':
            return self.pushImagesQueue(image)
        else: #generic catchall for other data sources
            return self.pushImagesDS(image)

    def pushImagesCluster(self, image):
        from PYME.cluster import HTTPRulePusher

        resultsFilename = _verifyClusterResultsFilename(genClusterResultFileName(image.filename))
        logging.debug('Results file: ' + resultsFilename)


        self.resultsMdh = MetaDataHandler.NestedClassMDHandler(self.analysisMDH)
        self.resultsMdh['DataFileID'] = fileID.genDataSourceID(image.dataSource)

        self.pusher = HTTPRulePusher.HTTPRulePusher(dataSourceID=image.filename,
                                                    metadata=self.resultsMdh, resultsFilename=resultsFilename)

        self.queueName = self.pusher.queueID
        self.results_filename = resultsFilename

        debugPrint('Queue created')

        self.onImagesPushed.send(self)
            

    def pushImagesHDF(self, image):
        dataFilename = image.seriesName
        resultsFilename = _verifyResultsFilename(genResultFileName(image.seriesName))
        self.queueName = resultsFilename

        self.tq.createQueue('HDFTaskQueue', self.queueName, dataFilename = dataFilename, resultsFilename=resultsFilename, startAt = 'notYet')

        mdhQ = MetaDataHandler.QueueMDHandler(self.tq, self.queueName, self.analysisMDH)
        mdhQ['DataFileID'] = fileID.genDataSourceID(image.dataSource)

#        evts = self.image.dataSource.getEvents()
#        if len(evts) > 0:
#            self.tq.addQueueEvents(self.image.seriesName, evts)

        self.resultsMdh = mdhQ

        self.tq.releaseTasks(self.queueName, self.analysisMDH['Analysis.StartAt'])

        self.onImagesPushed.send(self)


    def pushImagesQueue(self, image):
        image.mdh.copyEntriesFrom(self.analysisMDH)
        self.queueName = image.seriesName
        self.resultsMdh = image.mdh
        self.resultsMdh['DataFileID'] = fileID.genDataSourceID(image.dataSource)
        self.tq.releaseTasks(self.queueName , self.analysisMDH['Analysis.StartAt'])

        self.onImagesPushed.send(self)

    def pushImagesDS(self, image):
        resultsFilename = _verifyResultsFilename(genResultFileName(image.seriesName))
        self.queueName = resultsFilename
            
        debugPrint('Results file = %s' % resultsFilename) 
        
        self.resultsMdh = MetaDataHandler.NestedClassMDHandler(self.analysisMDH)
        self.resultsMdh['DataFileID'] = fileID.genDataSourceID(image.dataSource)
        
        mn = image.dataSource.moduleName
        #dsID = self.image.seriesName
        #if it's a buffered source, go back to underlying source
        if mn == 'BufferedDataSource':
            mn = image.dataSource.dataSource.moduleName

        self.tq.createQueue('DSTaskQueue', self.queueName, self.resultsMdh, 
                            mn, image.seriesName, 
                            resultsFilename, startAt = self.analysisMDH['Analysis.StartAt'])
        
        evts = image.dataSource.getEvents()
        if len(evts) > 0:
            self.tq.addQueueEvents(self.queueName, evts)
        
        debugPrint('Queue created')

        self.onImagesPushed.send(self)

    def _checkTQ(self):
        def _genURI(taskQueueName):
            try:
                from PYME.misc import hybrid_ns
                ns = hybrid_ns.getNS()
                return ns.resolve(taskQueueName)
            except:
                return 'PYRONAME://' + taskQueueName

        try:
            self.tq.isAlive() 
        except:
            self.tq = None
        
        if self.tq is None:
            import Pyro.core
            from PYME.misc.computerName import GetComputerName
            compName = GetComputerName()
            
            try:
                taskQueueName = 'TaskQueues.%s' % compName   
                self.tq = Pyro.core.getProxyForURI(_genURI(taskQueueName))
            except:
                logger.exception('Error finding task queue, looking for a local queue instead')
                taskQueueName = 'PrivateTaskQueues.%s' % compName
                self.tq = Pyro.core.getProxyForURI('PYRONAME://' + _genURI(taskQueueName))


#############        
class FitDefaults(object):
    def __init__(self, dsviewer, analysisController):
        self.dsviewer = dsviewer
        self.onMetaDataChange = analysisController.onMetaDataChange
        self.analysisMDH = analysisController.analysisMDH

        self.dsviewer.AddMenuItem('Analysis defaults', "Normal 2D analysis", self.OnStandard2D)
        self.dsviewer.AddMenuItem('Analysis defaults', "Calibrating the splitter", self.OnCalibrateSplitter)
        self.dsviewer.AddMenuItem('Analysis defaults', "2D with splitter", self.OnSplitter2D)
        self.dsviewer.AddMenuItem('Analysis defaults', "3D analysis", self.OnStandard3D)
        self.dsviewer.AddMenuItem('Analysis defaults', "3D with splitter", self.OnSplitter3D)
        self.dsviewer.AddMenuItem('Analysis defaults', "PRI", self.OnPRI3D)
        self.dsviewer.AddMenuItem('Analysis defaults', "DNA-PAINT", self.OnDNAPaint)
        self.dsviewer.AddMenuItem('Analysis defaults', "Set Maps from default location", self.OnSetMapsDefault)

    def OnCalibrateSplitter(self, event):
        self.analysisMDH['Analysis.FitModule'] = 'SplitterShiftEstFR'
        self.analysisMDH['Analysis.BGRange'] = (0,0)
        self.analysisMDH['Analysis.SubtractBackground'] = False
        self.analysisMDH['Analysis.DetectionThreshold'] = 2.
        self.analysisMDH['Analysis.ChunkSize'] = 1
        self.analysisMDH['Analysis.ROISize'] = 11

        self.onMetaDataChange.send(self, mdh=self.analysisMDH)

    def OnStandard2D(self, event):
        self.analysisMDH['Analysis.FitModule'] = 'LatGaussFitFR'
        self.analysisMDH['Analysis.BGRange'] = (-30,0)
        self.analysisMDH['Analysis.SubtractBackground'] = True
        self.analysisMDH['Analysis.DetectionThreshold'] = 1.

        self.onMetaDataChange.send(self, mdh=self.analysisMDH)
        
    def OnSplitter2D(self, event):
        self.analysisMDH['Analysis.FitModule'] = 'SplitterFitFNR'
        self.analysisMDH['Analysis.BGRange'] = (-30,0)
        self.analysisMDH['Analysis.SubtractBackground'] = True
        self.analysisMDH['Analysis.DetectionThreshold'] = 1.

        self.onMetaDataChange.send(self, mdh=self.analysisMDH)
    
    def OnStandard3D(self, event):
        self.analysisMDH['Analysis.FitModule'] = 'InterpFitR'
        self.analysisMDH['Analysis.BGRange'] = (-30,0)
        self.analysisMDH['Analysis.SubtractBackground'] = True
        self.analysisMDH['Analysis.DetectionThreshold'] = 1.

        self.onMetaDataChange.send(self, mdh=self.analysisMDH)
    
    def OnSplitter3D(self, event):
        self.analysisMDH['Analysis.FitModule'] = 'SplitterFitInterpNR'
        self.analysisMDH['Analysis.BGRange'] = (-30,0)
        self.analysisMDH['Analysis.SubtractBackground'] = True
        self.analysisMDH['Analysis.DetectionThreshold'] = 1.

        self.onMetaDataChange.send(self, mdh=self.analysisMDH)
        
    def OnPRI3D(self, event):
        self.analysisMDH['PRI.Axis'] = 'y'
        self.analysisMDH['Analysis.EstimatorModule'] = 'priEstimator'
        self.analysisMDH['Analysis.ROISize'] = 9

        self.onMetaDataChange.send(self, mdh=self.analysisMDH)

    def OnDNAPaint(self, event):
        self.analysisMDH['Analysis.BGRange'] = (0,0)
        self.analysisMDH['Analysis.DetectionThreshold'] = 2.

        self.onMetaDataChange.send(self, mdh=self.analysisMDH)

    def OnSetMapsDefault(self, event):
        from PYME.IO.FileUtils.nameUtils import getCalibrationDir
        
        caldir = getCalibrationDir(self.analysisMDH['Camera.SerialNumber'])
        itime = int(1000*self.analysisMDH['Camera.IntegrationTime'])
        darkpath = os.path.join(caldir,'dark_%dms.tif' % (itime))
        varpath = os.path.join(caldir,'variance_%dms.tif' % (itime))
        flatpath = os.path.join(caldir,'flatfield.tif')
        
        if os.path.exists(darkpath):
            self.analysisMDH['Camera.DarkMapID'] = darkpath
        if os.path.exists(varpath):
            self.analysisMDH['Camera.VarianceMapID'] = varpath
        if os.path.exists(flatpath):
            self.analysisMDH['Camera.FlatfieldMapID'] = flatpath

        self.onMetaDataChange.send(self, mdh=self.analysisMDH)

def _check_complete_mdh(mdh):
    en = mdh.getEntryNames()
    return all(x in en for x in ['Camera.TrueEMGain', 'Camera.NoiseFactor', 'Camera.ElectronsPerCount', 'Camera.ReadNoise', 'Camera.ADOffset'])

from ._base import Plugin
class LMAnalyser2(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        
        if not _check_complete_mdh(self.image.mdh):
           logger.warning('Series does not seem to have metadata needed for localization analysis')
            
        self.analysisController = AnalysisController(self.image.mdh)
        self.analysisSettingsView = AnalysisSettingsView(self.dsviewer, self.analysisController, self)

        self._fitdefaults = FitDefaults(self.dsviewer, self.analysisController)

        self.analysisController.onImagesPushed.connect(self.OnImagesPushed)

        self.queueName= None

        if 'fitResults' in dir(self.image):
            self.fitResults = self.image.fitResults
        else:
            self.fitResults = []
        
        if 'resultsMdh' in dir(self.image):
            self.resultsMdh = self.image.resultsMdh

        #a timer object to update for us
        self.timer = mytimer()
        self.timer.Start(5000)

        self.numAnalysed = 0
        self.numEvents = 0

        if self.image.filename.upper().startswith('PYME-CLUSTER:'):
            self.newStyleTaskDistribution = True
        else:
            self.newStyleTaskDistribution = NEW_STYLE_DISTRIBUTION
        
        dsviewer.pipeline = pipeline.Pipeline()
        self.ds = None

        dsviewer.paneHooks.append(self.GenFitStatusPanel)

        dsviewer.updateHooks.append(self.update)
        dsviewer.statusHooks.append(self.GetStatusText)

        if 'bgRange' in dir(self.image.data):
            self.dsviewer.AddMenuItem('View', "SubtractBackground", self.OnToggleBackground)
        
        #if ('auto_start_analysis' in dir(dsviewer)) and dsviewer.auto_start_analysis:
        #    print 'Automatically starting analysis'
        #    wx.CallLater(50,self.OnGo)
            
    def toggle_new_style(self, event=None):
        self.newStyleTaskDistribution = not(self.newStyleTaskDistribution)

    def SetFitInfo(self):
        self.view.pointMode = 'lm'
        mdh = self.analysisController.analysisMDH
        voxx, voxy, _ = mdh.voxelsize_nm
        
        self.view.points = np.vstack((self.fitResults['fitResults']['x0']/voxx, self.fitResults['fitResults']['y0']/voxy, self.fitResults['tIndex'])).T

        if 'Splitter' in mdh.getEntry('Analysis.FitModule'):
            self.view.pointMode = 'splitter'
            if 'BNR' in mdh['Analysis.FitModule']:
                self.view.pointColours = self.fitResults['ratio'] > 0.5
            else:
                self.view.pointColours = self.fitResults['fitResults']['Ag'] > self.fitResults['fitResults']['Ar']
            
        if not 'fitInf' in dir(self):
            self.fitInf = fitInfo.FitInfoPanel(self.dsviewer, self.fitResults, self.resultsMdh, self.do.ds)
            self.dsviewer.AddPage(page=self.fitInf, select=False, caption='Fit Info')
        else:
            self.fitInf.SetResults(self.fitResults, self.resultsMdh)
            
        
    def OnPointSelect(self, xp, yp):
        dist = np.sqrt((xp - self.fitResults['fitResults']['x0'])**2 + (yp - self.fitResults['fitResults']['y0'])**2)
        
        cand = dist.argmin()

        vs = self.image.voxelsize_nm
        self.dsviewer.do.xp = xp/(vs.x)
        self.dsviewer.do.yp = yp/(vs.y)
        self.dsviewer.do.zp = self.fitResults['tIndex'][cand]
        

    def OnToggleBackground(self, event):
        #self.SetMDItems()
        mdh = self.analysisController.analysisMDH
        if self.do.ds.bgRange is None:
            self.do.ds.bgRange = mdh['Analysis.BGRange']#[int(v) for v in self.tBackgroundFrames.GetValue().split(':')]
            self.do.ds.dataStart = mdh['Analysis.StartAt']#int(self.tStartAt.GetValue())
            
            self.do.ds.setBackgroundBufferPCT(mdh['Analysis.PCTBackground'])
        else:
            self.do.ds.bgRange = None
            self.do.ds.dataStart = 0
            
            
        self.do.Optimise()

    def GetStatusText(self):
        return 'Frames Analysed: %d    Events detected: %d' % (self.numAnalysed, self.numEvents)

    def OnTest(self, event):
        #self.SetMDItems()
        #threshold = self.image.mdh['Analysis.DetectionThreshold']
        self.testFrames()
        
    def OnTestFrame(self, event):      
        #self.SetMDItems()
        #threshold = self.image.mdh['Analysis.DetectionThreshold']
        
        ft, fr = self.testFrame()
        
        self.fitResults = fr.results
        self.resultsMdh = self.analysisController.analysisMDH       
        
        self.SetFitInfo()
        
    @property
    def use_cluster(self):
        logger.debug('use_cluster: newStyle=%s, tq=%s' % (self.newStyleTaskDistribution, self.analysisController.tq))
        return self.newStyleTaskDistribution and (self.image.filename.upper().startswith('PYME-CLUSTER') or (self.analysisController.tq is None))

    def OnGo(self, event=None):
        self._checkmap('DarkMapID')
        self._checkmap('VarianceMapID')
        self._checkmap('FlatfieldMapID')
        if self.use_cluster:
            self.analysisController.pushImagesCluster(self.image)
        else:
            self.analysisController.pushImages(self.image)

    def OnImagesPushed(self, **kwargs):
        if debug:
            print('Images pushed')

        self.numAnalysed = 0
        self.numEvents = 0
        self.fitResults = []

        self.queueName = self.analysisController.queueName
        self.resultsMdh = self.analysisController.resultsMdh

        self.timer.WantNotification.append(self.refresh_analysis)
        self.analysisSettingsView.bGo.Enable(False)
        
        self.analysisSettingsView.foldAnalPanes = True
        
        #auto load VisGUI display
        #from PYME.DSView import modules
        #modules.loadModule('LMDisplay', self.dsviewer)
        self.dsviewer.LoadModule('LMDisplay')


    def GenFitStatusPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Fit Status", pinned = True)

        pan = wx.Panel(item, -1)#, size = (160, 200))

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.progPan = progGraph.progPanel(pan, self.fitResults, size=(220, 100))
        self.progPan.draw()

        vsizer.Add(self.progPan, 0,wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 0)

        pan.SetSizer(vsizer)
        vsizer.Fit(pan)

        #_pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 0)
        item.AddNewElement(pan)
        _pnl.AddPane(item)

    @property
    def tq(self):
        #for backwards compatibility
        return self.analysisController.tq

    def refresh_analysis(self):
        if self.use_cluster:
            self._refresh_analysis_cluster()
        else:
            self._refresh_analysis_queue()

        self.progPan.draw()
        self.progPan.Refresh()
        self.dsviewer.Refresh()
        self.dsviewer.update()
            
    def _add_new_results(self, newResults):
        if len(newResults) > 0:
            if len(self.fitResults) == 0:
                self.fitResults = newResults
                self.ds = tabular.FitResultsSource(self.fitResults)
                self.ds.mdh = self.resultsMdh
                self.dsviewer.pipeline.OpenFile(ds=self.ds, imBounds=self.dsviewer.image.imgBounds)
                self.dsviewer.pipeline.mdh = self.resultsMdh
                try:
                    self.dsviewer.LMDisplay.SetFit()
                except:
                    pass
            else:
                self.fitResults = np.concatenate((self.fitResults, newResults))
                self.ds.setResults(self.fitResults)
                self.dsviewer.pipeline.recipe.prune_dependencies_from_namespace(['Localizations', ], True)
                self.dsviewer.pipeline.recipe.invalidate_data()
        
            self.progPan.fitResults = self.fitResults
            self.view.points = np.vstack(
                (self.fitResults['fitResults']['x0'], self.fitResults['fitResults']['y0'], self.fitResults['tIndex'])).T
            self.numEvents = len(self.fitResults)
        
            try:
                self.dsviewer.LMDisplay.RefreshView()
            except:
                pass
    
    def _refresh_analysis_queue(self):
        newNumAnalysed = self.tq.getNumberTasksCompleted(self.queueName)
        if newNumAnalysed > self.numAnalysed:
            self.numAnalysed = newNumAnalysed
            self._add_new_results(self.tq.getQueueData(self.queueName, 'FitResults', len(self.fitResults)))

        if (self.tq.getNumberOpenTasks(self.queueName) + self.tq.getNumberTasksInProgress(self.queueName)) == 0 and 'SpoolingFinished' in self.image.mdh.getEntryNames():
            self.dsviewer.statusbar.SetBackgroundColour(wx.GREEN)
            self.dsviewer.statusbar.Refresh()

    def _refresh_analysis_cluster(self):
        import requests
        from io import BytesIO
        try:
            # import cPickle on py2
            import cPickle as pickle
        except ImportError:
            import pickle
            
        queueInfo = requests.get(self.analysisController.pusher.taskQueueURI + '/distributor/queues').json()['result']
        newNumAnalysed = int(queueInfo[self.analysisController.pusher._ruleID]['tasksCompleted'])
        if newNumAnalysed > self.numAnalysed:
            self.numAnalysed = newNumAnalysed
            
            # load from server as pickle (try this instead of numpy loading to get around np.load slowness if performance is an issue)
            #newResults = pickle.loads(requests.get(self.analysisController.pusher.resultsURI.replace('__aggregate_h5r/', '') + '/FitResults?from=%d' % len(self.fitResults)).content)
            
            # load from server as .npy
            cont = requests.get(self.analysisController.pusher.resultsURI.replace('__aggregate_h5r/', '') + '/FitResults.npy?from=%d' % len(self.fitResults)).content
            newResults = np.load(BytesIO(cont))
            
            self._add_new_results(newResults)
        

    def analysis_refresh_bruteforce(self):
        """Refresh points display from cluster-based analysis. This works in the crudest possible way by simply repeatedly
        re-reading the results file. This should also work for old style analysis, with the same caveats (below) about IO.
        
        WARNING: This has potential performance implications as there wll be significant IO overhead to doing it this way!!!
                 Files should be closed when spooling is complete.
        
        TODO: - Allow incremental update (as is done in refresh_analysis above). Will probably need modification of PYMEDataServer.
              - Stop updates when spooling is complete (HOW? Check file size?)
              - What is the effect of the caches in clusterIO? Do we actually get the updated file?
              - Make sure this only gets called at a reasonable rate.
              - Add logic to call this (and make sure that self.results_filename is defined)
        
        """
        from PYME.IO import unifiedIO
        
        try:
            with unifiedIO.local_or_temp_filename(self.results_filename) as fn: #download a copy of the file if needed, so that we can pass pytables a local filename
                self.dsviewer.pipeline.OpenFile(fn)
                self.numEvents = len(self.dsviewer.pipeline.selectedDataSoure['x'])
                
                #populate the fitResults member TODO - is this actually needed?
                self.fitResults = self.dsviewer.pipeline.selectedDataSoure.resultsSource.fitResults #FIXME - this is really fragile!
                
        except IOError:
            pass
    
        self.progPan.draw()
        self.progPan.Refresh()
        self.dsviewer.Refresh()
        self.dsviewer.update()


    def update(self, dsviewer):
        if 'fitInf' in dir(self) and not self.dsviewer.playbackpanel.tPlay.IsRunning():
            try:
                self.fitInf.UpdateDisp(self.view.PointsHitTest())
            except:
                import traceback
                print((traceback.format_exc()))

    # def testFrames(self, detThresh = 0.9, offset = 0):
    #     from pylab import *
    #     close('all')
    #     if self.image.dataSource.moduleName == 'TQDataSource':
    #         self.checkTQ()
    #     matplotlib.interactive(False)
    #     clf()
    #     sq = min(self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 1000, self.image.dataSource.getNumSlices()/4)
    #     zps = array(range(self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 20, self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 24)  + range(sq, sq + 4) + range(self.image.dataSource.getNumSlices()/2,self.image.dataSource.getNumSlices() /2+4))
    #     zps += offset
    #     fitMod = self.fitFactories[self.cFitType.GetSelection()]
    #     #bgFrames = int(tBackgroundFrames.GetValue())
    #     bgFrames = [int(v) for v in self.tBackgroundFrames.GetValue().split(':')]
    #     print(zps)
    #     for i in range(12):
    #         print(i)
    #         #if 'Analysis.NumBGFrames' in md.getEntryNames():
    #         #bgi = range(max(zps[i] - bgFrames,mdh.getEntry('EstimatedLaserOnFrameNo')), zps[i])
    #         bgi = range(max(zps[i] + bgFrames[0],self.image.mdh.getEntry('EstimatedLaserOnFrameNo')), max(zps[i] + bgFrames[1],self.image.mdh.getEntry('EstimatedLaserOnFrameNo')))
    #         #else:
    #         #    bgi = range(max(zps[i] - 10,md.EstimatedLaserOnFrameNo), zps[i])
    #         mn = self.image.dataSource.moduleName
    #         if mn == 'BufferedDataSource':
    #             mn = self.image.dataSource.dataSource.moduleName

    #         if 'Splitter' in fitMod:
    #             ft = remFitBuf.fitTask(self.image.seriesName, zps[i], detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'SplitterObjFindR', bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
    #         else:
    #             ft = remFitBuf.fitTask(self.image.seriesName, zps[i], detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'LatObjFindFR', bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
    #         res = ft(taskQueue=self.tq)
    #         xp = floor(i/4)/3.
    #         yp = (3 - i%4)/4.
    #         #print xp, yp
    #         axes((xp,yp, 1./6,1./4.5))
    #         #d = ds[zps[i], :,:].squeeze().T
    #         d = self.image.dataSource.getSlice(zps[i]).T
    #         imshow(d, cmap=cm.hot, interpolation='nearest', hold=False, clim=(median(d.ravel()), d.max()))
    #         title('Frame %d' % zps[i])
    #         xlim(0, d.shape[1])
    #         ylim(0, d.shape[0])
    #         xticks([])
    #         yticks([])
    #         #print 'i = %d, ft.index = %d' % (i, ft.index)
    #         #subplot(4,6,2*i+13)
    #         xp += 1./6
    #         axes((xp,yp, 1./6,1./4.5))
    #         d = ft.ofd.filteredData.T
    #         #d = ft.data.squeeze().T
    #         imshow(d, cmap=cm.hot, interpolation='nearest', hold=False, clim=(median(d.ravel()), d.max()))
    #         plot([p.x for p in ft.ofd], [p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
    #         if ft.driftEst:
    #              plot([p.x for p in ft.ofdDr], [p.y for p in ft.ofdDr], 'o', mew=2, mec='b', mfc='none', ms=9)
    #         if ft.fitModule in remFitBuf.splitterFitModules:
    #                 plot([p.x for p in ft.ofd], [d.shape[0] - p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
    #         axis('tight')
    #         xlim(0, d.shape[1])
    #         ylim(0, d.shape[0])
    #         xticks([])
    #         yticks([])
    #     show()
    #     matplotlib.interactive(True)
        
    def _checkmap(self,mapid):
        maptype = 'Camera.%s' % mapid
        # check if requested maps can be read
        # the below test checks for non-empty string
        analysisMDH = self.analysisController.analysisMDH
        if analysisMDH.getOrDefault(maptype,''):
            logging.debug("trying to read map")
            try:
                md = remFitBuf.cameraMaps._getMap(analysisMDH,
                                              analysisMDH[maptype])
            except:
                md = None

            if md is None:
                logging.warn("CANNOT FIND %s at %s" % (maptype, analysisMDH[maptype]))
                analysisMDH[maptype] = '' # unset the map ID in this case

    def testFrame(self, gui=True):
        #from pylab import *
        import matplotlib
        from matplotlib import pyplot as plt
        #close('all')
        if self.image.dataSource.moduleName == 'TQDataSource':
            self.analysisController._checkTQ()
        if gui:    
            matplotlib.interactive(False)
            plt.figure()
        
        zp = self.do.zp

        analysisMDH = self.analysisController.analysisMDH
        
        self._checkmap('DarkMapID')
        self._checkmap('VarianceMapID')
        self._checkmap('FlatfieldMapID')

        mn = self.image.dataSource.moduleName
        if mn == 'BufferedDataSource':
            mn = self.image.dataSource.dataSource.moduleName

        ft = remFitBuf.fitTask(dataSourceID=self.image.seriesName, frameIndex=zp, metadata=MetaDataHandler.NestedClassMDHandler(analysisMDH), dataSourceModule=mn)
        res = ft(gui=gui,taskQueue=self.tq)
        
        if gui:
            plt.figure()
            try:
                d = ft.ofd.filteredData.T
                #d = ft.data.squeeze().T
                plt.imshow(d, cmap=plt.cm.hot, interpolation='nearest', clim=(np.median(d.ravel()), d.max()))
                plt.plot([p.x for p in ft.ofd], [p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
                if ft.driftEst:
                    plt.plot([p.x for p in ft.ofdDr], [p.y for p in ft.ofdDr], 'o', mew=2, mec='b', mfc='none', ms=9)
                #if ft.fitModule in remFitBuf.splitterFitModules:
                #        plot([p.x for p in ft.ofd], [d.shape[0] - p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
                #axis('tight')
                plt.xlim(0, d.shape[1])
                plt.ylim(d.shape[0], 0)
                plt.xticks([])
                plt.yticks([])
                
                
                    
                vx, vy, _ = self.image.voxelsize_nm
                
                plt.plot(res.results['fitResults']['x0']/vx, res.results['fitResults']['y0']/vy, '+b', mew=2)
                
                if 'startParams' in res.results.dtype.names:
                    plt.plot(res.results['startParams']['x0']/vx, res.results['startParams']['y0']/vy, 'xc', mew=2)
                
                if 'tIm' in dir(ft.ofd):
                    plt.figure()
                    plt.imshow(ft.ofd.tIm.T, cmap=plt.cm.hot, interpolation='nearest')
                    #axis('tight')
                    plt.xlim(0, d.shape[1])
                    plt.ylim(d.shape[0], 0)
                    plt.xticks([])
                    plt.yticks([])
                    plt.plot(res.results['fitResults']['x0']/vx, res.results['fitResults']['y0']/vy, '+b')
                    
                #figure()
                #imshow()
            except AttributeError:
                #d = self.image.data[:,:,zp].squeeze().T
                if isinstance(ft.bg, np.ndarray):
                    # We expect our background estimate to take the form of a numpy array, correct our data by subtracting the background
                    d = (ft.data - ft.bg).squeeze().T
                elif hasattr(ft.bg, 'get_background'):
                    # To support GPU background calculation and to minimize the number of CPU-GPU transfers, our background estimate can instead be
                    # a proxy for a background buffer on the GPU. During fitting this will usually be accessed directly from the GPU fitting code, but in
                    # this case (when displaying the data we fitted) we need to get at it from our CPU code using the `get_background` method of the proxy.
                    # 
                    # NB - the background returned by `get_background()` will not have been flatfielded. As this is just display code we ignore this,
                    # but any computational code should perform flatfielding correction on the results of the get_background call.
                    # 
                    # TODO - document GPU background interface via creating a base class in PYME.IO.buffers
                    d = (ft.data - ft.bg.get_background().reshape(ft.data.shape)).squeeze().T
                else:
                    raise RuntimeError('Background format not understood')
                    
                plt.imshow(d, cmap=plt.cm.jet, interpolation='nearest', clim = [0, d.max()])
                plt.xlim(0, d.shape[1])
                plt.ylim(d.shape[0], 0)
                
                vx, vy, _ = self.image.voxelsize_nm
                
                plt.plot(res.results['fitResults']['x0']/vx, res.results['fitResults']['y0']/vy, 'ow')
                pass

            plt.show()
    
            matplotlib.interactive(True)
        
        return ft, res



def Plug(dsviewer):
    LMAnalyser = LMAnalyser2(dsviewer)
    dsviewer.create_overlay_panel()
    
    return {'LMAnalyser':LMAnalyser}

