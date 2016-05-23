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
import wx
import Pyro.core
from PYME.Analysis import remFitBuf
import os
from PYME.Acquire import MetaDataHandler
import PYME.Analysis.FitFactories

from PYME.Analysis import MetaDataEdit as mde
#from pylab import *
from PYME.io.FileUtils import fileID
from PYME.io.FileUtils.nameUtils import genResultFileName
from PYME.LMVis import progGraph as progGraph

#from PYME.LMVis import gl_render
#from PYME.LMVis import workspaceTree
#import sys


from PYME.misc import extraCMaps


import numpy as np

from PYME.LMVis import pipeline, inpFilt

import numpy
import pylab

import PYME.misc.autoFoldPanel as afp
from PYME.Acquire.mytimer import mytimer
from PYME.DSView import fitInfo
from PYME.DSView.OverlaysPanel import OverlayPanel
import wx.lib.agw.aui as aui

debug = True

def debugPrint(msg):
    if debug:
        print(msg)
        



class LMAnalyser:
    FINDING_PARAMS = [#mde.FloatParam('Analysis.DetectionThreshold', 'Thresh:', 1.0),
                      mde.IntParam('Analysis.DebounceRadius', 'Debounce rad:', 4),
    ]
    
    DEFAULT_PARAMS = [#mde.ChoiceParam('Analysis.InterpModule','Interp:','LinearInterpolator', choices=Interpolators.interpolatorList, choiceNames=Interpolators.interpolatorDisplayList),
                      #mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
                      #mde.FilenameParam('PSFFilename', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf'),
                      #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
                      #mde.FloatParam('Analysis.AxialShift', 'Z Shift [nm]:', 0),
                      mde.BoolFloatParam('Analysis.PCTBackground' , 'Use percentile for background', default=False, helpText='', ondefault=0.25, offvalue=0),
                      mde.FilenameParam('Camera.VarianceMapID', 'Variance Map:', prompt='Please select variance map to use ...', wildcard='TIFF Files|*.tif', filename=''),
                      mde.FilenameParam('Camera.DarkMapID', 'Dark Map:', prompt='Please select dark map to use ...', wildcard='TIFF Files|*.tif', filename=''),
                      mde.FilenameParam('Camera.FlatfieldMapID', 'Flatfiled Map:', prompt='Please select flatfield map to use ...', wildcard='TIFF Files|*.tif', filename=''),
                      mde.BoolParam('Analysis.TrackFiducials', 'Track Fiducials', default=False),
                      mde.FloatParam('Analysis.FiducialThreshold', 'Fiducial Threshold', default=1.8),
    ]
    
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        if 'tq' in dir(dsviewer):
            self.tq = dsviewer.tq
        else:
            self.tq = None
        
        self.image = dsviewer.image
        self.view = dsviewer.view
        self.do = dsviewer.do
        
        self.queueName= None

        #this should only occur for files types which we weren't expecting to process
        #as LM data (eg tiffs)
        if not 'EstimatedLaserOnFrameNo' in self.image.mdh.getEntryNames():
            from PYME.Analysis import MetaData
            #try:
            MetaData.fillInBlanks(self.image.mdh, self.image.dataSource)
            #except IndexError:
            #    pass
            

        if 'fitResults' in dir(self.image):
            self.fitResults = self.image.fitResults
        else:
            self.fitResults = []
        
        if 'resultsMdh' in dir(self.image):
            self.resultsMdh = self.image.resultsMdh

        mTasks = wx.Menu()
        TASKS_STANDARD_2D = wx.NewId()
        TASKS_CALIBRATE_SPLITTER = wx.NewId()
        TASKS_2D_SPLITTER = wx.NewId()
        TASKS_3D = wx.NewId()
        TASKS_3D_SPLITTER = wx.NewId()
        TASKS_PRI = wx.NewId()
        mTasks.Append(TASKS_STANDARD_2D, "Normal 2D analysis", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_CALIBRATE_SPLITTER, "Calibrating the splitter", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_2D_SPLITTER, "2D with splitter", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_3D, "3D analysis", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_3D_SPLITTER, "3D with splitter", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_PRI, "PRI", "", wx.ITEM_NORMAL)
        self.dsviewer.menubar.Append(mTasks, "Set defaults for")
        
        wx.EVT_MENU(self.dsviewer, TASKS_CALIBRATE_SPLITTER, self.OnCalibrateSplitter)
        wx.EVT_MENU(self.dsviewer, TASKS_STANDARD_2D, self.OnStandard2D)
        wx.EVT_MENU(self.dsviewer, TASKS_2D_SPLITTER, self.OnSpitter2D)
        wx.EVT_MENU(self.dsviewer, TASKS_3D, self.OnStandard3D)
        wx.EVT_MENU(self.dsviewer, TASKS_3D_SPLITTER, self.OnSpliter3D)
        wx.EVT_MENU(self.dsviewer, TASKS_PRI, self.OnPRI3D)
        
        BG_SUBTRACT = wx.NewId()
        self.dsviewer.view_menu.AppendCheckItem(BG_SUBTRACT, 'Subtract Background')
        wx.EVT_MENU(self.dsviewer, BG_SUBTRACT, self.OnToggleBackground)
        

        #a timer object to update for us
        self.timer = mytimer()
        self.timer.Start(1000)

        self.analDispMode = 'z'

        self.numAnalysed = 0
        self.numEvents = 0
        
        dsviewer.pipeline = pipeline.Pipeline()
        self.ds = None
        
        self.foldAnalPanes = False

        dsviewer.paneHooks.append(self.GenPointFindingPanel)
        dsviewer.paneHooks.append(self.GenAnalysisPanel)
        dsviewer.paneHooks.append(self.GenFitStatusPanel)

        dsviewer.updateHooks.append(self.update)
        dsviewer.statusHooks.append(self.GetStatusText)

        if 'Protocol.DataStartsAt' in self.image.mdh.getEntryNames():
            self.do.zp = self.image.mdh.getEntry('Protocol.DataStartsAt')
        else:
            self.do.zp = self.image.mdh.getEntry('EstimatedLaserOnFrameNo')
        
        #if (len(self.fitResults) > 0) and not 'PYME_BUGGYOPENGL' in os.environ.keys():
        #    self.GenResultsView()

#    def GenResultsView(self):
#        voxx = 1e3*self.image.mdh.getEntry('voxelsize.x')
#        voxy = 1e3*self.image.mdh.getEntry('voxelsize.y')
#        
#        self.SetFitInfo()
#
#        from PYME.LMVis import gl_render
#        self.glCanvas = gl_render.LMGLCanvas(self.dsviewer, False, vp = self.do, vpVoxSize = voxx)
#        self.glCanvas.cmap = pylab.cm.gist_rainbow
#        self.glCanvas.pointSelectionCallbacks.append(self.OnPointSelect)
#
#        self.dsviewer.AddPage(page=self.glCanvas, select=True, caption='VisLite')
#
#        xsc = self.image.data.shape[0]*1.0e3*self.image.mdh.getEntry('voxelsize.x')/self.glCanvas.Size[0]
#        ysc = self.image.data.shape[1]*1.0e3*self.image.mdh.getEntry('voxelsize.y')/ self.glCanvas.Size[1]
#
#        if xsc > ysc:
#            self.glCanvas.setView(0, xsc*self.glCanvas.Size[0], 0, xsc*self.glCanvas.Size[1])
#        else:
#            self.glCanvas.setView(0, ysc*self.glCanvas.Size[0], 0, ysc*self.glCanvas.Size[1])
#
#        #we have to wait for the gui to be there before we start changing stuff in the GL view
#        #self.timer.WantNotification.append(self.AddPointsToVis)
#
#        self.glCanvas.Bind(wx.EVT_IDLE, self.OnIdle)
#        self.pointsAdded = False
        
    def SetFitInfo(self):
        self.view.pointMode = 'lm'
        voxx = 1e3*self.image.mdh.getEntry('voxelsize.x')
        voxy = 1e3*self.image.mdh.getEntry('voxelsize.y')
        self.view.points = numpy.vstack((self.fitResults['fitResults']['x0']/voxx, self.fitResults['fitResults']['y0']/voxy, self.fitResults['tIndex'])).T

        if 'Splitter' in self.image.mdh.getEntry('Analysis.FitModule'):
            self.view.pointMode = 'splitter'
            if 'BNR' in self.image.mdh['Analysis.FitModule']:
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
        
        self.dsviewer.do.xp = xp/(1.0e3*self.image.mdh.getEntry('voxelsize.x'))
        self.dsviewer.do.yp = yp/(1.0e3*self.image.mdh.getEntry('voxelsize.y'))
        self.dsviewer.do.zp = self.fitResults['tIndex'][cand]
        

    def OnIdle(self,event):
        if not self.pointsAdded:
            self.pointsAdded = True

            self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],self.fitResults['tIndex'].astype('f'))
            self.glCanvas.setCLim((0, self.fitResults['tIndex'].max()))

    def OnToggleBackground(self, event):
        self.SetMDItems()
        if self.do.ds.bgRange == None:
            self.do.ds.bgRange = [int(v) for v in self.tBackgroundFrames.GetValue().split(':')]
            self.do.ds.dataStart = int(self.tStartAt.GetValue())
            
            self.do.ds.setBackgroundBufferPCT(self.image.mdh['Analysis.PCTBackground'])
        else:
            self.do.ds.bgRange = None
            self.do.ds.dataStart = 0
            
            
        self.do.Optimise()

#    def AddPointsToVis(self):
#        self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],self.fitResults['tIndex'].astype('f'))
#        self.glCanvas.setCLim((0, self.fitResults['tIndex'].max()))
#
#        self.timer.WantNotification.remove(self.AddPointsToVis)

    def GetStatusText(self):
        return 'Frames Analysed: %d    Events detected: %d' % (self.numAnalysed, self.numEvents)
        
    def _populateStdOptionsPanel(self, pan, vsizer):
        for param in self.DEFAULT_PARAMS:
            pg = param.createGUI(pan, self.image.mdh)
            vsizer.Add(pg, 0,wx.BOTTOM|wx.EXPAND, 5)
        vsizer.Fit(pan)
            
    def _populateFindOptionsPanel(self, pan, vsizer):
        for param in self.FINDING_PARAMS:
            pg = param.createGUI(pan, self.image.mdh)
            vsizer.Add(pg, 0,wx.BOTTOM|wx.EXPAND, 5)
        #vsizer.Fit(pan)
    
        
    def _populateCustomAnalysisPanel(self, pan, vsizer):
        try:
            fitMod = self.fitFactories[self.cFitType.GetSelection()]
            fm = __import__('PYME.Analysis.FitFactories.' + fitMod, fromlist=['PYME', 'Analysis', 'FitFactories'])
            
            #vsizer = wx.BoxSizer(wx.VERTICAL)
            for param in fm.PARAMETERS:
                pg = param.createGUI(pan, self.image.mdh)
                vsizer.Add(pg, 0,wx.BOTTOM|wx.EXPAND, 5)
            vsizer.Fit(pan)
                
        except AttributeError:
            pass
        
    def OnFitModuleChanged(self, event):
        self.customOptionsSizer.Clear(True)
        self._populateCustomAnalysisPanel(self.customOptionsPan, self.customOptionsSizer)
        

    def GenAnalysisPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Analysis", pinned = not(self.foldAnalPanes))
        
        ##############################
        pan = wx.Panel(item, -1)
        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Start at:'), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        if 'Protocol.DataStartsAt' in self.image.mdh.getEntryNames():
            startAt = self.image.mdh.getEntry('Protocol.DataStartsAt')
        else:
            startAt = self.do.zp
        self.tStartAt = wx.TextCtrl(pan, -1, value='%d' % startAt, size=(50, -1))

        hsizer.Add(self.tStartAt, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Background:'), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tBackgroundFrames = wx.TextCtrl(pan, -1, value='-30:0', size=(50, -1))
        self.tBackgroundFrames.SetValue('%d:%d'% tuple(self.image.mdh.getOrDefault('Analysis.BGRange', [-30,0])))

        hsizer.Add(self.tBackgroundFrames, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)
        
        pan.SetSizer(vsizer)
        vsizer.Fit(pan)
        item.AddNewElement(pan)

        self.cbSubtractBackground = wx.CheckBox(item, -1, 'Subtract background in fit')
        self.cbSubtractBackground.SetValue(self.image.mdh.getOrDefault('Analysis.subtractBackground', True))


        item.AddNewElement(self.cbSubtractBackground)

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
        self.fitFactories = PYME.Analysis.FitFactories.resFitFactories
        print((self.fitFactories))

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Type:'), 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.cFitType = wx.Choice(pan, -1, choices = ['{:<35} \t- {:} '.format(f, PYME.Analysis.FitFactories.useFor[f]) for f in self.fitFactories], size=(110, -1))
        
        if 'Analysis.FitModule' in self.image.mdh.getEntryNames():
            #has already been analysed - most likely to want the same method again
            try:
                self.cFitType.SetSelection(self.fitFactories.index(self.image.mdh['Analysis.FitModule']))
                self.tThreshold.SetValue('%s' % self.image.mdh.getOrDefault('Analysis.DetectionThreshold', 1))
            except ValueError:
                self.cFitType.SetSelection(self.fitFactories.index('LatGaussFitFR'))
                
        #elif 'Camera.ROIPosY' in self.image.mdh.getEntryNames() and (self.image.mdh.getEntry('Camera.ROIHeight') + 1 + 2*(self.image.mdh.getEntry('Camera.ROIPosY')-1)) == 512:
        #    #we have a symetrical ROI about the centre - most likely want to analyse using splitter
        #    self.cFitType.SetSelection(self.fitFactories.index('SplitterFitQR'))
        #    self.tThreshold.SetValue('0.5')
        else:
            self.cFitType.SetSelection(self.fitFactories.index('LatGaussFitFR'))
            
        self.cFitType.Bind(wx.EVT_CHOICE, self.OnFitModuleChanged)

        hsizer.Add(self.cFitType, 1,wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)

        pan.SetSizer(vsizer)
        vsizer.Fit(pan)

        item.AddNewElement(pan)

#        self.cbDrift = wx.CheckBox(item, -1, 'Estimate Drift')
#        self.cbDrift.SetValue(False)
#
#        #_pnl.AddFoldPanelWindow(item, self.cbDrift, fpb.FPB_ALIGN_WIDTH, 7, 5)
#        item.AddNewElement(self.cbDrift)


        ########################################        
        #custom  (fit factory dependant) options        
        self.customOptionsPan = wx.Panel(item, -1)
        self.customOptionsSizer = wx.BoxSizer(wx.VERTICAL)
        self.customOptionsPan.SetSizer(self.customOptionsSizer)

        self._populateCustomAnalysisPanel(self.customOptionsPan, self.customOptionsSizer)        

        item.AddNewElement(self.customOptionsPan)
        
        ######################
        #Go
        self.bGo = wx.Button(item, -1, 'Go')


        self.bGo.Bind(wx.EVT_BUTTON, self.OnGo)
        item.AddNewElement(self.bGo)
        _pnl.AddPane(item)

        self.analysisPanel = item



            
    def SetMDItems(self):
        self.image.mdh.setEntry('Analysis.subtractBackground', self.cbSubtractBackground.GetValue())
        
        
        for param in self.DEFAULT_PARAMS:
            param.retrieveValue(self.image.mdh)
            
        for param in self.FINDING_PARAMS:
            param.retrieveValue(self.image.mdh)
            
            
        fitMod = self.fitFactories[self.cFitType.GetSelection()]
        fm = __import__('PYME.Analysis.FitFactories.' + fitMod, fromlist=['PYME', 'Analysis', 'FitFactories'])
        
        try: 
            plist = fm.PARAMETERS
        except AttributeError:
            plist = []     
        
        for param in plist:
            param.retrieveValue(self.image.mdh)
            


    def OnGo(self, event):
        threshold = float(self.tThreshold.GetValue())
        startAt = int(self.tStartAt.GetValue())
        driftEst = False#self.cbDrift.GetValue()
        fitMod = self.fitFactories[self.cFitType.GetSelection()]
        #interpolator = self.interpolators[self.cInterpType.GetSelection()]
        bgFrames = [int(v) for v in self.tBackgroundFrames.GetValue().split(':')]
        self.image.mdh.setEntry('Analysis.BGRange', bgFrames)
        
        
        self.SetMDItems()
        

        #self.image.mdh.setEntry('Analysis.InterpModule', interpolator)

            
        if debug:
            print('About to push images')

        if not driftEst:
            self.pushImages(startAt, threshold, fitMod)
        else:
            self.pushImagesD(startAt, threshold)
            
        if debug:
            print('Images pushed')

        #############
        #set up real time display
#        if not 'glCanvas' in dir(self):   #re-use existing canvas if present     
#            from PYME.LMVis import gl_render
#            self.glCanvas = gl_render.LMGLCanvas(self.dsviewer, False)
#            self.glCanvas.cmap = pylab.cm.gist_rainbow
#    
#            self.dsviewer.AddPage(page=self.glCanvas, select=True, caption='VisLite')
        

#        xsc = self.image.data.shape[0]*1.0e3*self.image.mdh.getEntry('voxelsize.x')/self.glCanvas.Size[0]
#        ysc = self.image.data.shape[1]*1.0e3*self.image.mdh.getEntry('voxelsize.y')/ self.glCanvas.Size[1]
#
#        if xsc > ysc:
#            self.glCanvas.setView(0, xsc*self.glCanvas.Size[0], 0, xsc*self.glCanvas.Size[1])
#        else:
#            self.glCanvas.setView(0, ysc*self.glCanvas.Size[0], 0, ysc*self.glCanvas.Size[1])

        self.numAnalysed = 0
        self.numEvents = 0
        self.fitResults = []

        self.timer.WantNotification.append(self.analRefresh)
        self.bGo.Enable(False)
        
        self.foldAnalPanes = True
        
        #auto load VisGUI display
        #from PYME.DSView import modules
        #modules.loadModule('LMDisplay', self.dsviewer)
        self.dsviewer.LoadModule('LMDisplay')
        
        #_pnl.Collapse(self.analysisPanel)

    def GenPointFindingPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Point Finding", pinned = not(self.foldAnalPanes))
#        item = _pnl.AddFoldPanel("Point Finding", collapsed=False,
#                                      foldIcons=self.Images)

        pan = wx.Panel(item, -1)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Threshold:'), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tThreshold = wx.TextCtrl(pan, -1, value='1.0', size=(40, -1))

        hsizer.Add(self.tThreshold, 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)
        
        #load any additional point finding settings
        self._populateFindOptionsPanel(pan, vsizer)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        bTest = wx.Button(pan, -1, 'Test', style=wx.BU_EXACTFIT)
        bTest.Bind(wx.EVT_BUTTON, self.OnTest)
        hsizer.Add(bTest, 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
        
        bTestF = wx.Button(pan, -1, 'Test This Frame', style=wx.BU_EXACTFIT)
        bTestF.Bind(wx.EVT_BUTTON, self.OnTestFrame)
        hsizer.Add(bTestF, 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
        
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)
        
        pan.SetSizerAndFit(vsizer)

        item.AddNewElement(pan)
        _pnl.AddPane(item)


    def OnTest(self, event):
        threshold = float(self.tThreshold.GetValue())
        self.SetMDItems()
        self.testFrames(threshold)
        
    def OnTestFrame(self, event):
        threshold = float(self.tThreshold.GetValue())        
        self.SetMDItems()
        
        ft, fr = self.testFrame(threshold)
        
        self.fitResults = fr.results
        self.resultsMdh = self.image.mdh       
        
        self.SetFitInfo()

    def GenFitStatusPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Fit Status", pinned = True)

        pan = wx.Panel(item, -1)#, size = (160, 200))

        vsizer = wx.BoxSizer(wx.VERTICAL)

#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#        hsizer.Add(wx.StaticText(pan, -1, 'Colour:'), 0, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
#
#        self.chProgDispColour = wx.Choice(pan, -1, choices = ['z', 'gFrac', 't'], size=(60, -1))
#        self.chProgDispColour.Bind(wx.EVT_CHOICE, self.OnProgDispColourChange)
#        hsizer.Add(self.chProgDispColour, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
#
#        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)
#
#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#        hsizer.Add(wx.StaticText(pan, -1, 'CMap:'), 0, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
#
#        self.chProgDispCMap = wx.Choice(pan, -1, choices = ['gist_rainbow', 'RdYlGn'], size=(60, -1))
#        self.chProgDispCMap.Bind(wx.EVT_CHOICE, self.OnProgDispCMapChange)
#        hsizer.Add(self.chProgDispCMap, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
#
#        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 7)

        self.progPan = progGraph.progPanel(pan, self.fitResults, size=(220, 100))
        self.progPan.draw()

        vsizer.Add(self.progPan, 0,wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 0)

        pan.SetSizer(vsizer)
        vsizer.Fit(pan)

        #_pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 0)
        item.AddNewElement(pan)
        _pnl.AddPane(item)

    def OnProgDispColourChange(self, event):
        #print 'foo'
        self.analDispMode = self.chProgDispColour.GetStringSelection()
        self.analRefresh()

    def OnProgDispCMapChange(self, event):
        #print 'foo'
        self.glCanvas.setCMap(pylab.cm.__getattribute__(self.chProgDispCMap.GetStringSelection()))

    def OnCalibrateSplitter(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('SplitterShiftEstFR'))
        self.tBackgroundFrames.SetValue('0:0')
        self.cbSubtractBackground.SetValue(False)
        self.tThreshold.SetValue('2')
        self.image.mdh['Analysis.ChunkSize'] = 1
        self.image.mdh['Analysis.ROISize'] = 11

    def OnStandard2D(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('LatGaussFitFR'))
        self.tBackgroundFrames.SetValue('-30:0')
        self.cbSubtractBackground.SetValue(True)
        self.tThreshold.SetValue('0.6')
        
    def OnSpitter2D(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('SplitterFitQR'))
        self.tBackgroundFrames.SetValue('-30:0')
        self.cbSubtractBackground.SetValue(True)
        self.tThreshold.SetValue('0.5')
    
    def OnStandard3D(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('InterpFitR'))
        self.tBackgroundFrames.SetValue('-30:0')
        self.cbSubtractBackground.SetValue(True)
        self.tThreshold.SetValue('1.0')
    
    def OnSpliter3D(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('SplitterFitInterpR'))
        self.tBackgroundFrames.SetValue('-30:0')
        self.cbSubtractBackground.SetValue(True)
        self.tThreshold.SetValue('1.0')
        
    def OnPRI3D(self, event):
        #self.cFitType.SetSelection(self.fitFactories.index('SplitterFitInterpR'))
        #self.tBackgroundFrames.SetValue('-30:0')
        #self.cbSubtractBackground.SetValue(True)
        #self.tThreshold.SetValue('1.0')
        self.image.mdh['PRI.Axis'] = 'y'
        self.image.mdh['Analysis.EstimatorModule'] = 'priEstimator'

    def analRefresh(self):
        newNumAnalysed = self.tq.getNumberTasksCompleted(self.queueName)
        if newNumAnalysed > self.numAnalysed:
            self.numAnalysed = newNumAnalysed
            newResults = self.tq.getQueueData(self.queueName, 'FitResults', len(self.fitResults))
            if len(newResults) > 0:
                if len(self.fitResults) == 0:
                    self.fitResults = newResults
                    self.ds = inpFilt.fitResultsSource(self.fitResults)
                    self.dsviewer.pipeline.OpenFile(ds=self.ds, imBounds = self.dsviewer.image.imgBounds)
                    self.dsviewer.pipeline.mdh = self.resultsMdh
                    try:
                        self.dsviewer.LMDisplay.SetFit()
                    except:
                        pass
                else:
                    self.fitResults = numpy.concatenate((self.fitResults, newResults))
                    self.ds.setResults(self.fitResults)
                    self.dsviewer.pipeline.Rebuild()
                    
                
                self.progPan.fitResults = self.fitResults

                self.view.points = numpy.vstack((self.fitResults['fitResults']['x0'], self.fitResults['fitResults']['y0'], self.fitResults['tIndex'])).T

                self.numEvents = len(self.fitResults)
                
                try:
                    self.dsviewer.LMDisplay.RefreshView()
                except:
                    pass

        if (self.tq.getNumberOpenTasks(self.queueName) + self.tq.getNumberTasksInProgress(self.queueName)) == 0 and 'SpoolingFinished' in self.image.mdh.getEntryNames():
            self.dsviewer.statusbar.SetBackgroundColour(wx.GREEN)
            self.dsviewer.statusbar.Refresh()

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


    #from fth5.py
    def checkTQ(self):
            
        try:
            #if 'PYME_TASKQUEUENAME' in os.environ.keys():
            #    taskQueueName = os.environ['PYME_TASKQUEUENAME']
            #else:
            #    taskQueueName = 'taskQueue'
            self.tq.isAlive()
        
        except:
            self.tq = None
        
        if self.tq == None:
            from PYME.misc.computerName import GetComputerName
            compName = GetComputerName()
            
            try:
                taskQueueName = 'TaskQueues.%s' % compName
                
                try:
                    from PYME.misc import pyme_zeroconf 
                    ns = pyme_zeroconf.getNS()
                    URI = ns.resolve(taskQueueName)
                except:
                    URI = 'PYRONAME://' + taskQueueName
            
                self.tq = Pyro.core.getProxyForURI(URI)
            except:
                taskQueueName = 'PrivateTaskQueues.%s' % compName

                self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)


    def pushImages(self, startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
        self.checkTQ()
        if debug:
            print('TQ checked')
        if self.image.dataSource.moduleName == 'HDFDataSource':
            self.pushImagesHDF(startingAt, detThresh, fitFcn)
        elif self.image.dataSource.moduleName == 'TQDataSource':
            self.pushImagesQueue(startingAt, detThresh, fitFcn)
        else: #generic catchall for other data sources
            self.pushImagesDS(startingAt, detThresh, fitFcn)

    def _verifyResultsFilename(self, resultsFilename):
        if os.path.exists(resultsFilename):
            di, fn = os.path.split(resultsFilename)
            i = 1
            stub = os.path.splitext(fn)[0]
            while os.path.exists(os.path.join(di, stub + '_%d.h5r' % i)):
                i += 1
            fdialog = wx.FileDialog(None, 'Analysis file already exists, please select a new filename',
                        wildcard='H5R files|*.h5r', defaultDir=di, defaultFile=stub + '_%d.h5r' % i, style=wx.SAVE)
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                resultsFilename = fdialog.GetPath().encode()
            else:
                raise RuntimeError('Invalid results file - not running')
                
        return resultsFilename
            

    def pushImagesHDF(self, startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
        dataFilename = self.image.seriesName
        resultsFilename = self._verifyResultsFilename(genResultFileName(self.image.seriesName))

        #self.image.seriesName = resultsFilename
        self.queueName = resultsFilename

        self.tq.createQueue('HDFTaskQueue', self.queueName, dataFilename = dataFilename, resultsFilename=resultsFilename, startAt = 'notYet')

        mdhQ = MetaDataHandler.QueueMDHandler(self.tq, self.queueName, self.image.mdh)
        mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
        mdhQ.setEntry('Analysis.FitModule', fitFcn)
        mdhQ.setEntry('Analysis.DataFileID', fileID.genDataSourceID(self.image.dataSource))

#        evts = self.image.dataSource.getEvents()
#        if len(evts) > 0:
#            self.tq.addQueueEvents(self.image.seriesName, evts)

        self.resultsMdh = mdhQ

        self.tq.releaseTasks(self.queueName, startingAt)


    def pushImagesQueue(self, startingAt=0, detThresh = .9, fitFcn='LatGaussFitFR'):
        self.image.mdh.setEntry('Analysis.DetectionThreshold', detThresh)
        self.image.mdh.setEntry('Analysis.FitModule', fitFcn)
        self.image.mdh.setEntry('Analysis.DataFileID', fileID.genDataSourceID(self.image.dataSource))
        self.queueName = self.image.seriesName
        self.resultsMdh = self.image.mdh
        self.tq.releaseTasks(self.image.seriesName, startingAt)

    def pushImagesDS(self, startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
        resultsFilename = self._verifyResultsFilename(genResultFileName(self.image.seriesName))
        self.queueName = resultsFilename
            
        debugPrint('Results file = %s' % resultsFilename) 
        
        mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
        mdh.setEntry('Analysis.DetectionThreshold', detThresh)
        mdh.setEntry('Analysis.FitModule', fitFcn)
        mdh.setEntry('Analysis.DataFileID', fileID.genDataSourceID(self.image.dataSource))
        
        self.resultsMdh = mdh
        
        mn = self.image.dataSource.moduleName
        #dsID = self.image.seriesName
        #if it's a buffered source, go back to underlying source
        if mn == 'BufferedDataSource':
            mn = self.image.dataSource.dataSource.moduleName

        self.tq.createQueue('DSTaskQueue', self.queueName, mdh, 
                            mn, self.image.seriesName, 
                            resultsFilename, startAt = startingAt)
        
        evts = self.image.dataSource.getEvents()
        if len(evts) > 0:
            self.tq.addQueueEvents(self.queueName, evts)
        
        debugPrint('Queue created')
        



    def testFrames(self, detThresh = 0.9, offset = 0):
        from pylab import *
        close('all')
        if self.image.dataSource.moduleName == 'TQDataSource':
            self.checkTQ()
        matplotlib.interactive(False)
        clf()
        sq = min(self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 1000, self.image.dataSource.getNumSlices()/4)
        zps = array(range(self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 20, self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 24)  + range(sq, sq + 4) + range(self.image.dataSource.getNumSlices()/2,self.image.dataSource.getNumSlices() /2+4))
        zps += offset
        fitMod = self.fitFactories[self.cFitType.GetSelection()]
        #bgFrames = int(tBackgroundFrames.GetValue())
        bgFrames = [int(v) for v in self.tBackgroundFrames.GetValue().split(':')]
        print(zps)
        for i in range(12):
            print(i)
            #if 'Analysis.NumBGFrames' in md.getEntryNames():
            #bgi = range(max(zps[i] - bgFrames,mdh.getEntry('EstimatedLaserOnFrameNo')), zps[i])
            bgi = range(max(zps[i] + bgFrames[0],self.image.mdh.getEntry('EstimatedLaserOnFrameNo')), max(zps[i] + bgFrames[1],self.image.mdh.getEntry('EstimatedLaserOnFrameNo')))
            #else:
            #    bgi = range(max(zps[i] - 10,md.EstimatedLaserOnFrameNo), zps[i])
            mn = self.image.dataSource.moduleName
            if mn == 'BufferedDataSource':
                mn = self.image.dataSource.dataSource.moduleName

            if 'Splitter' in fitMod:
                ft = remFitBuf.fitTask(self.image.seriesName, zps[i], detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'SplitterObjFindR', bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
            else:
                ft = remFitBuf.fitTask(self.image.seriesName, zps[i], detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'LatObjFindFR', bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
            res = ft(taskQueue=self.tq)
            xp = floor(i/4)/3.
            yp = (3 - i%4)/4.
            #print xp, yp
            axes((xp,yp, 1./6,1./4.5))
            #d = ds[zps[i], :,:].squeeze().T
            d = self.image.dataSource.getSlice(zps[i]).T
            imshow(d, cmap=cm.hot, interpolation='nearest', hold=False, clim=(median(d.ravel()), d.max()))
            title('Frame %d' % zps[i])
            xlim(0, d.shape[1])
            ylim(0, d.shape[0])
            xticks([])
            yticks([])
            #print 'i = %d, ft.index = %d' % (i, ft.index)
            #subplot(4,6,2*i+13)
            xp += 1./6
            axes((xp,yp, 1./6,1./4.5))
            d = ft.ofd.filteredData.T
            #d = ft.data.squeeze().T
            imshow(d, cmap=cm.hot, interpolation='nearest', hold=False, clim=(median(d.ravel()), d.max()))
            plot([p.x for p in ft.ofd], [p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
            if ft.driftEst:
                 plot([p.x for p in ft.ofdDr], [p.y for p in ft.ofdDr], 'o', mew=2, mec='b', mfc='none', ms=9)
            if ft.fitModule in remFitBuf.splitterFitModules:
                    plot([p.x for p in ft.ofd], [d.shape[0] - p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
            axis('tight')
            xlim(0, d.shape[1])
            ylim(0, d.shape[0])
            xticks([])
            yticks([])
        show()
        matplotlib.interactive(True)
        
    def testFrame(self, detThresh = 0.9, offset = 0, gui=True):
        from pylab import *
        #close('all')
        if self.image.dataSource.moduleName == 'TQDataSource':
            self.checkTQ()
        if gui:    
            matplotlib.interactive(False)
            figure()
        #sq = min(self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 1000, self.image.dataSource.getNumSlices()/4)
        #zps = array(range(self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 20, self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 24)  + range(sq, sq + 4) + range(self.image.dataSource.getNumSlices()/2,self.image.dataSource.getNumSlices() /2+4))
        #zps += offset
        
        zp = self.do.zp
        fitMod = self.fitFactories[self.cFitType.GetSelection()]
        self.image.mdh.setEntry('Analysis.FitModule', fitMod)
        #bgFrames = int(tBackgroundFrames.GetValue())
        bgFrames = [int(v) for v in self.tBackgroundFrames.GetValue().split(':')]
        #print zps
        bgi = range(max(zp + bgFrames[0],self.image.mdh.getEntry('EstimatedLaserOnFrameNo')), max(zp + bgFrames[1],self.image.mdh.getEntry('EstimatedLaserOnFrameNo')))
            #else:
            #    bgi = range(max(zps[i] - 10,md.EstimatedLaserOnFrameNo), zps[i])
        mn = self.image.dataSource.moduleName
        if mn == 'BufferedDataSource':
            mn = self.image.dataSource.dataSource.moduleName

        #if 'Splitter' in fitMod:
        #    ft = remFitBuf.fitTask(self.image.seriesName, zp, detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'SplitterObjFindR', bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
        #else:
        #    ft = remFitBuf.fitTask(self.image.seriesName, zp, detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'LatObjFindFR', bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
        ft = remFitBuf.fitTask(self.image.seriesName, zp, detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), fitMod, bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
        res = ft(gui=gui,taskQueue=self.tq)
        
        if gui:
            figure()
            try:
                d = ft.ofd.filteredData.T
                #d = ft.data.squeeze().T
                imshow(d, cmap=cm.hot, interpolation='nearest', hold=False, clim=(median(d.ravel()), d.max()))
                plot([p.x for p in ft.ofd], [p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
                if ft.driftEst:
                     plot([p.x for p in ft.ofdDr], [p.y for p in ft.ofdDr], 'o', mew=2, mec='b', mfc='none', ms=9)
                #if ft.fitModule in remFitBuf.splitterFitModules:
                #        plot([p.x for p in ft.ofd], [d.shape[0] - p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
                #axis('tight')
                xlim(0, d.shape[1])
                ylim(d.shape[0], 0)
                xticks([])
                yticks([])
                
                
                    
                vx = 1e3*self.image.mdh['voxelsize.x']
                vy = 1e3*self.image.mdh['voxelsize.y']
                plot(res.results['fitResults']['x0']/vx, res.results['fitResults']['y0']/vy, '+b', mew=2)
                
                if 'startParams' in res.results.dtype.names:
                    plot(res.results['startParams']['x0']/vx, res.results['startParams']['y0']/vy, 'xc', mew=2)
                
                if 'tIm' in dir(ft.ofd):
                    figure()
                    imshow(ft.ofd.tIm.T, cmap=cm.hot, interpolation='nearest', hold=False)
                    #axis('tight')
                    xlim(0, d.shape[1])
                    ylim(d.shape[0], 0)
                    xticks([])
                    yticks([])
                    plot(res.results['fitResults']['x0']/vx, res.results['fitResults']['y0']/vy, '+b')
                    
                #figure()
                #imshow()
            except AttributeError:
                #d = self.image.data[:,:,zp].squeeze().T
                d = (ft.data.squeeze() - ft.bg.squeeze()).T
                imshow(d, cmap=cm.jet, interpolation='nearest', clim = [0, d.max()])
                xlim(0, d.shape[1])
                ylim(d.shape[0], 0)
                
                vx = 1e3*self.image.mdh['voxelsize.x']
                vy = 1e3*self.image.mdh['voxelsize.y']
                plot(res.results['fitResults']['x0']/vx, res.results['fitResults']['y0']/vy, 'ow')
                pass
                    
            show()
    
            matplotlib.interactive(True)
        
        return ft, res

def Plug(dsviewer):
    dsviewer.LMAnalyser = LMAnalyser(dsviewer)

    if not 'overlaypanel' in dir(dsviewer):    
        dsviewer.overlaypanel = OverlayPanel(dsviewer, dsviewer.view, dsviewer.image.mdh)
        dsviewer.overlaypanel.SetSize(dsviewer.overlaypanel.GetBestSize())
        pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        dsviewer._mgr.AddPane(dsviewer.overlaypanel, pinfo2)
    
        dsviewer.panesToMinimise.append(pinfo2)

