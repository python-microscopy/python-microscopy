# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:43:10 2015

@author: david
"""

import wx

import os

from pylab import *


import numpy
import pylab

import PYME.misc.autoFoldPanel as afp

from PYME.DSView import fitInfo
from PYME.DSView.OverlaysPanel import OverlayPanel
import wx.lib.agw.aui as aui

from PYME.Analysis.LMVis import gl_render
#from PYME.Analysis.LMVis import workspaceTree
#import sys

#import pylab
from PYME.misc import extraCMaps
from PYME.FileUtils import nameUtils

#import os
#from PYME.Analysis.LMVis import gl_render3D

from PYME.Analysis.LMVis import renderers
from PYME.Analysis.LMVis import pipeline


#try importing our drift correction stuff
HAVE_DRIFT_CORRECTION = False
try:
    from PYMEnf.DriftCorrection.driftGUI import CreateDriftPane
    HAVE_DRIFT_CORRECTION = True
    #from PYMEnf.DriftCorrection import driftGUI
    #renderers.renderMetadataProviders.append(driftGUI.dp.SaveMetadata)
except:
    pass

from PYME.Analysis.LMVis.colourFilterGUI import CreateColourFilterPane
from PYME.Analysis.LMVis.displayPane import CreateDisplayPane
from PYME.Analysis.LMVis.filterPane import CreateFilterPane

from PYME.Analysis.LMVis import progGraph as progGraph


import numpy as np

debug = True

def debugPrint(msg):
    if debug:
        print(msg)
        



class LMDisplay(object):    
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        
        self.image = dsviewer.image
        self.view = dsviewer.view
        self.do = dsviewer.do

        if 'fitResults' in dir(self.image):
            self.fitResults = self.image.fitResults
        else:
            self.fitResults = []
        
        if 'resultsMdh' in dir(self.image):
            self.resultsMdh = self.image.resultsMdh



        #a timer object to update for us
        #self.timer = mytimer()
        #self.timer.Start(10000)

        self.analDispMode = 'z'

        self.numAnalysed = 0
        self.numEvents = 0

        #dsviewer.paneHooks.append(self.GenFitStatusPanel)
        dsviewer.paneHooks.append(self.GenPanels)

        dsviewer.updateHooks.append(self.update)
        #dsviewer.statusHooks.append(self.GetStatusText)

        
        if (len(self.fitResults) > 0) and not 'PYME_BUGGYOPENGL' in os.environ.keys():
            self.GenResultsView()   
            
        if not 'pipeline' in dir(dsviewer):
            dsviewer.pipeline = pipeline.Pipeline(visFr=self)
        
        self.pipeline = dsviewer.pipeline
        self.pipeline.visFr = self
        
        #self.Quads = None
        dsviewer.menubar.Insert(dsviewer.menubar.GetMenuCount()-1, self.CreateMenuBar(subMenu=True), 'Points')      

        self.viewMode = 'points' #one of points, triangles, quads, or voronoi
        self.colData = 't'
        
#        if 'PYME_BUGGYOPENGL' in os.environ.keys():
#            pylab.plot(pylab.randn(10))


        self.glCanvas = gl_render.LMGLCanvas(self.dsviewer)
        self.dsviewer.AddPage(page=self.glCanvas, select=True, caption='View')
        self.glCanvas.cmap = pylab.cm.gist_rainbow #pylab.cm.hot

        #self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.refv = False

        #statusLog.SetStatusDispFcn(self.SetStatus)
        
        renderers.renderMetadataProviders.append(self.SaveMetadata)

    def OnIdle(self, event):
        if self.glCanvas.init and not self.refv:
            self.refv = True
            print((self.viewMode, self.colData))
            
            self.RefreshView()
            self.displayPane.OnPercentileCLim(None)
            self.Refresh()
            self.Update()
            print('refreshed')

        
    def SaveMetadata(self, mdh):
        mdh['Filter.Keys'] = self.pipeline.filterKeys      
        
        if HAVE_DRIFT_CORRECTION and 'x' in self.pipeline.mapping.mappings.keys(): #drift correction has been applied
            self.driftPane.dp.SaveMetadata(mdh)
            
    def CreateFoldPanel(self):
        self.dsviewer.CreateFoldPanel()

    def GenPanels(self, sidePanel):    
        self.GenDataSourcePanel(sidePanel)

        self.filterPane = CreateFilterPane(sidePanel, self.pipeline.filterKeys, self.pipeline, self)

        if HAVE_DRIFT_CORRECTION:
            self.driftPane = CreateDriftPane(sidePanel, self.pipeline.mapping, self.pipeline)
            
        self.colourFilterPane = CreateColourFilterPane(sidePanel, self.pipeline.colourFilter, self.pipeline)
        self.displayPane = CreateDisplayPane(sidePanel, self.glCanvas, self)
        
        if self.viewMode == 'quads':
            self.GenQuadTreePanel(sidePanel)

        if self.viewMode == 'points' or self.viewMode == 'tracks':
            pass
            self.GenPointsPanel(sidePanel)

        if self.viewMode == 'blobs':
            self.GenBlobPanel(sidePanel)

        if self.viewMode == 'interp_triangles':
            self.GenPointsPanel(sidePanel,'Vertex Colours')

        
        self.glCanvas.Refresh()

    def GenDataSourcePanel(self, pnl):
        item = afp.foldingPane(pnl, -1, caption="Data Source", pinned = False)

        self.dsRadioIds = []
        for ds in self.pipeline.dataSources:
            rbid = wx.NewId()
            self.dsRadioIds.append(rbid)
            rb = wx.RadioButton(item, rbid, ds._name)
            rb.SetValue(ds == self.pipeline.selectedDataSource)

            rb.Bind(wx.EVT_RADIOBUTTON, self.OnSourceChange)
            item.AddNewElement(rb)

        pnl.AddPane(item)


    def OnSourceChange(self, event):
        dsind = self.dsRadioIds.index(event.GetId())
        self.pipeline.selectedDataSource = self.pipeline.dataSources[dsind]
        self.RegenFilter()
        
    
    def GenQuadTreePanel(self, pnl):
        from PYME.Analysis.QuadTree import pointQT
        
        item = afp.foldingPane(pnl, -1, caption="QuadTree", pinned = True)
#        item = self._pnl.AddFoldPanel("QuadTree", collapsed=False,
#                                      foldIcons=self.Images)

        pan = wx.Panel(item, -1)
        bsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Leaf Size:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tQTLeafSize = wx.TextCtrl(pan, -1, '%d' % pointQT.QT_MAXRECORDS)
        hsizer.Add(self.tQTLeafSize, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        self.stQTSNR = wx.StaticText(pan, -1, 'Effective SNR = %3.2f' % pylab.sqrt(pointQT.QT_MAXRECORDS/2.0))
        bsizer.Add(self.stQTSNR, 0, wx.ALL, 5)

        #hsizer = wx.BoxSizer(wx.HORIZONTAL)
        #hsizer.Add(wx.StaticText(pan, -1, 'Goal pixel size [nm]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        #self.tQTSize = wx.TextCtrl(pan, -1, '20000')
        #hsizer.Add(self.tQTLeafSize, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        #bsizer.Add(hsizer, 0, wx.ALL, 0)
        
        pan.SetSizer(bsizer)
        bsizer.Fit(pan)

        
        item.AddNewElement(pan)
        #self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        self.tQTLeafSize.Bind(wx.EVT_TEXT, self.OnQTLeafChange)

        pnl.AddPane(item)

    

    def OnQTLeafChange(self, event):
        from PYME.Analysis.QuadTree import pointQT
        
        leafSize = int(self.tQTLeafSize.GetValue())
        if not leafSize >= 1:
            raise RuntimeError('QuadTree leaves must be able to contain at least 1 item')

        pointQT.QT_MAXRECORDS = leafSize
        self.stQTSNR.SetLabel('Effective SNR = %3.2f' % pylab.sqrt(pointQT.QT_MAXRECORDS/2.0))

        self.pipeline.Quads = None
        self.RefreshView()


    def GenBlobPanel(self, pnl):
#        item = self._pnl.AddFoldPanel("Objects", collapsed=False,
#                                      foldIcons=self.Images)
        item = afp.foldingPane(pnl, -1, caption="Objects", pinned = True)

        pan = wx.Panel(item, -1)
        bsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Threshold [nm]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tBlobDist = wx.TextCtrl(pan, -1, '%3.0f' % self.pipeline.objThreshold,size=(40,-1))
        hsizer.Add(self.tBlobDist, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Min Size [events]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tMinObjSize = wx.TextCtrl(pan, -1, '%d' % self.pipeline.objMinSize, size=(40, -1))
        hsizer.Add(self.tMinObjSize, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Jittering:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tObjJitter = wx.TextCtrl(pan, -1, '%d' % self.pipeline.blobJitter, size=(40, -1))
        hsizer.Add(self.tObjJitter, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        self.bApplyThreshold = wx.Button(pan, -1, 'Apply')
        bsizer.Add(self.bApplyThreshold, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.bObjMeasure = wx.Button(pan, -1, 'Measure')
        #self.bObjMeasure.Enable(False)
        bsizer.Add(self.bObjMeasure, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Object Colour:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.cBlobColour = wx.Choice(pan, -1, choices=['Index', 'Random'])
        self.cBlobColour.SetSelection(0)
        self.cBlobColour.Bind(wx.EVT_CHOICE, self.OnSetBlobColour)

        hsizer.Add(self.cBlobColour, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        pan.SetSizer(bsizer)
        bsizer.Fit(pan)


        #self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        item.AddNewElement(pan)

        self.bApplyThreshold.Bind(wx.EVT_BUTTON, self.OnObjApplyThreshold)
        self.bObjMeasure.Bind(wx.EVT_BUTTON, self.OnObjMeasure)

        pnl.AddPane(item)

    def OnSetBlobColour(self, event):
        bcolour = self.cBlobColour.GetStringSelection()

        if bcolour == 'Index':
            c = self.objCInd.astype('f')
        elif bcolour == 'Random':
            r = pylab.rand(self.objCInd.max() + 1)
            c = r[self.objCInd.astype('i')]
        else:
            c = self.pipeline.objectMeasures[bcolour][self.objCInd.astype('i')]

        self.glCanvas.c = c
        self.glCanvas.setColour()
        self.OnGLViewChanged()
        
        self.displayPane.hlCLim.SetData(self.glCanvas.c, self.glCanvas.clim[0], self.glCanvas.clim[1])




    def GenPointsPanel(self, pnl, title='Points'):
        item = afp.foldingPane(pnl, -1, caption=title, pinned = True)
#        item = self._pnl.AddFoldPanel(title, collapsed=False,
#                                      foldIcons=self.Images)

        pan = wx.Panel(item, -1)
        bsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Size [nm]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPointSize = wx.TextCtrl(pan, -1, '%3.2f' % self.glCanvas.pointSize)
        hsizer.Add(self.tPointSize, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        
        colData = ['<None>']

        if not self.pipeline.colourFilter == None:
            colData += list(self.pipeline.keys())

        colData += list(self.pipeline.GeneratedMeasures.keys())

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Colour:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.chPointColour = wx.Choice(pan, -1, choices=colData, size=(100, -1))
        if self.colData in colData:
            self.chPointColour.SetSelection(colData.index(self.colData))
        hsizer.Add(self.chPointColour, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)
        
        pan.SetSizer(bsizer)
        bsizer.Fit(pan)

        item.AddNewElement(pan)
        #self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        self.tPointSize.Bind(wx.EVT_TEXT, self.OnPointSizeChange)
        self.chPointColour.Bind(wx.EVT_CHOICE, self.OnChangePointColour)

        pnl.AddPane(item)

    def UpdatePointColourChoices(self):
        if self.viewMode == 'points': #only change if we are in points mode
            colData = ['<None>']

            if not self.pipeline.colourFilter == None:
                colData += list(self.pipeline.keys())

            colData += list(self.pipeline.GeneratedMeasures.keys())

            self.chPointColour.Clear()
            for cd in colData:
                self.chPointColour.Append(cd)

            if self.colData in colData:
                self.chPointColour.SetSelection(colData.index(self.colData))

    def OnPointSizeChange(self, event):
        self.glCanvas.pointSize = float(self.tPointSize.GetValue())
        self.glCanvas.Refresh()

    def OnChangePointColour(self, event):
        self.colData = event.GetString()
        
        self.RefreshView()

    def pointColour(self):
        pointColour = None
        
        if self.colData == '<None>':
            pointColour = None
        elif not self.pipeline.colourFilter == None:
            if self.colData in self.pipeline.keys():
                pointColour = self.pipeline[self.colData]
            elif self.colData in self.pipeline.GeneratedMeasures.keys():
                pointColour = self.pipeline.GeneratedMeasures[self.colData]
            else:
                pointColour = None

        return pointColour
        
    def Bind(self, *args, **kwargs):
        self.dsviewer.Bind(*args, **kwargs)

    def AddExtrasMenuItem(self,label, callback):
        '''Add an item to the VisGUI extras menu. 
        
        parameters:
            label       textual label to use for the menu item.
            callback    function to call when user selects the menu item. This 
                        function should accept one argument, which will be the
                        wxPython event generated by the menu selection.
        '''
        
        ID_NEWITEM = wx.NewId()
        self.extras_menu.Append(ID_NEWITEM, label)
        self.dsviewer.Bind(wx.EVT_MENU, callback, id=ID_NEWITEM)

    def CreateMenuBar(self, subMenu = False):

        # Make a menubar
        file_menu = wx.Menu()

        ID_OPEN = wx.ID_OPEN
        #ID_SAVE_MEASUREMENTS = wx.ID_SAVE
        #ID_QUIT = wx.ID_EXIT

        #ID_OPEN_RAW = wx.NewId()
        ID_OPEN_CHANNEL = wx.NewId()

        ID_VIEW_POINTS = wx.NewId()
        ID_VIEW_TRIANGS = wx.NewId()
        ID_VIEW_QUADS = wx.NewId()

        ID_VIEW_BLOBS = wx.NewId()

        ID_VIEW_VORONOI = wx.NewId()
        ID_VIEW_INTERP_TRIANGS = wx.NewId()
        ID_VIEW_TRACKS = wx.NewId()

        ID_VIEW_FIT = wx.NewId()
        ID_VIEW_FIT_ROI = wx.NewId()
        

        ID_TOGGLE_SETTINGS = wx.NewId()

        #ID_ABOUT = wx.ID_ABOUT

        ID_VIEW_3D_POINTS = wx.NewId()
        ID_VIEW_3D_TRIANGS = wx.NewId()
        ID_VIEW_3D_BLOBS = wx.NewId()

        ID_VIEW_BLOBS = wx.NewId()
        
        
        file_menu.Append(ID_OPEN, "&Open")
        #file_menu.Append(ID_OPEN_RAW, "Open &Raw/Prebleach Data")
        file_menu.Append(ID_OPEN_CHANNEL, "Open Extra &Channel")
        
        #file_menu.AppendSeparator()
        #file_menu.Append(ID_SAVE_MEASUREMENTS, "&Save Measurements")

        #file_menu.AppendSeparator()
        
        #file_menu.Append(ID_QUIT, "&Exit")

        self.view_menu = wx.Menu()

        try: #stop us bombing on Mac
            self.view_menu.AppendRadioItem(ID_VIEW_POINTS, '&Points')
            self.view_menu.AppendRadioItem(ID_VIEW_TRIANGS, '&Triangles')
            self.view_menu.AppendRadioItem(ID_VIEW_QUADS, '&Quad Tree')
            self.view_menu.AppendRadioItem(ID_VIEW_VORONOI, '&Voronoi')
            self.view_menu.AppendRadioItem(ID_VIEW_INTERP_TRIANGS, '&Interpolated Triangles')
            self.view_menu.AppendRadioItem(ID_VIEW_BLOBS, '&Blobs')
            self.view_menu.AppendRadioItem(ID_VIEW_TRACKS, '&Tracks')
        except:
            self.view_menu.Append(ID_VIEW_POINTS, '&Points')
            self.view_menu.Append(ID_VIEW_TRIANGS, '&Triangles')
            self.view_menu.Append(ID_VIEW_QUADS, '&Quad Tree')
            self.view_menu.Append(ID_VIEW_VORONOI, '&Voronoi')
            self.view_menu.Append(ID_VIEW_INTERP_TRIANGS, '&Interpolated Triangles')
            self.view_menu.Append(ID_VIEW_BLOBS, '&Blobs')
            self.view_menu.Append(ID_VIEW_TRACKS, '&Tracks')

        self.view_menu.Check(ID_VIEW_POINTS, True)
        #self.view_menu.Enable(ID_VIEW_QUADS, False)

        self.view_menu.AppendSeparator()
        self.view_menu.Append(ID_VIEW_FIT, '&Fit')
        self.view_menu.Append(ID_VIEW_FIT_ROI, 'Fit &ROI')

        self.ID_VIEW_CLIP_ROI = wx.NewId()
        self.view_menu.Append(self.ID_VIEW_CLIP_ROI, 'Clip to ROI\tF8')


        self.view_menu.AppendSeparator()
        self.view_menu.AppendCheckItem(ID_TOGGLE_SETTINGS, "Show Settings")
        self.view_menu.Check(ID_TOGGLE_SETTINGS, True)

        #self.view3d_menu = wx.Menu()

#        try: #stop us bombing on Mac
#            self.view3d_menu.AppendRadioItem(ID_VIEW_3D_POINTS, '&Points')
#            self.view3d_menu.AppendRadioItem(ID_VIEW_3D_TRIANGS, '&Triangles')
#            self.view3d_menu.AppendRadioItem(ID_VIEW_3D_BLOBS, '&Blobs')
#        except:
        #self.view3d_menu.Append(ID_VIEW_3D_POINTS, '&Points')
        #self.view3d_menu.Append(ID_VIEW_3D_TRIANGS, '&Triangles')
        #self.view3d_menu.Append(ID_VIEW_3D_BLOBS, '&Blobs')

        #self.view3d_menu.Enable(ID_VIEW_3D_TRIANGS, False)
        #self.view3d_menu.Enable(ID_VIEW_3D_BLOBS, False)

        #self.view_menu.Check(ID_VIEW_3D_POINTS, True)

        self.gen_menu = wx.Menu()
        renderers.init_renderers(self, self.dsviewer)

        self.extras_menu = wx.Menu()
        from PYME.Analysis.LMVis import Extras
        Extras.InitPlugins(self)
        
        try:
            #see if we can find any 'non free' plugins
            from PYMEnf.Analysis.LMVis import Extras
            #Extras.InitPlugins(self)
        except ImportError:
            pass

        #help_menu = wx.Menu()
        #help_menu.Append(ID_ABOUT, "&About")

        if subMenu:
            menu_bar = wx.Menu()
            menu_bar.AppendSubMenu(file_menu, "&File")
            menu_bar.AppendSubMenu(self.view_menu, "&View")
            menu_bar.AppendSubMenu(self.gen_menu, "&Generate Image")
            menu_bar.AppendSubMenu(self.extras_menu, "&Extras")
            #menu_bar.AppendSubMenu(self.view3d_menu, "View &3D")
        else:
            menu_bar = wx.MenuBar()

            menu_bar.Append(file_menu, "&File")
            menu_bar.Append(self.view_menu, "&View")
            menu_bar.Append(self.gen_menu, "&Generate Image")
            menu_bar.Append(self.extras_menu, "&Extras")
            #menu_bar.Append(self.view3d_menu, "View &3D")
            
        #menu_bar.Append(help_menu, "&Help")

        #self.dsviewer.Bind(wx.EVT_MENU, self.OnAbout, id=ID_ABOUT)
        #self.dsviewer.Bind(wx.EVT_MENU, self.OnQuit, id=ID_QUIT)
        #self.dsviewer.Bind(wx.EVT_MENU, self.OnToggleWindow, id=ID_TOGGLE_SETTINGS)

        self.dsviewer.Bind(wx.EVT_MENU, self.OnOpenFile, id=ID_OPEN)
        #self.dsviewer.Bind(wx.EVT_MENU, self.OnOpenChannel, id=ID_OPEN_CHANNEL)
        #self.dsviewer.Bind(wx.EVT_MENU, self.OnOpenRaw, id=ID_OPEN_RAW)

        #self.dsviewer.Bind(wx.EVT_MENU, self.OnSaveMeasurements, id=ID_SAVE_MEASUREMENTS)

        self.dsviewer.Bind(wx.EVT_MENU, self.OnViewPoints, id=ID_VIEW_POINTS)
        self.dsviewer.Bind(wx.EVT_MENU, self.OnViewTriangles, id=ID_VIEW_TRIANGS)
        self.dsviewer.Bind(wx.EVT_MENU, self.OnViewQuads, id=ID_VIEW_QUADS)
        self.dsviewer.Bind(wx.EVT_MENU, self.OnViewVoronoi, id=ID_VIEW_VORONOI)
        self.dsviewer.Bind(wx.EVT_MENU, self.OnViewInterpTriangles, id=ID_VIEW_INTERP_TRIANGS)

        self.dsviewer.Bind(wx.EVT_MENU, self.OnViewBlobs, id=ID_VIEW_BLOBS)
        self.dsviewer.Bind(wx.EVT_MENU, self.OnViewTracks, id=ID_VIEW_TRACKS)

        self.dsviewer.Bind(wx.EVT_MENU, self.SetFit, id=ID_VIEW_FIT)
        self.dsviewer.Bind(wx.EVT_MENU, self.OnFitROI, id=ID_VIEW_FIT_ROI)

        #self.dsviewer.Bind(wx.EVT_MENU, self.OnView3DPoints, id=ID_VIEW_3D_POINTS)
        #self.dsviewer.Bind(wx.EVT_MENU, self.OnView3DTriangles, id=ID_VIEW_3D_TRIANGS)
        #self.Bind(wx.EVT_MENU, self.OnView3DBlobs, id=ID_VIEW_3D_BLOBS)

        return menu_bar

    def OnViewPoints(self,event):
        self.viewMode = 'points'
        #self.glCanvas.cmap = pylab.cm.hsv
        self.RefreshView()
        self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewTracks(self,event):
        self.viewMode = 'tracks'
        #self.glCanvas.cmap = pylab.cm.hsv
        self.RefreshView()
        self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewBlobs(self,event):
        self.viewMode = 'blobs'
        self.RefreshView()
        self.CreateFoldPanel()
        #self.OnPercentileCLim(None)

    def OnViewTriangles(self,event):
        self.viewMode = 'triangles'
        self.RefreshView()
        self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewQuads(self,event):
        self.viewMode = 'quads'
        self.RefreshView()
        self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewVoronoi(self,event):
        self.viewMode = 'voronoi'
        self.RefreshView()
        self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewInterpTriangles(self,event):
        self.viewMode = 'interp_triangles'
        self.RefreshView()
        self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnOpenFile(self, event):
        filename = wx.FileSelector("Choose a file to open", 
                                   nameUtils.genResultDirectoryPath(), 
                                   default_extension='h5r', 
                                   wildcard='PYME Results Files (*.h5r)|*.h5r|Tab Formatted Text (*.txt)|*.txt|Matlab data (*.mat)|*.mat')

        #print filename
        if not filename == '':
            self.OpenFile(filename)
            
      
            
    def OpenFile(self, filename):
        args = {}
        
        if os.path.splitext(filename)[1] == '.h5r':
            pass
        elif os.path.splitext(filename)[1] == '.mat':
            from PYME.Analysis.LMVis import importTextDialog
            from scipy.io import loadmat
            
            mf = loadmat(filename)

            dlg = importTextDialog.ImportMatDialog(self, [k for k in mf.keys() if not k.startswith('__')])
            ret = dlg.ShowModal()

            if not ret == wx.ID_OK:
                dlg.Destroy()
                return #we cancelled
                
            args['FieldNames'] = dlg.GetFieldNames()
            args['VarName'] = dlg.GetVarName()
            args['PixelSize'] = dlg.GetPixelSize()
            
            
            dlg.Destroy()

        else: #assume it's a text file
            from PYME.Analysis.LMVis import importTextDialog
            
            dlg = importTextDialog.ImportTextDialog(self, filename)
            ret = dlg.ShowModal()

            if not ret == wx.ID_OK:
                dlg.Destroy()
                return #we cancelled
                
            args['FieldNames'] = dlg.GetFieldNames()
            args['SkipRows'] = dlg.GetNumberComments()
            args['PixelSize'] = dlg.GetPixelSize()
            
            #print 'Skipping %d rows' %args['SkipRows']
            dlg.Destroy()

        print('Creating Pipeline')
        self.pipeline.OpenFile(filename, **args)
        print('Pipeline Created')
        
        
        #############################
        #now do all the gui stuff
        #self._removeOldTabs()
        #self._createNewTabs()
        
        #self.CreateFoldPanel()
        #print('Gui stuff done')
        
        self.SetFit()
            



        

    def RegenFilter(self):
        self.pipeline.Rebuild()

        self.filterPane.stFilterNumPoints.SetLabel('%d of %d events' % (len(self.pipeline.filter['x']), len(self.pipeline.selectedDataSource['x'])))

        self.RefreshView()


    def RefreshView(self):
        if not self.pipeline.ready:
            return #get out of here

        if len(self.pipeline['x']) == 0:
            wx.MessageBox('No data points - try adjusting the filter', 
                          "len(filter['x']) ==0")
            return

        if self.glCanvas.init == 0: #glcanvas is not initialised
            return

        bCurr = wx.BusyCursor()

        if self.pipeline.objects == None:
#            if 'bObjMeasure' in dir(self):
#                self.bObjMeasure.Enable(False)
            self.objectMeasures = None

#            if not self.rav == None: #remove previous event viewer
#                i = 0
#                found = False
#                while not found and i < self.notebook.GetPageCount():
#                    if self.notebook.GetPage(i) == self.rav:
#                        self.notebook.DeletePage(i)
#                        found = True
#                    else:
#                        i += 1
#                        
#                self.rav = None

        if self.viewMode == 'points':
            self.glCanvas.setPoints(self.pipeline['x'], 
                                    self.pipeline['y'], self.pointColour())
                                    
            if 'glCanvas3D' in dir(self):
                self.glCanvas3D.setPoints(self.pipeline['x'], 
                                      self.pipeline['y'], 
                                      self.pipeline['z'], 
                                      self.pointColour())
                self.glCanvas3D.setCLim(self.glCanvas.clim, (-5e5, -5e5))
        elif self.viewMode == 'tracks':
            self.glCanvas.setTracks(self.pipeline['x'], 
                                    self.pipeline['y'], 
                                    self.pipeline['clumpIndex'], 
                                    self.pointColour())
                                    
        elif self.viewMode == 'triangles':
            self.glCanvas.setTriang(self.pipeline.getTriangles())

        elif self.viewMode == 'voronoi':
            status = statusLog.StatusLogger("Generating Voronoi Diagram ... ")
            self.glCanvas.setVoronoi(self.pipeline.getTriangles())
            

        elif self.viewMode == 'quads':
            if self.pipeline.Quads == None:
                status = statusLog.StatusLogger("Generating QuadTree ...")
                self.pipeline.GenQuads()
                

            self.glCanvas.setQuads(self.pipeline.Quads)

        elif self.viewMode == 'interp_triangles':
            self.glCanvas.setIntTriang(self.pipeline.getTriangles(), self.pointColour())

        elif self.viewMode == 'blobs':
            if self.pipeline.objects == None:
                #check to see that we don't have too many points
                if len(self.pipeline['x']) > 1e5:
                    goAhead = wx.MessageBox('You have %d events in the selected ROI;\nThis could take a LONG time ...' % len(self.pipeline['x']), 'Continue with blob detection', wx.YES_NO|wx.ICON_EXCLAMATION)
    
                    if not goAhead == wx.YES:
                        return

            self.glCanvas.setBlobs(*self.pipeline.getBlobs())
            self.objCInd = self.glCanvas.c

        self.displayPane.hlCLim.SetData(self.glCanvas.c, self.glCanvas.clim[0], 
                                        self.glCanvas.clim[1])

        #if not self.colp == None and self.colp.IsShown():
        #    self.colp.refresh()

        #self.sh.shell.user_ns.update(self.__dict__)
        wx.EndBusyCursor()
        #self.workspaceView.RefreshItems()


    def SetFit(self,event = None):
        xsc = self.pipeline.imageBounds.width()*1./self.glCanvas.Size[0]
        ysc = self.pipeline.imageBounds.height()*1./self.glCanvas.Size[1]

        if xsc > ysc:
            self.glCanvas.setView(self.pipeline.imageBounds.x0, self.pipeline.imageBounds.x1, 
                                  self.pipeline.imageBounds.y0, self.pipeline.imageBounds.y0 + xsc*self.glCanvas.Size[1])
        else:
            self.glCanvas.setView(self.pipeline.imageBounds.x0, self.pipeline.imageBounds.x0 + ysc*self.glCanvas.Size[0], 
                                  self.pipeline.imageBounds.y0, self.pipeline.imageBounds.y1)

    def OnFitROI(self,event = None):
        if 'x' in self.pipeline.filterKeys.keys():
            xbounds = self.pipeline.filterKeys['x']
        else:
            xbounds = (self.pipeline.imageBounds.x0, self.pipeline.imageBounds.x1)

        if 'y' in self.pipeline.filterKeys.keys():
            ybounds = self.pipeline.filterKeys['y']
        else:
            ybounds = (self.pipeline.imageBounds.y0, self.pipeline.imageBounds.y1)
        
        xsc = (xbounds[1] - xbounds[0])*1./self.glCanvas.Size[0]
        ysc = (ybounds[1] - ybounds[0])*1./self.glCanvas.Size[1]

        if xsc > ysc:
            self.glCanvas.setView(xbounds[0], xbounds[1], ybounds[0], 
                                  ybounds[0] + xsc*self.glCanvas.Size[1])
        else:
            self.glCanvas.setView(xbounds[0], xbounds[0] + ysc*self.glCanvas.Size[0], 
                                  ybounds[0], ybounds[1])


    def SetStatus(self, statusText):
        self.statusbar.SetStatusText(statusText, 0)

    def GenResultsView(self):
        voxx = 1e3*self.image.mdh.getEntry('voxelsize.x')
        voxy = 1e3*self.image.mdh.getEntry('voxelsize.y')
        
        self.SetFitInfo()

        from PYME.Analysis.LMVis import gl_render
        self.glCanvas = gl_render.LMGLCanvas(self.dsviewer, False, vp = self.do, vpVoxSize = voxx)
        self.glCanvas.cmap = pylab.cm.gist_rainbow
        self.glCanvas.pointSelectionCallbacks.append(self.OnPointSelect)

        self.dsviewer.AddPage(page=self.glCanvas, select=True, caption='VisLite')

        xsc = self.image.data.shape[0]*1.0e3*self.image.mdh.getEntry('voxelsize.x')/self.glCanvas.Size[0]
        ysc = self.image.data.shape[1]*1.0e3*self.image.mdh.getEntry('voxelsize.y')/ self.glCanvas.Size[1]

        if xsc > ysc:
            self.glCanvas.setView(0, xsc*self.glCanvas.Size[0], 0, xsc*self.glCanvas.Size[1])
        else:
            self.glCanvas.setView(0, ysc*self.glCanvas.Size[0], 0, ysc*self.glCanvas.Size[1])

        #we have to wait for the gui to be there before we start changing stuff in the GL view
        #self.timer.WantNotification.append(self.AddPointsToVis)

        self.glCanvas.Bind(wx.EVT_IDLE, self.OnIdle)
        self.pointsAdded = False
        
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


    def AddPointsToVis(self):
        self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],self.fitResults['tIndex'].astype('f'))
        self.glCanvas.setCLim((0, self.fitResults['tIndex'].max()))

        self.timer.WantNotification.remove(self.AddPointsToVis)

    def GetStatusText(self):
        return 'Frames Analysed: %d    Events detected: %d' % (self.numAnalysed, self.numEvents)
        


    def GenFitStatusPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Fit Status", pinned = True)

        pan = wx.Panel(item, -1, size = (160, 300))

        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Colour:'), 0, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        self.chProgDispColour = wx.Choice(pan, -1, choices = ['z', 'gFrac', 't'], size=(60, -1))
        self.chProgDispColour.Bind(wx.EVT_CHOICE, self.OnProgDispColourChange)
        hsizer.Add(self.chProgDispColour, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)

        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'CMap:'), 0, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        self.chProgDispCMap = wx.Choice(pan, -1, choices = ['gist_rainbow', 'RdYlGn'], size=(60, -1))
        self.chProgDispCMap.Bind(wx.EVT_CHOICE, self.OnProgDispCMapChange)
        hsizer.Add(self.chProgDispCMap, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)

        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 7)

        self.progPan = progGraph.progPanel(pan, self.fitResults, size=(220, 250))
        self.progPan.draw()

        vsizer.Add(self.progPan, 1,wx.ALL|wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND, 0)

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

   

    def analRefresh(self):
        newNumAnalysed = self.tq.getNumberTasksCompleted(self.image.seriesName)
        if newNumAnalysed > self.numAnalysed:
            self.numAnalysed = newNumAnalysed
            newResults = self.tq.getQueueData(self.image.seriesName, 'FitResults', len(self.fitResults))
            if len(newResults) > 0:
                if len(self.fitResults) == 0:
                    self.fitResults = newResults
                else:
                    self.fitResults = numpy.concatenate((self.fitResults, newResults))
                self.progPan.fitResults = self.fitResults

                self.view.points = numpy.vstack((self.fitResults['fitResults']['x0'], self.fitResults['fitResults']['y0'], self.fitResults['tIndex'])).T

                self.numEvents = len(self.fitResults)

                if self.analDispMode == 'z' and (('zm' in dir(self)) or ('z0' in self.fitResults['fitResults'].dtype.fields)):
                    #display z as colour
                    if 'zm' in dir(self): #we have z info
                        if 'z0' in self.fitResults['fitResults'].dtype.fields:
                            z = 1e3*self.zm(self.fitResults['tIndex'].astype('f')).astype('f')
                            z_min = z.min() - 500
                            z_max = z.max() + 500
                            z = z + self.fitResults['fitResults']['z0']
                            self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],z)
                            self.glCanvas.setCLim((z_min, z_max))
                        else:
                            z = self.zm(self.fitResults['tIndex'].astype('f')).astype('f')
                            self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],z)
                            self.glCanvas.setCLim((z.min(), z.max()))
                    elif 'z0' in self.fitResults['fitResults'].dtype.fields:
                        z = self.fitResults['fitResults']['z0']
                        self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],z)
                        self.glCanvas.setCLim((-1e3, 1e3))

                elif self.analDispMode == 'gFrac' and 'Ag' in self.fitResults['fitResults'].dtype.fields:
                    #display ratio of colour channels as point colour
                    c = self.fitResults['fitResults']['Ag']/(self.fitResults['fitResults']['Ag'] + self.fitResults['fitResults']['Ar'])
                    self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],c)
                    self.glCanvas.setCLim((0, 1))
                elif self.analDispMode == 'gFrac' and 'ratio' in self.fitResults['fitResults'].dtype.fields:
                    #display ratio of colour channels as point colour
                    c = self.fitResults['fitResults']['ratio']
                    self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],c)
                    self.glCanvas.setCLim((0, 1))
                else:
                    #default to time
                    self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],self.fitResults['tIndex'].astype('f'))
                    self.glCanvas.setCLim((0, self.numAnalysed))

        if (self.tq.getNumberOpenTasks(self.image.seriesName) + self.tq.getNumberTasksInProgress(self.image.seriesName)) == 0 and 'SpoolingFinished' in self.image.mdh.getEntryNames():
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


    

def Plug(dsviewer):
    dsviewer.LMDisplay = LMDisplay(dsviewer)

    if not 'overlaypanel' in dir(dsviewer):    
        dsviewer.overlaypanel = OverlayPanel(dsviewer, dsviewer.view, dsviewer.image.mdh)
        dsviewer.overlaypanel.SetSize(dsviewer.overlaypanel.GetBestSize())
        pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        dsviewer._mgr.AddPane(dsviewer.overlaypanel, pinfo2)
    
        dsviewer.panesToMinimise.append(pinfo2)