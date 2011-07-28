#!/usr/bin/python
##################
# VisGUI.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
import wx.py.shell

import PYME.misc.autoFoldPanel as afp
import wx.lib.agw.aui as aui

#hacked so py2exe works
from PYME.DSView.dsviewer_npy import View3D

from PYME.Analysis.LMVis import gl_render
from PYME.Analysis.LMVis import workspaceTree
import sys
from PYME.Analysis.LMVis import inpFilt

import pylab
from PYME.misc import extraCMaps
from PYME.FileUtils import nameUtils

from PYME.Analysis.EdgeDB import edges

import os
from PYME.Analysis.LMVis import gl_render3D

from matplotlib import delaunay

from PYME.Analysis.QuadTree import pointQT
#import Image

from PYME.Analysis.LMVis import importTextDialog
from PYME.Analysis.LMVis import visHelpers
from PYME.Analysis.LMVis import imageView
from PYME.Analysis.LMVis import colourPanel
from PYME.Analysis.LMVis import renderers

try:
#    from PYME.Analysis.LMVis import gen3DTriangs
    from PYME.Analysis.LMVis import recArrayView
    from PYME.Analysis.LMVis import objectMeasure
except:
    pass

#try importing our drift correction stuff
HAVE_DRIFT_CORRECTION = False
try:
    from PYME.Analysis.DriftCorrection.driftGUI import CreateDriftPane
    HAVE_DRIFT_CORRECTION = True
except:
    pass

from PYME.Analysis.LMVis.colourFilterGUI import CreateColourFilterPane
from PYME.Analysis.LMVis.displayPane import CreateDisplayPane
from PYME.Analysis.LMVis.filterPane import CreateFilterPane

from PYME.Analysis import piecewiseMapping
from PYME.Analysis import MetadataTree

#import time
import numpy as np
import scipy.special

#import tables
from PYME.Analysis import MetaData
from PYME.Acquire import MetaDataHandler

from PYME.DSView import eventLogViewer

#import threading


from PYME.misc.auiFloatBook import AuiNotebookWithFloatingPages

from PYME.Analysis.LMVis import statusLog
from PYME.Analysis.LMVis.visHelpers import ImageBounds#, GeneratedImage

from PYME.DSView.image import ImageStack, GeneratedImage


class VisGUIFrame(wx.Frame):    
    def __init__(self, parent, filename=None, id=wx.ID_ANY, title="PYME Visualise", pos=wx.DefaultPosition,
                 size=(700,650), style=wx.DEFAULT_FRAME_STYLE):

        wx.Frame.__init__(self, parent, id, title, pos, size, style)
        self._mgr = aui.AuiManager(agwFlags = aui.AUI_MGR_DEFAULT | aui.AUI_MGR_AUTONB_NO_CAPTION)
        atabstyle = self._mgr.GetAutoNotebookStyle()
        self._mgr.SetAutoNotebookStyle((atabstyle ^ aui.AUI_NB_BOTTOM) | aui.AUI_NB_TOP)
        self._mgr.SetManagedWindow(self)

        self._flags = 0
               
        self.SetMenuBar(self.CreateMenuBar())

        self.statusbar = self.CreateStatusBar(1, wx.ST_SIZEGRIP)
        #self.statusbar.SetStatusWidths([-4, -4])
        self.statusbar.SetStatusText("", 0)
       
        self._leftWindow1 = wx.Panel(self, -1, size = wx.Size(220, 1000))
        self._pnl = 0
        
        #self.notebook = AuiNotebookWithFloatingPages(id=-1, parent=self, style=wx.aui.AUI_NB_TAB_SPLIT)

        self.MainWindow = self #so we can access from shell
        self.sh = wx.py.shell.Shell(id=-1,
              parent=self, size=wx.Size(-1, -1), style=0, locals=self.__dict__,
              introText='Python SMI bindings - note that help, license etc below is for Python, not PySMI\n\n')

        self._mgr.AddPane(self.sh, aui.AuiPaneInfo().
                          Name("Shell").Caption("Console").Centre().CloseButton(False).CaptionVisible(False))

        self.elv = None
        self.colp = None
        self.mdp = None
        self.rav = None

        self.filesToClose = []
        self.generatedImages = []

        self.dataSources = []
        self.selectedDataSource = None
        self.filterKeys = {'error_x': (0,30), 'A':(5,2000), 'sig' : (95, 200)}

        self.filter = None
        self.mapping = None
        self.colourFilter = None

        self.fluorSpecies = {}
        self.chromaticShifts = {}
        self.t_p_dye = 0.1
        self.t_p_other = 0.1
        self.t_p_background = .01

        self.objThreshold = 30
        self.objMinSize = 10
        self.blobJitter = 0
        self.objects = None

        self.imageBounds = ImageBounds(0,0,0,0)

        #generated Quad-tree will allow visualisations with pixel sizes of self.QTGoalPixelSize*2^N for any N
        self.QTGoalPixelSize = 5 #nm

        self.viewMode = 'points' #one of points, triangles, quads, or voronoi
        self.Triangles = None
        self.edb = None
        self.GeneratedMeasures = {}
        self.Quads = None
        #self.pointColour = None
        self.colData = 't'

        #self.sh = WxController(self.notebook)
        #print self.sh.shell.user_ns
        #self.__dict__.update(self.sh.shell.user_ns)
        #self.sh.shell.user_ns = self.__dict__

        #self.notebook.AddPage(page=self.sh, select=True, caption='Console')

        #self.sh.execute_command('from pylab import *', hidden=True)
        #self.sh.execute_command('from PYME.DSView.dsviewer_npy import View3D', hidden=True)

        self.sh.Execute('from pylab import *')
        self.sh.Execute('from PYME.DSView.dsviewer_npy import View3D')
        #self.sh.runfile(os.path.join(os.path.dirname(__file__),'driftutil.py'))

        self.workspace = workspaceTree.WorkWrap(self.__dict__)

        ##### Make certain things visible in the workspace tree

        #components of the pipeline
        col = self.workspace.newColour()
        self.workspace.addKey('dataSources', col)
        self.workspace.addKey('selectedDataSource', col)
        self.workspace.addKey('filter', col)
        self.workspace.addKey('mapping', col)
        self.workspace.addKey('colourFilter', col)

        #Generated stuff
        col = self.workspace.newColour()
        self.workspace.addKey('GeneratedMeasures', col)
        self.workspace.addKey('generatedImages', col)
        self.workspace.addKey('objects', col)

        #main window, so we can get everything else if needed
        col = self.workspace.newColour()
        self.workspace.addKey('MainWindow', col)

        ######

        self.workspaceView = workspaceTree.WorkspaceTree(self, workspace=self.workspace, shell=self.sh)
        self.AddPage(page=self.workspaceView, select=False, caption='Workspace')

        self.glCanvas = gl_render.LMGLCanvas(self)
        self.AddPage(page=self.glCanvas, select=True, caption='View')
        self.glCanvas.cmap = pylab.cm.gist_rainbow #pylab.cm.hot

        #self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_MOVE, self.OnMove)
        self.Bind(wx.EVT_CLOSE, self.OnQuit)

        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.refv = False

        statusLog.SetStatusDispFcn(self.SetStatus)

        self.CreateFoldPanel()
        self._mgr.AddPane(self._leftWindow1, aui.AuiPaneInfo().
                          Name("sidebar").Left().CloseButton(False).CaptionVisible(False))

        #self._mgr.AddPane(self.notebook, aui.AuiPaneInfo().
        #                  Name("shell").Centre().CaptionVisible(False).CloseButton(False))

        self._mgr.Update()

        if not filename==None:
            #self.glCanvas.OnPaint(None)
            self.OpenFile(filename)

        nb = self._mgr.GetNotebooks()[0]
        nb.SetSelection(2)

        #print 'about to refresh'
        #self.RefreshView()
        #print 'refreshed'
        #self.Refresh()

    def OnIdle(self, event):
        if not self.refv:
            self.refv = True
            self.RefreshView()
            self.Refresh()
            print 'refreshed'

    def AddPage(self, page=None, select=False,caption='Dummy'):
        self._mgr.Update()
        pn = self._mgr.GetPaneByName("Shell")
        if pn.IsNotebookPage():
            print pn.notebook_id
            nbs = self._mgr.GetNotebooks()
            if len(nbs) > pn.notebook_id:
                currPage = nbs[pn.notebook_id].GetSelection()
            self._mgr.AddPane(page, aui.AuiPaneInfo().
                          Name(caption.replace(' ', '')).Caption(caption).CloseButton(False).NotebookPage(pn.notebook_id))
            if (not select) and len(nbs) > pn.notebook_id:
                nbs[pn.notebook_id].SetSelection(currPage)
        else:
            self._mgr.AddPane(page, aui.AuiPaneInfo().
                          Name(caption.replace(' ', '')).Caption(caption).CloseButton(False), target=pn)
            #nb = self._mgr.GetNotebooks()[0]
            #if not select:
            #    nb.SetSelection(0)

        self._mgr.Update()

    def OnSize(self, event):

        #wx.LayoutAlgorithm().LayoutWindow(self, self.notebook)
        event.Skip()

    def OnMove(self, event):
        #pass
        self.Refresh()
        event.Skip()
        

    def OnQuit(self, event):
        while len(self.filesToClose) > 0:
            self.filesToClose.pop().close()

        pylab.close('all')
        self.Destroy()


    def OnAbout(self, event):
        msg = "PYME Visualise\n\n Visualisation of localisation microscopy data\nDavid Baddeley 2009"
              
        dlg = wx.MessageDialog(self, msg, "About PYME Visualise",
                               wx.OK | wx.ICON_INFORMATION)
        dlg.SetFont(wx.Font(8, wx.NORMAL, wx.NORMAL, wx.NORMAL, False, "Verdana"))
        dlg.ShowModal()
        dlg.Destroy()

    def OnToggleWindow(self, event):
        self._mgr.ShowPane(self._leftWindow1,not self._leftWindow1.IsShown())
        self.glCanvas.Refresh()    

    def CreateFoldPanel(self):
        # delete earlier panel
        self._leftWindow1.DestroyChildren()

        # recreate the foldpanelbar
        hsizer = wx.BoxSizer(wx.VERTICAL)

        s = self._leftWindow1.GetBestSize()

        self._pnl = afp.foldPanel(self._leftWindow1, -1, wx.DefaultPosition,s)

        self.GenDataSourcePanel()
        #self.GenFilterPanel()
        self.filterPane = CreateFilterPane(self._pnl, self.filterKeys, self)

        if HAVE_DRIFT_CORRECTION:
            CreateDriftPane(self._pnl, self.mapping, self)
            #self.GenDriftPanel()
            
        self.colourFilterPane = CreateColourFilterPane(self._pnl, self.colourFilter, self)
        self.displayPane = CreateDisplayPane(self._pnl, self.glCanvas, self)
        
        if self.viewMode == 'quads':
            self.GenQuadTreePanel()

        if self.viewMode == 'points' or self.viewMode == 'tracks':
            self.GenPointsPanel()

        if self.viewMode == 'blobs':
            self.GenBlobPanel()

        if self.viewMode == 'interp_triangles':
            self.GenPointsPanel('Vertex Colours')

        hsizer.Add(self._pnl, 1, wx.EXPAND, 0)
        self._leftWindow1.SetSizerAndFit(hsizer)
        
        self.glCanvas.Refresh()

    def GenDataSourcePanel(self):
        item = afp.foldingPane(self._pnl, -1, caption="Data Source", pinned = False)

        self.dsRadioIds = []
        for ds in self.dataSources:
            rbid = wx.NewId()
            self.dsRadioIds.append(rbid)
            rb = wx.RadioButton(item, rbid, ds._name)
            rb.SetValue(ds == self.selectedDataSource)

            rb.Bind(wx.EVT_RADIOBUTTON, self.OnSourceChange)
            item.AddNewElement(rb)

        self._pnl.AddPane(item)


    def OnSourceChange(self, event):
        dsind = self.dsRadioIds.index(event.GetId())
        self.selectedDataSource = self.dataSources[dsind]
        self.RegenFilter()

    def ClearGenerated(self):
        self.Triangles = None
        self.edb = None
        self.GeneratedMeasures = {}
        self.Quads = None

        self.RefreshView()
        
    
    def GenQuadTreePanel(self):
        item = afp.foldingPane(self._pnl, -1, caption="QuadTree", pinned = True)
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

        self._pnl.AddPane(item)

    

    def OnQTLeafChange(self, event):
        leafSize = int(self.tQTLeafSize.GetValue())
        if not leafSize >= 1:
            raise RuntimeError('QuadTree leaves must be able to contain at least 1 item')

        pointQT.QT_MAXRECORDS = leafSize
        self.stQTSNR.SetLabel('Effective SNR = %3.2f' % pylab.sqrt(pointQT.QT_MAXRECORDS/2.0))

        self.Quads = None
        self.RefreshView()


    def GenBlobPanel(self):
#        item = self._pnl.AddFoldPanel("Objects", collapsed=False,
#                                      foldIcons=self.Images)
        item = afp.foldingPane(self._pnl, -1, caption="Objects", pinned = True)

        pan = wx.Panel(item, -1)
        bsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Threshold [nm]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tBlobDist = wx.TextCtrl(pan, -1, '%3.0f' % self.objThreshold,size=(40,-1))
        hsizer.Add(self.tBlobDist, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Min Size [events]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tMinObjSize = wx.TextCtrl(pan, -1, '%d' % self.objMinSize, size=(40, -1))
        hsizer.Add(self.tMinObjSize, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Jittering:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tObjJitter = wx.TextCtrl(pan, -1, '%d' % self.blobJitter, size=(40, -1))
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

        self._pnl.AddPane(item)

    def OnSetBlobColour(self, event):
        bcolour = self.cBlobColour.GetStringSelection()

        if bcolour == 'Index':
            c = self.objCInd.astype('f')
        elif bcolour == 'Random':
            r = pylab.rand(self.objCInd.max() + 1)
            c = r[self.objCInd.astype('i')]
        else:
            c = self.objectMeasures[bcolour][self.objCInd.astype('i')]

        self.glCanvas.c = c
        self.glCanvas.setColour()
        self.OnGLViewChanged()
        
        self.displayPane.hlCLim.SetData(self.glCanvas.c, self.glCanvas.clim[0], self.glCanvas.clim[1])

    def OnObjApplyThreshold(self, event):
        self.objects = None
        self.objThreshold = float(self.tBlobDist.GetValue())
        self.objMinSize = int(self.tMinObjSize.GetValue())
        self.blobJitter = int(self.tObjJitter.GetValue())

        #self.bObjMeasure.Enable(True)

        self.RefreshView()

    def OnObjMeasure(self, event):
        self.objectMeasures = objectMeasure.measureObjects(self.objects, self.objThreshold)
        if self.rav == None:
            self.rav = recArrayView.recArrayPanel(self, self.objectMeasures)
            self.AddPage(self.rav, 'Measurements')
        else:
            self.rav.grid.SetData(self.objectMeasures)

        self.cBlobColour.Clear()
        self.cBlobColour.Append('Index')

        for n in self.objectMeasures.dtype.names:
            self.cBlobColour.Append(n)

        self.RefreshView()


    def GenPointsPanel(self, title='Points'):
        item = afp.foldingPane(self._pnl, -1, caption=title, pinned = True)
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

        if not self.colourFilter == None:
            colData += self.colourFilter.keys()

        colData += self.GeneratedMeasures.keys()

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

        self._pnl.AddPane(item)

    def UpdatePointColourChoices(self):
        if self.viewMode == 'points': #only change if we are in points mode
            colData = ['<None>']

            if not self.colourFilter == None:
                colData += self.colourFilter.keys()

            colData += self.GeneratedMeasures.keys()

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
        elif not self.colourFilter == None:
            if self.colData in self.colourFilter.keys():
                pointColour = self.colourFilter[self.colData]
            elif self.colData in self.GeneratedMeasures.keys():
                pointColour = self.GeneratedMeasures[self.colData]
            else:
                pointColour = None

        return pointColour

    

    def CreateMenuBar(self):

        # Make a menubar
        file_menu = wx.Menu()

        ID_OPEN = wx.ID_OPEN
        ID_SAVE_MEASUREMENTS = wx.ID_SAVE
        ID_QUIT = wx.ID_EXIT

        ID_OPEN_RAW = wx.NewId()
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

        ID_ABOUT = wx.ID_ABOUT

        ID_VIEW_3D_POINTS = wx.NewId()
        ID_VIEW_3D_TRIANGS = wx.NewId()
        ID_VIEW_3D_BLOBS = wx.NewId()

        ID_VIEW_BLOBS = wx.NewId()
        
        
        file_menu.Append(ID_OPEN, "&Open")
        file_menu.Append(ID_OPEN_RAW, "Open &Raw/Prebleach Data")
        file_menu.Append(ID_OPEN_CHANNEL, "Open Extra &Channel")
        
        file_menu.AppendSeparator()
        file_menu.Append(ID_SAVE_MEASUREMENTS, "&Save Measurements")

        file_menu.AppendSeparator()
        
        file_menu.Append(ID_QUIT, "&Exit")

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

        self.view3d_menu = wx.Menu()

#        try: #stop us bombing on Mac
#            self.view3d_menu.AppendRadioItem(ID_VIEW_3D_POINTS, '&Points')
#            self.view3d_menu.AppendRadioItem(ID_VIEW_3D_TRIANGS, '&Triangles')
#            self.view3d_menu.AppendRadioItem(ID_VIEW_3D_BLOBS, '&Blobs')
#        except:
        self.view3d_menu.Append(ID_VIEW_3D_POINTS, '&Points')
        self.view3d_menu.Append(ID_VIEW_3D_TRIANGS, '&Triangles')
        self.view3d_menu.Append(ID_VIEW_3D_BLOBS, '&Blobs')

        #self.view3d_menu.Enable(ID_VIEW_3D_TRIANGS, False)
        self.view3d_menu.Enable(ID_VIEW_3D_BLOBS, False)

        #self.view_menu.Check(ID_VIEW_3D_POINTS, True)

        self.gen_menu = wx.Menu()
        renderers.init_renderers(self)

        self.extras_menu = wx.Menu()
        from PYME.Analysis.LMVis import Extras
        Extras.InitPlugins(self)

        help_menu = wx.Menu()
        help_menu.Append(ID_ABOUT, "&About")

        menu_bar = wx.MenuBar()

        menu_bar.Append(file_menu, "&File")
        menu_bar.Append(self.view_menu, "&View")
        menu_bar.Append(self.gen_menu, "&Generate Image")
        menu_bar.Append(self.extras_menu, "&Extras")
        menu_bar.Append(self.view3d_menu, "View &3D")
            
        menu_bar.Append(help_menu, "&Help")

        self.Bind(wx.EVT_MENU, self.OnAbout, id=ID_ABOUT)
        self.Bind(wx.EVT_MENU, self.OnQuit, id=ID_QUIT)
        self.Bind(wx.EVT_MENU, self.OnToggleWindow, id=ID_TOGGLE_SETTINGS)

        self.Bind(wx.EVT_MENU, self.OnOpenFile, id=ID_OPEN)
        self.Bind(wx.EVT_MENU, self.OnOpenChannel, id=ID_OPEN_CHANNEL)
        self.Bind(wx.EVT_MENU, self.OnOpenRaw, id=ID_OPEN_RAW)

        self.Bind(wx.EVT_MENU, self.OnSaveMeasurements, id=ID_SAVE_MEASUREMENTS)

        self.Bind(wx.EVT_MENU, self.OnViewPoints, id=ID_VIEW_POINTS)
        self.Bind(wx.EVT_MENU, self.OnViewTriangles, id=ID_VIEW_TRIANGS)
        self.Bind(wx.EVT_MENU, self.OnViewQuads, id=ID_VIEW_QUADS)
        self.Bind(wx.EVT_MENU, self.OnViewVoronoi, id=ID_VIEW_VORONOI)
        self.Bind(wx.EVT_MENU, self.OnViewInterpTriangles, id=ID_VIEW_INTERP_TRIANGS)

        self.Bind(wx.EVT_MENU, self.OnViewBlobs, id=ID_VIEW_BLOBS)
        self.Bind(wx.EVT_MENU, self.OnViewTracks, id=ID_VIEW_TRACKS)

        self.Bind(wx.EVT_MENU, self.SetFit, id=ID_VIEW_FIT)
        self.Bind(wx.EVT_MENU, self.OnFitROI, id=ID_VIEW_FIT_ROI)

        self.Bind(wx.EVT_MENU, self.OnView3DPoints, id=ID_VIEW_3D_POINTS)
        self.Bind(wx.EVT_MENU, self.OnView3DTriangles, id=ID_VIEW_3D_TRIANGS)
        #self.Bind(wx.EVT_MENU, self.OnView3DBlobs, id=ID_VIEW_3D_BLOBS)

        return menu_bar

    def OnViewPoints(self,event):
        self.viewMode = 'points'
        #self.glCanvas.cmap = pylab.cm.hsv
        self.RefreshView()
        self.CreateFoldPanel()
        self.dispPanel.OnPercentileCLim(None)

    def OnViewTracks(self,event):
        self.viewMode = 'tracks'
        #self.glCanvas.cmap = pylab.cm.hsv
        self.RefreshView()
        self.CreateFoldPanel()
        self.dispPanel.OnPercentileCLim(None)

    def OnViewBlobs(self,event):
        self.viewMode = 'blobs'
        self.RefreshView()
        self.CreateFoldPanel()
        #self.OnPercentileCLim(None)

    def OnViewTriangles(self,event):
        self.viewMode = 'triangles'
        self.RefreshView()
        self.CreateFoldPanel()
        self.dispPanel.OnPercentileCLim(None)

    def OnViewQuads(self,event):
        self.viewMode = 'quads'
        self.RefreshView()
        self.CreateFoldPanel()
        self.dispPanel.OnPercentileCLim(None)

    def OnViewVoronoi(self,event):
        self.viewMode = 'voronoi'
        self.RefreshView()
        self.CreateFoldPanel()
        self.dispPanel.OnPercentileCLim(None)

    def OnViewInterpTriangles(self,event):
        self.viewMode = 'interp_triangles'
        self.RefreshView()
        self.CreateFoldPanel()
        self.dispPanel.OnPercentileCLim(None)

    def OnView3DPoints(self,event):
        if 'z' in self.colourFilter.keys():
            if not 'glCanvas3D' in dir(self):
                self.glCanvas3D = gl_render3D.LMGLCanvas(self)
                self.AddPage(page=self.glCanvas3D, select=True, caption='3D')

            self.glCanvas3D.setPoints(self.colourFilter['x'], self.colourFilter['y'], self.colourFilter['z'], self.pointColour())
            self.glCanvas3D.setCLim(self.glCanvas.clim, (-5e5, -5e5))

    def OnView3DTriangles(self,event):
        if 'z' in self.colourFilter.keys():
            if not 'glCanvas3D' in dir(self):
                self.glCanvas3D = gl_render3D.LMGLCanvas(self)
                self.AddPage(page=self.glCanvas3D, select=True, caption='3D')

            self.glCanvas3D.setTriang(self.colourFilter['x'], self.colourFilter['y'], self.colourFilter['z'], 'z', sizeCutoff=self.glCanvas3D.edgeThreshold)
            self.glCanvas3D.setCLim(self.glCanvas3D.clim, (0, 5e-5))

    def genNeighbourDists(self, forceRetriang = False):
        bCurr = wx.BusyCursor()

        if self.Triangles == None or forceRetriang:
                statTri = statusLog.StatusLogger("Generating Triangulation ...")
                self.Triangles = delaunay.Triangulation(self.colourFilter['x'] + .1*np.random.normal(size=len(self.colourFilter['x'])), self.colourFilter['y']+ .1*np.random.normal(size=len(self.colourFilter['x'])))

        statNeigh = statusLog.StatusLogger("Calculating mean neighbour distances ...")
        self.GeneratedMeasures['neighbourDistances'] = pylab.array(visHelpers.calcNeighbourDists(self.Triangles))
    
    def OnSaveMeasurements(self, event):
        fdialog = wx.FileDialog(None, 'Save measurements ...',
            wildcard='Numpy array|*.npy|Tab formatted text|*.txt', style=wx.SAVE|wx.HIDE_READONLY)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            outFilename = fdialog.GetPath().encode()

            if outFilename.endswith('.txt'):
                of = open(outFilename, 'w')
                of.write('\t'.join(self.objectMeasures.dtype.names) + '\n')

                for obj in self.objectMeasures:
                    of.write('\t'.join([repr(v) for v in obj]) + '\n')
                of.close()

            else:
                np.save(outFilename, self.objectMeasures)


    def OnOpenFile(self, event):
        filename = wx.FileSelector("Choose a file to open", nameUtils.genResultDirectoryPath(), default_extension='h5r', wildcard='PYME Results Files (*.h5r)|*.h5r|Tab Formatted Text (*.txt)|*.txt|Matlab data (*.mat)|*.mat')

        #print filename
        if not filename == '':
            self.OpenFile(filename)

    def OpenFile(self, filename):
        while len(self.filesToClose) > 0:
            self.filesToClose.pop().close()
        
        self.dataSources = []
        if 'zm' in dir(self):
            del self.zm
        self.filter = None
        self.mapping = None
        self.colourFilter = None
        self.filename = filename
        #print os.path.splitext(filename)[1]
        if os.path.splitext(filename)[1] == '.h5r':
                try:
                    self.selectedDataSource = inpFilt.h5rSource(filename)
                    self.dataSources.append(self.selectedDataSource)

                    self.filesToClose.append(self.selectedDataSource.h5f)

                    if 'DriftResults' in self.selectedDataSource.h5f.root:
                        self.dataSources.append(inpFilt.h5rDSource(self.selectedDataSource.h5f))

                        if len(self.selectedDataSource['x']) == 0:
                            self.selectedDataSource = self.dataSources[-1]

                except:
                    self.selectedDataSource = inpFilt.h5rDSource(filename)
                    self.dataSources.append(self.selectedDataSource)
                    
                    self.filesToClose.append(self.selectedDataSource.h5f)

                #once we get around to storing the some metadata with the results
                if 'MetaData' in self.selectedDataSource.h5f.root:
                    self.mdh = MetaDataHandler.HDFMDHandler(self.selectedDataSource.h5f)

                    if 'Camera.ROIWidth' in self.mdh.getEntryNames():
                        x0 = 0
                        y0 = 0

                        x1 = self.mdh.getEntry('Camera.ROIWidth')*1e3*self.mdh.getEntry('voxelsize.x')
                        y1 = self.mdh.getEntry('Camera.ROIHeight')*1e3*self.mdh.getEntry('voxelsize.y')

                        if 'Splitter' in self.mdh.getEntry('Analysis.FitModule'):
                            y1 = y1/2

                        self.imageBounds = ImageBounds(x0, y0, x1, y1)
                    else:
                        self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)

                else:
                    self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)

                if not self.elv == None: #remove previous event viewer
                    i = 0
                    found = False
                    while not found and i < self.notebook.GetPageCount():
                        if self.notebook.GetPage(i) == self.elv:
                            self.notebook.DeletePage(i)
                            found = True
                        else:
                            i += 1

                

                if 'fitResults_Ag' in self.selectedDataSource.keys():
                    #if we used the splitter set up a mapping so we can filter on total amplitude and ratio
                    #if not 'fitError_Ag' in self.selectedDataSource.keys():

                    if 'fitError_Ag' in self.selectedDataSource.keys():
                        self.selectedDataSource = inpFilt.mappingFilter(self.selectedDataSource, A='fitResults_Ag + fitResults_Ar', gFrac='fitResults_Ag/(fitResults_Ag + fitResults_Ar)', error_gFrac = 'sqrt((fitError_Ag/fitResults_Ag)**2 + (fitError_Ag**2 + fitError_Ar**2)/(fitResults_Ag + fitResults_Ar)**2)*fitResults_Ag/(fitResults_Ag + fitResults_Ar)')
                        sg = self.selectedDataSource['fitError_Ag']
                        sr = self.selectedDataSource['fitError_Ar']
                        g = self.selectedDataSource['fitResults_Ag']
                        r = self.selectedDataSource['fitResults_Ar']
                        I = self.selectedDataSource['A']
                        self.selectedDataSource.colNorm = np.sqrt(2*np.pi)*sg*sr/(2*np.sqrt(sg**2 + sr**2)*I)*(
                            scipy.special.erf((sg**2*r + sr**2*(I-g))/(np.sqrt(2)*sg*sr*np.sqrt(sg**2+sr**2)))
                            - scipy.special.erf((sg**2*(r-I) - sr**2*g)/(np.sqrt(2)*sg*sr*np.sqrt(sg**2+sr**2))))
                        self.selectedDataSource.setMapping('ColourNorm', '1.0*colNorm')
                    else:
                        self.selectedDataSource = inpFilt.mappingFilter(self.selectedDataSource, A='fitResults_Ag + fitResults_Ar', gFrac='fitResults_Ag/(fitResults_Ag + fitResults_Ar)', error_gFrac = '0*x + 0.01')
                        self.selectedDataSource.setMapping('fitError_Ag', '1*sqrt(fitResults_Ag/1)')
                        self.selectedDataSource.setMapping('fitError_Ar', '1*sqrt(fitResults_Ar/1)')
                        sg = self.selectedDataSource['fitError_Ag']
                        sr = self.selectedDataSource['fitError_Ar']
                        g = self.selectedDataSource['fitResults_Ag']
                        r = self.selectedDataSource['fitResults_Ar']
                        I = self.selectedDataSource['A']
                        self.selectedDataSource.colNorm = np.sqrt(2*np.pi)*sg*sr/(2*np.sqrt(sg**2 + sr**2)*I)*(
                            scipy.special.erf((sg**2*r + sr**2*(I-g))/(np.sqrt(2)*sg*sr*np.sqrt(sg**2+sr**2)))
                            - scipy.special.erf((sg**2*(r-I) - sr**2*g)/(np.sqrt(2)*sg*sr*np.sqrt(sg**2+sr**2))))
                        self.selectedDataSource.setMapping('ColourNorm', '1.0*colNorm')

                    self.dataSources.append(self.selectedDataSource)

                    if not self.colp == None: #remove previous colour viewer
                        i = 0
                        found = False
                        while not found and i < self.notebook.GetPageCount():
                            if self.notebook.GetPage(i) == self.colp:
                                self.notebook.DeletePage(i)
                                found = True
                            else:
                                i += 1

                    self.colp = colourPanel.colourPanel(self, self)
#                    if 'Sample.Labelling' in self.mdh.getEntryNames():
#                        self.colp.SpecFromMetadata(self.mdh)
                    self.AddPage(self.colp, caption='Colour')
                elif 'fitResults_sigxl' in self.selectedDataSource.keys():
                    self.selectedDataSource = inpFilt.mappingFilter(self.selectedDataSource)
                    self.dataSources.append(self.selectedDataSource)

                    self.selectedDataSource.setMapping('sig', 'fitResults_sigxl + fitResults_sigyu')
                    self.selectedDataSource.setMapping('sig_d', 'fitResults_sigxl - fitResults_sigyu')

                    self.selectedDataSource.dsigd_dz = -30.
                    self.selectedDataSource.setMapping('fitResults_z0', 'dsigd_dz*sig_d')
                else:
                    self.selectedDataSource = inpFilt.mappingFilter(self.selectedDataSource)
                    self.dataSources.append(self.selectedDataSource)
    
                    

                if ('Events' in self.selectedDataSource.resultsSource.h5f.root) and ('StartTime' in self.mdh.keys()):
                    self.events = self.selectedDataSource.resultsSource.h5f.root.Events[:]

                    self.elv = eventLogViewer.eventLogPanel(self, self.events, self.mdh, [0, self.selectedDataSource['tIndex'].max()]);
                    self.AddPage(self.elv, caption='Events')

                    evKeyNames = set()
                    for e in self.events:
                        evKeyNames.add(e['EventName'])

                    charts = []

                    if 'ProtocolFocus' in evKeyNames:
                        self.zm = piecewiseMapping.GeneratePMFromEventList(self.events, self.mdh, self.mdh.getEntry('StartTime'), self.mdh.getEntry('Protocol.PiezoStartPos'))
                        self.z_focus = 1.e3*self.zm(self.selectedDataSource['t'])
                        #self.elv.SetCharts([('Focus [um]', self.zm, 'ProtocolFocus'),])
                        charts.append(('Focus [um]', self.zm, 'ProtocolFocus'))

                        self.selectedDataSource.z_focus = self.z_focus
                        self.selectedDataSource.setMapping('focus', 'z_focus')

                    if 'ScannerXPos' in self.elv.evKeyNames:
                        x0 = 0
                        if 'Positioning.Stage_X' in self.mdh.getEntryNames():
                            x0 = self.mdh.getEntry('Positioning.Stage_X')
                        self.xm = piecewiseMapping.GeneratePMFromEventList(self.elv.eventSource, self.mdh, self.mdh.getEntry('StartTime'), x0, 'ScannerXPos', 0)
                        charts.append(('XPos [um]', self.xm, 'ScannerXPos'))

                        self.selectedDataSource.scan_x = 1.e3*self.xm(self.selectedDataSource['t']-.01)
                        self.selectedDataSource.setMapping('ScannerX', 'scan_x')
                        self.selectedDataSource.setMapping('x', 'x + scan_x')

                    if 'ScannerYPos' in self.elv.evKeyNames:
                        y0 = 0
                        if 'Positioning.Stage_Y' in self.mdh.getEntryNames():
                            y0 = self.mdh.getEntry('Positioning.Stage_Y')
                        self.ym = piecewiseMapping.GeneratePMFromEventList(self.elv.eventSource, self.mdh, self.mdh.getEntry('StartTime'), y0, 'ScannerYPos', 0)
                        charts.append(('YPos [um]', self.ym, 'ScannerYPos'))

                        self.selectedDataSource.scan_y = 1.e3*self.ym(self.selectedDataSource['t']-.01)
                        self.selectedDataSource.setMapping('ScannerY', 'scan_y')
                        self.selectedDataSource.setMapping('y', 'y + scan_y')

                    if 'ScannerXPos' in self.elv.evKeyNames or 'ScannerYPos' in self.elv.evKeyNames:
                        self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)

                    self.elv.SetCharts(charts)

                if not 'foreShort' in dir(self.selectedDataSource):
                    self.selectedDataSource.foreShort = 1.

                if not 'focus' in self.selectedDataSource.mappings.keys():
                    self.selectedDataSource.focus= np.zeros(self.selectedDataSource['x'].shape)
                    
                if 'fitResults_z0' in self.selectedDataSource.keys():
                    self.selectedDataSource.setMapping('z', 'fitResults_z0 + foreShort*focus')
                else:
                    self.selectedDataSource.setMapping('z', 'foreShort*focus')

                if not self.mdp == None: #remove previous colour viewer
                    i = 0
                    found = False
                    while not found and i < self.notebook.GetPageCount():
                        if self.notebook.GetPage(i) == self.mdp:
                            self.notebook.DeletePage(i)
                            found = True
                        else:
                            i += 1

                if 'mdh' in dir(self):
                    self.mdp = MetadataTree.MetadataPanel(self, self.mdh, editable=False)
                    self.AddPage(self.mdp, caption='Metadata')
                        
        elif os.path.splitext(filename)[1] == '.mat': #matlab file
            from scipy.io import loadmat
            mf = loadmat(filename)

            dlg = importTextDialog.ImportMatDialog(self, [k for k in mf.keys() if not k.startswith('__')])

            ret = dlg.ShowModal()

            if not ret == wx.ID_OK:
                return #we cancelled

            #try:
            #print dlg.GetFieldNames()
            ds = inpFilt.matfileSource(filename, dlg.GetFieldNames(), dlg.GetVarName())
            self.selectedDataSource = ds
            self.dataSources.append(ds)

            self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)
        else: #assume it's a text file
            dlg = importTextDialog.ImportTextDialog(self, filename)

            ret = dlg.ShowModal()

            if not ret == wx.ID_OK:
                return #we cancelled

            #try:
            #print dlg.GetFieldNames()
            ds = inpFilt.textfileSource(filename, dlg.GetFieldNames())
            self.selectedDataSource = ds
            self.dataSources.append(ds)

            self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)

        self.SetTitle('PYME Visualise - ' + filename)
        #for k in self.filterKeys.keys():

        #if we've done a 3d fit
        #print self.selectedDataSource.keys()
        for k in self.filterKeys.keys():
            if not k in self.selectedDataSource.keys():
                self.filterKeys.pop(k)

        #print self.filterKeys
        self.RegenFilter()

        self.CreateFoldPanel()
        if not self.colp == None:
            if 'Sample.Labelling' in self.mdh.getEntryNames():
                self.colp.SpecFromMetadata(self.mdh)
            else:
                self.colp.refresh()
        self.SetFit()

    def OnOpenChannel(self, event):
        filename = wx.FileSelector("Choose a file to open", nameUtils.genResultDirectoryPath(), default_extension='h5r', wildcard='PYME Results Files (*.h5r)|*.h5r|Tab Formatted Text (*.txt)|*.txt')

        #print filename
        if not filename == '':
            self.OpenChannel(filename)

    def OpenChannel(self, filename):
        self.filter = None
        self.mapping = None
        self.colourFilter = None
        print os.path.splitext(filename)[1]
        if os.path.splitext(filename)[1] == '.h5r':
                try:
                    self.selectedDataSource = inpFilt.h5rSource(filename)
                    self.dataSources.append(self.selectedDataSource)

                    self.filesToClose.append(self.selectedDataSource.h5f)

                    if 'DriftResults' in self.selectedDataSource.h5f.root:
                        self.dataSources.append(inpFilt.h5rDSource(self.selectedDataSource.h5f))

                        if len(self.selectedDataSource['x']) == 0:
                            self.selectedDataSource = self.dataSources[-1]
                except:
                    self.selectedDataSource = inpFilt.h5rDSource(filename)
                    self.dataSources.append(self.selectedDataSource)

                    self.filesToClose.append(self.selectedDataSource.h5f)

        else: #assume it's a text file
            dlg = importTextDialog.ImportTextDialog(self)

            ret = dlg.ShowModal()

            if not ret == wx.ID_OK:
                return #we cancelled

            ds = inpFilt.textfileSource(filename, dlg.GetFieldNames())
            self.selectedDataSource = ds
            self.dataSources.append(ds)

        self.RegenFilter()
        self.CreateFoldPanel()
        if not self.colp == None:
            self.colp.refresh()
        self.SetFit()

    def OnOpenRaw(self, event):
        #print self.filename
        tmp = os.path.split(self.filename)[0].split(os.sep)
        #print tmp
        tgtDir = os.sep.join(tmp[:-2] + tmp[-1:])
        print tgtDir
        filename = wx.FileSelector("Choose a file to open", tgtDir, default_extension='h5', wildcard='PYME Spool Files (*.h5)|*.h5|Khoros Data Format (*.kdf)|*.kdf|Tiff (*.tif)|*.tif')
        if not filename == '':
            self.OpenRaw(filename)

    def OpenRaw(self, filename):
        ext = os.path.splitext(filename)[-1]
        if ext == '.kdf': #KDF file
            from PYME.FileUtils import read_kdf
            im = read_kdf.ReadKdfData(filename).squeeze()

            dlg = wx.TextEntryDialog(self, 'Pixel Size [nm]:', 'Please enter the x-y pixel size', '70')
            dlg.ShowModal()

            pixelSize = float(dlg.GetValue())

            imb = ImageBounds(0,0,pixelSize*im.shape[0],pixelSize*im.shape[1])
            
            img = GeneratedImage(im,imb, pixelSize )
            imf = imageView.ImageViewFrame(self,img, self.glCanvas, title=filename,zdim=2)
            self.generatedImages.append(imf)
            imf.Show()
        elif ext == '.h5': #h5 spool
            from PYME.Analysis.DataSources import HDFDataSource
            from PYME.DSView.arrayViewPanel import ArraySettingsAndViewPanel
            dataSource = HDFDataSource.DataSource(filename, None)
            h5f = dataSource.h5File
            #h5f = tables.openFile(filename)

            if 'MetaData' in h5f.root: #should be true the whole time
                md = MetaData.TIRFDefault
                md.copyEntriesFrom(MetaDataHandler.HDFMDHandler(h5f))

            #md = MetaData.genMetaDataFromHDF(h5f)

            
            im = h5f.root.ImageData
            
            self.filesToClose.append(h5f)
            
            pixelSize = md.voxelsize.x*1e3

            imb = ImageBounds(0,0,pixelSize*im.shape[1],pixelSize*im.shape[2])

            img = GeneratedImage(im,imb, pixelSize )
            if 'EstimatedLaserOnFrameNo' in md.getEntryNames():
                zp = min(md.EstimatedLaserOnFrameNo+10,(h5f.root.ImageData.shape[0]-1))
            else:
                zp = 0
            #imf = imageView.ImageViewFrame(self,img, self.glCanvas, title=filename,zp=zp)
            imf = imageView.MultiChannelImageViewFrame(self, self.glCanvas, [img], title=filename, zdim=0, zp=zp)

            self.generatedImages.append(imf)
            imf.Show()

            vp = ArraySettingsAndViewPanel(self, dataSource)
            self.AddPage(page=vp, select=True, caption=filename)
            vp.view.pointMode = 'lm'
            vp.view.vox_x = 1e3*md.getEntry('voxelsize.x')
            vp.view.vox_y = 1e3*md.getEntry('voxelsize.y')
            vp.view.filter = self.colourFilter

            if 'gFrac' in self.colourFilter.keys(): #test for splitter mode
                vp.view.pointMode = 'splitter'

            #vp.view.points = np.vstack((self.colourFilter['x']/voxx, self.colourFilter['y']/voxy, self.colourFilter['t'])).T

            self.vp = vp

        elif ext == '.tif': #Tiff file
            from PYME.FileUtils import readTiff
            im = readTiff.read3DTiff(filename).squeeze()

            xmlfn = os.path.splitext(filename)[0] + '.xml'
            if os.path.exists(xmlfn):
                md = MetaData.TIRFDefault
                md.copyEntriesFrom(MetaDataHandler.XMLMDHandler(xmlfn))
            else:
                md = MetaData.ConfocDefault

                from PYME.DSView.voxSizeDialog import VoxSizeDialog

                dlg = VoxSizeDialog(self)
                dlg.ShowModal()

                md.setEntry('voxelsize.x', dlg.GetVoxX())
                md.setEntry('voxelsize.y', dlg.GetVoxY())
                md.setEntry('voxelsize.z', dlg.GetVoxZ())

            pixelSize = md.voxelsize.x*1e3

            imb = ImageBounds(0,0,pixelSize*im.shape[0],pixelSize*im.shape[1])

            img = GeneratedImage(im,imb, pixelSize )
            imf = imageView.MultiChannelImageViewFrame(self, self.glCanvas, [img], title=filename,zdim=2)
            self.generatedImages.append(imf)
            imf.Show()
        else:
            raise RuntimeError('Unrecognised Data Format')




    def RegenFilter(self):
        if not self.selectedDataSource == None:
            self.filter = inpFilt.resultsFilter(self.selectedDataSource, **self.filterKeys)
            if self.mapping:
                self.mapping.resultsSource = self.filter
            else:
                self.mapping = inpFilt.mappingFilter(self.filter)

            if not self.colourFilter:
                self.colourFilter = inpFilt.colourFilter(self.mapping, self)

        self.filterPane.stFilterNumPoints.SetLabel('%d of %d events' % (len(self.filter['x']), len(self.selectedDataSource['x'])))

        self.Triangles = None
        self.edb = None
        self.objects = None
        self.GeneratedMeasures = {}
        self.Quads = None

        self.RefreshView()


    def RefreshView(self):
        if self.colourFilter == None:
            return #get out of here

        if len(self.colourFilter['x']) == 0:
            wx.MessageBox('No data points - try adjusting the filter', "len(filter['x']) ==0")
            return

        if self.glCanvas.init == 0: #glcanvas is not initialised
            return

        bCurr = wx.BusyCursor()


        if self.objects == None:
#            if 'bObjMeasure' in dir(self):
#                self.bObjMeasure.Enable(False)
            self.objectMeasures = None

            if not self.rav == None: #remove previous event viewer
                i = 0
                found = False
                while not found and i < self.notebook.GetPageCount():
                    if self.notebook.GetPage(i) == self.rav:
                        self.notebook.DeletePage(i)
                        found = True
                    else:
                        i += 1
                        
                self.rav = None

        if self.viewMode == 'points':
            self.glCanvas.setPoints(self.colourFilter['x'], self.colourFilter['y'], self.pointColour())
        elif self.viewMode == 'tracks':
            self.glCanvas.setTracks(self.colourFilter['x'], self.colourFilter['y'], self.colourFilter['clumpIndex'], self.pointColour())
        elif self.viewMode == 'triangles':
            if self.Triangles == None:
                status = statusLog.StatusLogger("Generating Triangulation ...")
                self.Triangles = delaunay.Triangulation(self.colourFilter['x'] + 0.1*np.random.normal(size=self.colourFilter['x'].shape), self.colourFilter['y'] + 0.1*np.random.normal(size=self.colourFilter['x'].shape))
                
            self.glCanvas.setTriang(self.Triangles)

        elif self.viewMode == 'voronoi':
            if self.Triangles == None:
                status = statusLog.StatusLogger("Generating Triangulation ...")
                self.Triangles = delaunay.Triangulation(self.colourFilter['x']+ 0.1*np.random.normal(size=self.colourFilter['x'].shape), self.colourFilter['y']+ 0.1*np.random.normal(size=self.colourFilter['x'].shape))
                

            status = statusLog.StatusLogger("Generating Voronoi Diagram ... ")
            self.glCanvas.setVoronoi(self.Triangles)
            

        elif self.viewMode == 'quads':
            if self.Quads == None:
                status = statusLog.StatusLogger("Generating QuadTree ...")
                self.GenQuads()
                

            self.glCanvas.setQuads(self.Quads)

        elif self.viewMode == 'interp_triangles':
            if self.Triangles == None:
                status = statusLog.StatusLogger("Generating Triangulation ...")
                self.Triangles = delaunay.Triangulation(self.colourFilter['x']+ 0.1*np.random.normal(size=self.colourFilter['x'].shape), self.colourFilter['y']+ 0.1*np.random.normal(size=self.colourFilter['x'].shape))

            self.glCanvas.setIntTriang(self.Triangles, self.pointColour())

        elif self.viewMode == 'blobs':
            if self.objects == None:
                #check to see that we don't have too many points
                if len(self.colourFilter['x']) > 1e5:
                    goAhead = wx.MessageBox('You have %d events in the selected ROI;\nThis could take a LONG time ...' % len(self.colourFilter['x']), 'Continue with blob detection', wx.YES_NO|wx.ICON_EXCLAMATION)

                    if not goAhead == wx.YES:
                        return

                if self.Triangles == None:
                    status = statusLog.StatusLogger("Generating Triangulation ...")
                    self.Triangles = delaunay.Triangulation(self.colourFilter['x']+ 0.1*np.random.normal(size=self.colourFilter['x'].shape), self.colourFilter['y']+ 0.1*np.random.normal(size=self.colourFilter['x'].shape))

                if self.edb == None:
                    self.edb = edges.EdgeDB(self.Triangles)

                if self.blobJitter == 0:
                    #T = delny.Triangulation(pylab.array([self.colourFilter['x'] + 0.1*pylab.randn(len(self.colourFilter['x'])), self.colourFilter['y']+ 0.1*pylab.randn(len(self.colourFilter['x']))]).T)

                    #self.objects = gen3DTriangs.segment(T, self.objThreshold, self.objMinSize)

                    #edb = edges.EdgeDB(self.Triangles)
                    self.objIndices = edges.objectIndices(self.edb.segment(self.objThreshold), self.objMinSize)
                    self.objects = [pylab.vstack((self.Triangles.x[oi], self.Triangles.y[oi])).T for oi in self.objIndices]
                else:
                    if not 'neighbourDistances' in self.GeneratedMeasures.keys():
                        self.genNeighbourDists()
                    x_ = pylab.hstack([self.colourFilter['x'] + 0.5*self.GeneratedMeasures['neighbourDistances']*pylab.randn(len(self.colourFilter['x'])) for i in range(self.blobJitter)])
                    y_ = pylab.hstack([self.colourFilter['y'] + 0.5*self.GeneratedMeasures['neighbourDistances']*pylab.randn(len(self.colourFilter['x'])) for i in range(self.blobJitter)])

                    #T = delny.Triangulation(pylab.array([x_, y_]).T)
                    T = delaunay.Triangulation(x_, y_)
                    #self.objects = gen3DTriangs.segment(T, self.objThreshold, self.objMinSize)
                    edb = edges.EdgeDB(T)
                    objIndices = edges.objectIndices(edb.segment(self.objThreshold), self.objMinSize)
                    self.objects = [pylab.vstack((T.x[oi], T.y[oi])).T for oi in objIndices]

#                if 'bObjMeasure' in dir(self):
#                    self.bObjMeasure.Enable(True)

            self.glCanvas.setBlobs(self.objects, self.objThreshold)
            self.objCInd = self.glCanvas.c

        self.displayPane.hlCLim.SetData(self.glCanvas.c, self.glCanvas.clim[0], self.glCanvas.clim[1])

        if not self.colp == None and self.colp.IsShown():
            self.colp.refresh()

        #self.sh.shell.user_ns.update(self.__dict__)
        wx.EndBusyCursor()
        self.workspaceView.RefreshItems()



    def GenQuads(self):
        di = max(self.imageBounds.x1 - self.imageBounds.x0, self.imageBounds.y1 - self.imageBounds.y0)

        np = di/self.QTGoalPixelSize

        di = self.QTGoalPixelSize*2**pylab.ceil(pylab.log2(np))

        
        self.Quads = pointQT.qtRoot(self.imageBounds.x0, self.imageBounds.x0+di, self.imageBounds.y0, self.imageBounds.y0 + di)

        for xi, yi in zip(self.colourFilter['x'],self.colourFilter['y']):
            self.Quads.insert(pointQT.qtRec(xi,yi, None))

    def SetFit(self,event = None):
        xsc = self.imageBounds.width()*1./self.glCanvas.Size[0]
        ysc = self.imageBounds.height()*1./self.glCanvas.Size[1]

        #print xsc
        #print ysc

        if xsc > ysc:
            self.glCanvas.setView(self.imageBounds.x0, self.imageBounds.x1, self.imageBounds.y0, self.imageBounds.y0 + xsc*self.glCanvas.Size[1])
        else:
            self.glCanvas.setView(self.imageBounds.x0, self.imageBounds.x0 + ysc*self.glCanvas.Size[0], self.imageBounds.y0, self.imageBounds.y1)

    def OnFitROI(self,event = None):
        if 'x' in self.filterKeys.keys():
            xbounds = self.filterKeys['x']
        else:
            xbounds = (self.imageBounds.x0, self.imageBounds.x1)

        if 'y' in self.filterKeys.keys():
            ybounds = self.filterKeys['y']
        else:
            ybounds = (self.imageBounds.y0, self.imageBounds.y1)
        
        xsc = (xbounds[1] - xbounds[0])*1./self.glCanvas.Size[0]
        ysc = (ybounds[1] - ybounds[0])*1./self.glCanvas.Size[1]

        #print xsc
        #print ysc

        if xsc > ysc:
            self.glCanvas.setView(xbounds[0], xbounds[1], ybounds[0], ybounds[0] + xsc*self.glCanvas.Size[1])
        else:
            self.glCanvas.setView(xbounds[0], xbounds[0] + ysc*self.glCanvas.Size[0], ybounds[0], ybounds[1])

    def OnGLViewChanged(self):
        for genI in self.generatedImages:
            genI.Refresh()

    def SetStatus(self, statusText):
        self.statusbar.SetStatusText(statusText, 0)


class VisGuiApp(wx.App):
    def __init__(self, filename, *args):
        self.filename = filename
        wx.App.__init__(self, *args)
        
        
    def OnInit(self):
        wx.InitAllImageHandlers()
        self.main = VisGUIFrame(None, self.filename)
        self.main.Show()
        self.SetTopWindow(self.main)
        return True


def main(filename):
    application = VisGuiApp(filename, 0)
    application.MainLoop()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    filename = None

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    if wx.GetApp() == None: #check to see if there's already a wxApp instance (running from ipython -pylab or -wthread)
        main(filename)
    else:
        #time.sleep(1)
        visFr = VisGUIFrame(None, filename)
        visFr.Show()
        visFr.RefreshView()


