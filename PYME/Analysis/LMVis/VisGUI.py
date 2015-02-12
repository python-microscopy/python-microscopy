#!/usr/bin/python
##################
# VisGUI.py
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

import wx
import wx.py.shell

import PYME.misc.autoFoldPanel as afp
import wx.lib.agw.aui as aui

#hacked so py2exe works
from PYME.DSView.dsviewer_npy import View3D

from PYME.Analysis.LMVis import gl_render
#from PYME.Analysis.LMVis import workspaceTree
import sys

import pylab
from PYME.misc import extraCMaps
from PYME.FileUtils import nameUtils

import os
from PYME.Analysis.LMVis import gl_render3D

from PYME.Analysis.LMVis import colourPanel
from PYME.Analysis.LMVis import renderers
from PYME.Analysis.LMVis import pipeline

try:
    from PYME.Analysis.LMVis import recArrayView
except:
    pass

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

from PYME.Analysis import MetadataTree

import numpy as np
import scipy.special

from PYME.DSView import eventLogViewer

from PYME.Analysis.LMVis import statusLog

class VisGUIFrame(wx.Frame):    
    def __init__(self, parent, filename=None, id=wx.ID_ANY, 
                 title="PYME Visualise", pos=wx.DefaultPosition,
                 size=(700,650), style=wx.DEFAULT_FRAME_STYLE):

        wx.Frame.__init__(self, parent, id, title, pos, size, style)
        self._mgr = aui.AuiManager(agwFlags = aui.AUI_MGR_DEFAULT | aui.AUI_MGR_AUTONB_NO_CAPTION)
        atabstyle = self._mgr.GetAutoNotebookStyle()
        self._mgr.SetAutoNotebookStyle((atabstyle ^ aui.AUI_NB_BOTTOM) | aui.AUI_NB_TOP)
        self._mgr.SetManagedWindow(self)

        self._flags = 0
        
        self.pipeline = pipeline.Pipeline(visFr=self)
        
        #self.Quads = None
               
        self.SetMenuBar(self.CreateMenuBar())

        self.statusbar = self.CreateStatusBar(1, wx.ST_SIZEGRIP)

        self.statusbar.SetStatusText("", 0)
       
        self._leftWindow1 = wx.Panel(self, -1, size = wx.Size(220, 1000))
        self._pnl = 0

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

        self.generatedImages = []

        self.viewMode = 'points' #one of points, triangles, quads, or voronoi
        self.colData = 't'
        
#        if 'PYME_BUGGYOPENGL' in os.environ.keys():
#            pylab.plot(pylab.randn(10))

        self.sh.Execute('from pylab import *')
        self.sh.Execute('from PYME.DSView.dsviewer_npy import View3D')

        #self.workspace = workspaceTree.WorkWrap(self.__dict__)
        ##### Make certain things visible in the workspace tree

        #components of the pipeline
        #col = self.workspace.newColour()
        #self.workspace.addKey('pipeline', col)
        
        #Generated stuff
        #col = self.workspace.newColour()
        #self.workspace.addKey('GeneratedMeasures', col)
        #self.workspace.addKey('generatedImages', col)
        #self.workspace.addKey('objects', col)

        #main window, so we can get everything else if needed
        #col = self.workspace.newColour()
        #self.workspace.addKey('MainWindow', col)

        ######

        #self.workspaceView = workspaceTree.WorkspaceTree(self, workspace=self.workspace, shell=self.sh)
        #self.AddPage(page=wx.StaticText(self, -1, 'foo'), select=False, caption='Workspace')

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

        self._mgr.Update()

        if not filename==None:
            self.OpenFile(filename)

        nb = self._mgr.GetNotebooks()[0]
        nb.SetSelection(0)
        
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

    def AddPage(self, page=None, select=False,caption='Dummy', update=True):
        if update:
            self._mgr.Update()
        pn = self._mgr.GetPaneByName("Shell")
        if pn.IsNotebookPage():
            #print pn.notebook_id
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

        if update:        
            self._mgr.Update()

    def OnMove(self, event):
        self.Refresh()
        self.Update()
        event.Skip()      

    def OnQuit(self, event):
        while len(self.pipeline.filesToClose) > 0:
            self.pipeline.filesToClose.pop().close()

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
        
    def SaveMetadata(self, mdh):
        mdh['Filter.Keys'] = self.pipeline.filterKeys      
        
        if HAVE_DRIFT_CORRECTION and 'x' in self.pipeline.mapping.mappings.keys(): #drift correction has been applied
            self.driftPane.dp.SaveMetadata(mdh)

    def CreateFoldPanel(self):
        # delete earlier panel
        self._leftWindow1.DestroyChildren()

        # recreate the foldpanelbar
        hsizer = wx.BoxSizer(wx.VERTICAL)

        s = self._leftWindow1.GetBestSize()

        self._pnl = afp.foldPanel(self._leftWindow1, -1, wx.DefaultPosition,s)

        self.GenDataSourcePanel()

        self.filterPane = CreateFilterPane(self._pnl, self.pipeline.filterKeys, self.pipeline, self)

        if HAVE_DRIFT_CORRECTION:
            self.driftPane = CreateDriftPane(self._pnl, self.pipeline.mapping, self.pipeline)
            
        self.colourFilterPane = CreateColourFilterPane(self._pnl, self.pipeline.colourFilter, self.pipeline)
        self.displayPane = CreateDisplayPane(self._pnl, self.glCanvas, self)
        
        if self.viewMode == 'quads':
            self.GenQuadTreePanel()

        if self.viewMode == 'points' or self.viewMode == 'tracks':
            pass
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
        for ds in self.pipeline.dataSources:
            rbid = wx.NewId()
            self.dsRadioIds.append(rbid)
            rb = wx.RadioButton(item, rbid, ds._name)
            rb.SetValue(ds == self.pipeline.selectedDataSource)

            rb.Bind(wx.EVT_RADIOBUTTON, self.OnSourceChange)
            item.AddNewElement(rb)

        self._pnl.AddPane(item)


    def OnSourceChange(self, event):
        dsind = self.dsRadioIds.index(event.GetId())
        self.pipeline.selectedDataSource = self.pipeline.dataSources[dsind]
        self.RegenFilter()
        
    
    def GenQuadTreePanel(self):
        from PYME.Analysis.QuadTree import pointQT
        
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
        from PYME.Analysis.QuadTree import pointQT
        
        leafSize = int(self.tQTLeafSize.GetValue())
        if not leafSize >= 1:
            raise RuntimeError('QuadTree leaves must be able to contain at least 1 item')

        pointQT.QT_MAXRECORDS = leafSize
        self.stQTSNR.SetLabel('Effective SNR = %3.2f' % pylab.sqrt(pointQT.QT_MAXRECORDS/2.0))

        self.pipeline.Quads = None
        self.RefreshView()


    def GenBlobPanel(self):
#        item = self._pnl.AddFoldPanel("Objects", collapsed=False,
#                                      foldIcons=self.Images)
        item = afp.foldingPane(self._pnl, -1, caption="Objects", pinned = True)

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

        self._pnl.AddPane(item)

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

    def OnObjApplyThreshold(self, event):
        self.pipeline.objects = None
        self.pipeline.objThreshold = float(self.tBlobDist.GetValue())
        self.pipeline.objMinSize = int(self.tMinObjSize.GetValue())
        self.pipeline.blobJitter = int(self.tObjJitter.GetValue())

        #self.bObjMeasure.Enable(True)

        self.RefreshView()

    def OnObjMeasure(self, event):
        om = self.pipeline.measureObjects()
        if self.rav == None:
            self.rav = recArrayView.recArrayPanel(self, om)
            self.AddPage(self.rav, 'Measurements')
        else:
            self.rav.grid.SetData(om)

        self.cBlobColour.Clear()
        self.cBlobColour.Append('Index')

        for n in om.dtype.names:
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

        self._pnl.AddPane(item)

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
        
        try:
            #see if we can find any 'non free' plugins
            from PYMEnf.Analysis.LMVis import Extras
            Extras.InitPlugins(self)
        except ImportError:
            pass

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

    def OnView3DPoints(self,event):
        if 'z' in self.pipeline.keys():
            if not 'glCanvas3D' in dir(self):
                #self.glCanvas3D = gl_render3D.LMGLCanvas(self)
                #self.AddPage(page=self.glCanvas3D, select=True, caption='3D')
                self.glCanvas3D = gl_render3D.showGLFrame()

            #else:            
            self.glCanvas3D.setPoints(self.pipeline['x'], 
                                  self.pipeline['y'], 
                                  self.pipeline['z'], 
                                  self.pointColour())
            self.glCanvas3D.setCLim(self.glCanvas.clim, (-5e5, -5e5))

    def OnView3DTriangles(self,event):
        if 'z' in self.pipeline.keys():
            if not 'glCanvas3D' in dir(self):
                #self.glCanvas3D = gl_render3D.LMGLCanvas(self)
                #self.AddPage(page=self.glCanvas3D, select=True, caption='3D')
                self.glCanvas3D = gl_render3D.showGLFrame()

            self.glCanvas3D.setTriang(self.pipeline['x'], 
                                      self.pipeline['y'], 
                                      self.pipeline['z'], 'z', 
                                      sizeCutoff=self.glCanvas3D.edgeThreshold)
                                      
            self.glCanvas3D.setCLim(self.glCanvas3D.clim, (0, 5e-5))

   
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
        filename = wx.FileSelector("Choose a file to open", 
                                   nameUtils.genResultDirectoryPath(), 
                                   default_extension='h5r', 
                                   wildcard='PYME Results Files (*.h5r)|*.h5r|Tab Formatted Text (*.txt)|*.txt|Matlab data (*.mat)|*.mat')

        #print filename
        if not filename == '':
            self.OpenFile(filename)
            
    @property
    def notebook(self):
        return self._mgr.GetNotebooks()[0]
            
    def _removeOldTabs(self):
        if not self.elv == None: #remove previous event viewer
            i = 0
            found = False
            while not found and i < self.notebook.GetPageCount():
                if self.notebook.GetPage(i) == self.elv:
                    self.notebook.DeletePage(i)
                    found = True
                else:
                    i += 1
                    
        if not self.colp == None: #remove previous colour viewer
            i = 0
            found = False
            while not found and i < self.notebook.GetPageCount():
                if self.notebook.GetPage(i) == self.colp:
                    self.notebook.DeletePage(i)
                    found = True
                else:
                    i += 1
                    
        if not self.mdp == None: #remove previous metadata viewer
            i = 0
            found = False
            while not found and i < self.notebook.GetPageCount():
                if self.notebook.GetPage(i) == self.mdp:
                    self.notebook.DeletePage(i)
                    found = True
                else:
                    i += 1
                    
    def _createNewTabs(self):
        #print 'md'
        self.mdp = MetadataTree.MetadataPanel(self, self.pipeline.mdh, editable=False)
        self.AddPage(self.mdp, caption='Metadata')
        
        #print 'cp'        
        if 'gFrac' in self.pipeline.filter.keys():
            self.colp = colourPanel.colourPanel(self, self.pipeline, self)
            self.AddPage(self.colp, caption='Colour', update=False)
            
        #print 'ev'
        if not self.pipeline.events == None:
            self.elv = eventLogViewer.eventLogPanel(self, self.pipeline.events, 
                                                        self.pipeline.mdh, 
                                                        [0, self.pipeline.selectedDataSource['tIndex'].max()])
    
            self.elv.SetCharts(self.pipeline.eventCharts)
            
            self.AddPage(self.elv, caption='Events', update=False)
            
        #print 'ud'
        self._mgr.Update()
            
        
            
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
        self.SetTitle('PYME Visualise - ' + filename)
        
        #############################
        #now do all the gui stuff
        self._removeOldTabs()
        self._createNewTabs()
        
        self.CreateFoldPanel()
        print('Gui stuff done')
        
        self.SetFit()
            

    def OnOpenChannel(self, event):
        filename = wx.FileSelector("Choose a file to open", 
                                   nameUtils.genResultDirectoryPath(), 
                                   default_extension='h5r', 
                                   wildcard='PYME Results Files (*.h5r)|*.h5r|Tab Formatted Text (*.txt)|*.txt')

        #print filename
        if not filename == '':
            self.OpenChannel(filename)

    def OnOpenRaw(self, event):
        from PYME.DSView import ViewIm3D, ImageStack
        ViewIm3D(ImageStack(), mode='visGUI', glCanvas=self.glCanvas)
        
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
        self.Bind(wx.EVT_MENU, callback, id=ID_NEWITEM)
        


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

        if not self.colp == None and self.colp.IsShown():
            self.colp.refresh()

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

    #def OnGLViewChanged(self):
    #    for genI in self.generatedImages:
    #        genI.Refresh()

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


