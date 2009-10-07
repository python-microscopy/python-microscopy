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

#!/usr/bin/python
import os.path
import wx
import wx.py.shell

import wx.lib.foldpanelbar as fpb
from PYME.misc.fbpIcons import *

from PYME.Analysis.LMVis import gl_render
import sys
from PYME.Analysis.LMVis import inpFilt
from PYME.Analysis.LMVis import editFilterDialog
import pylab
from PYME.FileUtils import nameUtils
import os


try:
    import delaunay as delny
except:
    pass

from scikits import delaunay

from PYME.Analysis.QuadTree import pointQT, QTrend
import Image

from PYME.Analysis.LMVis import genImageDialog
from PYME.Analysis.LMVis import importTextDialog
from PYME.Analysis.LMVis import visHelpers
from PYME.Analysis.LMVis import imageView
from PYME.Analysis.LMVis import histLimits
try:
    from PYME.Analysis.LMVis import gen3DTriangs
    from PYME.Analysis.LMVis import recArrayView
    from PYME.Analysis.LMVis import objectMeasure
except:
    pass

from PYME.Analysis import intelliFit
from PYME.Analysis import piecewiseMapping

#import time
import numpy as np

import tables
from PYME.Analysis import MetaData
from PYME.Acquire import MetaDataHandler

from PYME.DSView import eventLogViewer

#import threading

from PYME.misc import editList
from PYME.misc.auiFloatBook import AuiNotebookWithFloatingPages

from PYME.Analysis.LMVis import statusLog
#from IPython.frontend.wx.wx_frontend import WxController
#from IPython.kernel.core.interpreter import Interpreter


# ----------------------------------------------------------------------------
# Visualisation of analysed localisation microscopy data
#
# David Baddeley 2009
#
# Some of the code in this file borrowed from the wxPython examples
# ----------------------------------------------------------------------------

from PYME.Analysis.LMVis.visHelpers import ImageBounds, GeneratedImage


class VisGUIFrame(wx.Frame):
    
    def __init__(self, parent, filename=None, id=wx.ID_ANY, title="PYME Visualise", pos=wx.DefaultPosition,
                 size=(700,650), style=wx.DEFAULT_FRAME_STYLE):

        wx.Frame.__init__(self, parent, id, title, pos, size, style)

        self._flags = 0
        
        #self.SetIcon(GetMondrianIcon())
        
        self.SetMenuBar(self.CreateMenuBar())

        self.statusbar = self.CreateStatusBar(1, wx.ST_SIZEGRIP)
        #self.statusbar.SetStatusWidths([-4, -4])
        self.statusbar.SetStatusText("", 0)
        #self.statusbar.SetStatusText("", 1)

        self._leftWindow1 = wx.SashLayoutWindow(self, 101, wx.DefaultPosition,
                                                wx.Size(200, 1000), wx.NO_BORDER |
                                                wx.SW_3D | wx.CLIP_CHILDREN)

        self._leftWindow1.SetDefaultSize(wx.Size(220, 1000))
        self._leftWindow1.SetOrientation(wx.LAYOUT_VERTICAL)
        self._leftWindow1.SetAlignment(wx.LAYOUT_LEFT)
        self._leftWindow1.SetSashVisible(wx.SASH_RIGHT, True)
        self._leftWindow1.SetExtraBorderSize(10)

        self._pnl = 0

        # will occupy the space not used by the Layout Algorithm
        #self.remainingSpace = wx.Panel(self, -1, style=wx.SUNKEN_BORDER)
        #self.glCanvas = gl_render.LMGLCanvas(self.remainingSpace)
        #self.glCanvas = wx.Panel(self, -1, style=wx.SUNKEN_BORDER)
        self.notebook = AuiNotebookWithFloatingPages(id=-1, parent=self, style=wx.aui.AUI_NB_TAB_SPLIT|wx.aui.AUI_NB_TAB_SPLIT|wx.aui.AUI_NB_TAB_SPLIT)

        self.sh = wx.py.shell.Shell(id=-1,
              parent=self.notebook, size=wx.Size(-1, -1), style=0, locals=self.__dict__,
              introText='Python SMI bindings - note that help, license etc below is for Python, not PySMI\n\n')

        self.notebook.AddPage(page=self.sh, select=True, caption='Console')

        self.glCanvas = gl_render.LMGLCanvas(self.notebook)
        self.notebook.AddPage(page=self.glCanvas, select=True, caption='View')
        self.glCanvas.cmap = pylab.cm.hot

        self.elv = None
        self.rav = None

        self.ID_WINDOW_TOP = 100
        self.ID_WINDOW_LEFT1 = 101
        self.ID_WINDOW_RIGHT1 = 102
        self.ID_WINDOW_BOTTOM = 103
    
        self._leftWindow1.Bind(wx.EVT_SASH_DRAGGED_RANGE, self.OnFoldPanelBarDrag,
                               id=100, id2=103)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_MOVE, self.OnMove)

        
        self._pc_clim_change = False

        self.filesToClose = []
        self.generatedImages = []

        self.dataSources = []
        self.selectedDataSource = None
        self.filterKeys = {'error_x': (0,30), 'A':(5,2000), 'sig' : (150/2.35, 350/2.35)}

        self.filter = None
        self.mapping = None

        self.driftCorrParams = {}
        self.driftCorrFcn = None
        self.optimiseFcn = 'fmin'
        self.driftExprX = 'x + a*t'
        self.driftExprY = 'y + b*t'

        self.objThreshold = 30
        self.objMinSize = 10
        self.blobJitter = 0
        self.objects = None

        self.imageBounds = ImageBounds(0,0,0,0)

        #generated Quad-tree will allow visualisations with pixel sizes of self.QTGoalPixelSize*2^N for any N
        self.QTGoalPixelSize = 5 #nm

        self.scaleBarLengths = {'<None>':None, '50nm':50,'200nm':200, '500nm':500, '1um':1000, '5um':5000}


        self.viewMode = 'points' #one of points, triangles, quads, or voronoi
        self.Triangles = None
        self.GeneratedMeasures = {}
        self.Quads = None
        self.pointColour = None

        statusLog.SetStatusDispFcn(self.SetStatus)

        self.CreateFoldPanel()

        if not filename==None:
            #self.glCanvas.OnPaint(None)
            self.OpenFile(filename)

        wx.LayoutAlgorithm().LayoutWindow(self, self.notebook)

        print 'about to refresh'
        self.Refresh()

#        namespace = dict()
#
#        namespace['visFr'] = self
#        namespace['filter'] = self.filter
#        namespace['mapping'] = self.mapping
#        namespace['GeneratedMeasures'] = self.GeneratedMeasures
#
#        #namespace['Testfunc'] = Testfunc
#        self.interp = Interpreter(user_ns=namespace)
#
#        self.f = wx.Frame(self, -1, 'IPython Console', size=(600, 500), pos=(700, 100))
#        self.sh = WxController(self.f, wx.ID_ANY, shell=self.interp)
#        self.sh.SetSize((600,500))
#        sizer = wx.BoxSizer(wx.VERTICAL)
#        sizer.Add(self.sh, 1, wx.EXPAND)
#        self.f.SetSizer(sizer)
#        self.f.Show()

        

    def OnSize(self, event):

        wx.LayoutAlgorithm().LayoutWindow(self, self.notebook)
        event.Skip()

    def OnMove(self, event):
        #pass
        self.Refresh()
        event.Skip()
        

    def OnQuit(self, event):
        while len(self.filesToClose) > 0:
            self.filesToClose.pop().close()
 
        self.Destroy()


    def OnAbout(self, event):

        msg = "PYME Visualise\n\n Visualisation of localisation microscopy data\nDavid Baddeley 2009"
              
        dlg = wx.MessageDialog(self, msg, "About PYME Visualise",
                               wx.OK | wx.ICON_INFORMATION)
        dlg.SetFont(wx.Font(8, wx.NORMAL, wx.NORMAL, wx.NORMAL, False, "Verdana"))
        dlg.ShowModal()
        dlg.Destroy()


    def OnToggleWindow(self, event):
        
        self._leftWindow1.Show(not self._leftWindow1.IsShown())
        # Leaves bits of itself behind sometimes
        wx.LayoutAlgorithm().LayoutWindow(self, self.notebook)
        self.glCanvas.Refresh()

        event.Skip()
        

    def OnFoldPanelBarDrag(self, event):

        if event.GetDragStatus() == wx.SASH_STATUS_OUT_OF_RANGE:
            return

        if event.GetId() == self.ID_WINDOW_LEFT1:
            self._leftWindow1.SetDefaultSize(wx.Size(event.GetDragRect().width, 1000))


        # Leaves bits of itself behind sometimes
        wx.LayoutAlgorithm().LayoutWindow(self, self.notebook)
        self.glCanvas.Refresh()

        event.Skip()
        

    def CreateFoldPanel(self):

        # delete earlier panel
        self._leftWindow1.DestroyChildren()

        # recreate the foldpanelbar

        s = self._leftWindow1.GetBestSize()

        self._pnl = fpb.FoldPanelBar(self._leftWindow1, -1, wx.DefaultPosition,
                                     s, fpb.FPB_DEFAULT_STYLE,0)

        self.Images = wx.ImageList(16,16)
        self.Images.Add(GetExpandedIconBitmap())
        self.Images.Add(GetCollapsedIconBitmap())
            
        self.GenDataSourcePanel()
        self.GenFilterPanel()

        self.GenDriftPanel()

        self.GenDisplayPanel()
        
        if self.viewMode == 'quads':
            self.GenQuadTreePanel()

        if self.viewMode == 'points':
            self.GenPointsPanel()

        if self.viewMode == 'blobs':
            self.GenBlobPanel()

        if self.viewMode == 'interp_triangles':
            self.GenPointsPanel('Vertex Colours')
       

        #item = self._pnl.AddFoldPanel("Filters", False, foldIcons=self.Images)
        #item = self._pnl.AddFoldPanel("Visualisation", False, foldIcons=self.Images)
        wx.LayoutAlgorithm().LayoutWindow(self, self.notebook)
        self.glCanvas.Refresh()


    def GenDataSourcePanel(self):
        item = self._pnl.AddFoldPanel("Data Source", collapsed=True,
                                      foldIcons=self.Images)
        
        self.dsRadioIds = []
        for ds in self.dataSources:
            rbid = wx.NewId()
            self.dsRadioIds.append(rbid)
            rb = wx.RadioButton(item, rbid, ds._name)
            rb.SetValue(ds == self.selectedDataSource)

            rb.Bind(wx.EVT_RADIOBUTTON, self.OnSourceChange)
            self._pnl.AddFoldPanelWindow(item, rb, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10) 


    def OnSourceChange(self, event):
        dsind = self.dsRadioIds.index(event.GetId())
        self.selectedDataSource = self.dataSources[dsind]
        self.RegenFilter()

    def GenDisplayPanel(self):
        item = self._pnl.AddFoldPanel("Display", collapsed=False,
                                      foldIcons=self.Images)
        

        #Colourmap
        cmapnames = pylab.cm.cmapnames

        curCMapName = self.glCanvas.cmap.name
        #curCMapName = 'hot'

        cmapReversed = False
        
        if curCMapName[-2:] == '_r':
            cmapReversed = True
            curCMapName = curCMapName[:-2]

        cmInd = cmapnames.index(curCMapName)


        ##
        pan = wx.Panel(item, -1)

        box = wx.StaticBox(pan, -1, 'Colourmap:')
        bsizer = wx.StaticBoxSizer(box)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.cColourmap = wx.Choice(pan, -1, choices=cmapnames)
        self.cColourmap.SetSelection(cmInd)

        hsizer.Add(self.cColourmap, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.cbCmapReverse = wx.CheckBox(pan, -1, 'Invert')
        self.cbCmapReverse.SetValue(cmapReversed)

        hsizer.Add(self.cbCmapReverse, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        bdsizer = wx.BoxSizer()
        bdsizer.Add(bsizer, 1, wx.EXPAND|wx.ALL, 0)

        pan.SetSizer(bdsizer)
        bdsizer.Fit(pan)

        
        self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        self.cColourmap.Bind(wx.EVT_CHOICE, self.OnCMapChange)
        self.cbCmapReverse.Bind(wx.EVT_CHECKBOX, self.OnCMapChange)
        
        
        #CLim
        pan = wx.Panel(item, -1)

        box = wx.StaticBox(pan, -1, 'CLim:')
        bsizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Min: '), 0, wx.LEFT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tCLimMin = wx.TextCtrl(pan, -1, '%3.2f' % self.glCanvas.clim[0], size=(40,-1))
        hsizer.Add(self.tCLimMin, 0,wx.RIGHT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 5)

        hsizer.Add(wx.StaticText(pan, -1, '  Max: '), 0, wx.LEFT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tCLimMax = wx.TextCtrl(pan, -1, '%3.2f' % self.glCanvas.clim[1], size=(40,-1))
        hsizer.Add(self.tCLimMax, 0, wx.RIGHT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 5)
 
        bsizer.Add(hsizer, 0, wx.ALL, 0)

        self.hlCLim = histLimits.HistLimitPanel(pan, -1, self.glCanvas.c, self.glCanvas.clim[0], self.glCanvas.clim[1], size=(150, 100))
        bsizer.Add(self.hlCLim, 0, wx.ALL|wx.EXPAND, 5)

        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.tPercentileCLim = wx.TextCtrl(pan, -1, '.95', size=(40,-1))
        hsizer.Add(self.tPercentileCLim, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        
        bPercentile = wx.Button(pan, -1, 'Set Percentile')
        hsizer.Add(bPercentile, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
 
        bsizer.Add(hsizer, 0, wx.ALL, 0)

        bdsizer = wx.BoxSizer()
        bdsizer.Add(bsizer, 1, wx.EXPAND|wx.ALL, 0)

        pan.SetSizer(bdsizer)
        bdsizer.Fit(pan)

        #self.hlCLim.Refresh()

        self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        self.tCLimMin.Bind(wx.EVT_TEXT, self.OnCLimChange)
        self.tCLimMax.Bind(wx.EVT_TEXT, self.OnCLimChange)

        self.hlCLim.Bind(histLimits.EVT_LIMIT_CHANGE, self.OnCLimHistChange)

        bPercentile.Bind(wx.EVT_BUTTON, self.OnPercentileCLim)
        
        #self._pnl.AddFoldPanelSeparator(item)


        #LUT
        cbLUTDraw = wx.CheckBox(item, -1, 'Show LUT')
        cbLUTDraw.SetValue(self.glCanvas.LUTDraw)
        self._pnl.AddFoldPanelWindow(item, cbLUTDraw, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

        cbLUTDraw.Bind(wx.EVT_CHECKBOX, self.OnLUTDrawCB)

        
        #Scale Bar
        pan = wx.Panel(item, -1)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Scale Bar: '), 0, wx.LEFT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 5)


        chInd = self.scaleBarLengths.values().index(self.glCanvas.scaleBarLength)
        
        chScaleBar = wx.Choice(pan, -1, choices = self.scaleBarLengths.keys())
        chScaleBar.SetSelection(chInd)
        hsizer.Add(chScaleBar, 0,wx.RIGHT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 5)

        pan.SetSizer(hsizer)
        hsizer.Fit(pan)
        
        self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

        chScaleBar.Bind(wx.EVT_CHOICE, self.OnChangeScaleBar)

        
    def OnCMapChange(self, event):
        cmapname = pylab.cm.cmapnames[self.cColourmap.GetSelection()]
        if self.cbCmapReverse.GetValue():
            cmapname += '_r'

        self.glCanvas.setCMap(pylab.cm.__dict__[cmapname])
        self.OnGLViewChanged()

    def OnLUTDrawCB(self, event):
        self.glCanvas.LUTDraw = event.IsChecked()
        self.glCanvas.Refresh()
        
    def OnChangeScaleBar(self, event):
        self.glCanvas.scaleBarLength = self.scaleBarLengths[event.GetString()]
        self.glCanvas.Refresh()

    def OnCLimChange(self, event):
        if self._pc_clim_change: #avoid setting CLim twice
            self._pc_clim_change = False #clear flag
        else:
            cmin = float(self.tCLimMin.GetValue())
            cmax = float(self.tCLimMax.GetValue())

            self.glCanvas.setCLim((cmin, cmax))

    def OnCLimHistChange(self, event):
        self.glCanvas.setCLim((event.lower, event.upper))
        self._pc_clim_change = True
        self.tCLimMax.SetValue('%3.2f' % self.glCanvas.clim[1])
        self._pc_clim_change = True
        self.tCLimMin.SetValue('%3.2f' % self.glCanvas.clim[0])

    def OnPercentileCLim(self, event):
        pc = float(self.tPercentileCLim.GetValue())

        self.glCanvas.setPercentileCLim(pc)

        self._pc_clim_change = True
        self.tCLimMax.SetValue('%3.2f' % self.glCanvas.clim[1])
        self._pc_clim_change = True
        self.tCLimMin.SetValue('%3.2f' % self.glCanvas.clim[0])

        self.hlCLim.SetValue(self.glCanvas.clim)

        
        
            
    def GenFilterPanel(self):
        item = self._pnl.AddFoldPanel("Filter", collapsed=True,
                                      foldIcons=self.Images)

        self.lFiltKeys = wx.ListCtrl(item, -1, style=wx.LC_REPORT|wx.LC_SINGLE_SEL|wx.SUNKEN_BORDER, size=(-1, 200))

        self._pnl.AddFoldPanelWindow(item, self.lFiltKeys, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

        self.lFiltKeys.InsertColumn(0, 'Key')
        self.lFiltKeys.InsertColumn(1, 'Min')
        self.lFiltKeys.InsertColumn(2, 'Max')

        for key, value in self.filterKeys.items():
            ind = self.lFiltKeys.InsertStringItem(sys.maxint, key)
            self.lFiltKeys.SetStringItem(ind,1, '%3.2f' % value[0])
            self.lFiltKeys.SetStringItem(ind,2, '%3.2f' % value[1])

        self.lFiltKeys.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        self.lFiltKeys.SetColumnWidth(1, wx.LIST_AUTOSIZE)
        self.lFiltKeys.SetColumnWidth(2, wx.LIST_AUTOSIZE)

        # only do this part the first time so the events are only bound once
        if not hasattr(self, "ID_FILT_ADD"):
            self.ID_FILT_ADD = wx.NewId()
            self.ID_FILT_DELETE = wx.NewId()
            self.ID_FILT_EDIT = wx.NewId()
           
            self.Bind(wx.EVT_MENU, self.OnFilterAdd, id=self.ID_FILT_ADD)
            self.Bind(wx.EVT_MENU, self.OnFilterDelete, id=self.ID_FILT_DELETE)
            self.Bind(wx.EVT_MENU, self.OnFilterEdit, id=self.ID_FILT_EDIT)

        # for wxMSW
        self.lFiltKeys.Bind(wx.EVT_COMMAND_RIGHT_CLICK, self.OnFilterListRightClick)

        # for wxGTK
        self.lFiltKeys.Bind(wx.EVT_RIGHT_UP, self.OnFilterListRightClick)

        self.lFiltKeys.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnFilterItemSelected)
        self.lFiltKeys.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.OnFilterItemDeselected)
        self.lFiltKeys.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnFilterEdit)

        self.stFilterNumPoints = wx.StaticText(item, -1, '')

        if not self.filter == None:
            self.stFilterNumPoints.SetLabel('%d of %d events' % (len(self.filter['x']), len(self.selectedDataSource['x'])))

        self._pnl.AddFoldPanelWindow(item, self.stFilterNumPoints, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

        self.bClipToSelection = wx.Button(item, -1, 'Clip to selection')
        self._pnl.AddFoldPanelWindow(item, self.bClipToSelection, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

        self.bClipToSelection.Bind(wx.EVT_BUTTON, self.OnFilterClipToSelection)
        
    def OnFilterListRightClick(self, event):

        x = event.GetX()
        y = event.GetY()

        item, flags = self.lFiltKeys.HitTest((x, y))

 
        # make a menu
        menu = wx.Menu()
        # add some items
        menu.Append(self.ID_FILT_ADD, "Add")

        if item != wx.NOT_FOUND and flags & wx.LIST_HITTEST_ONITEM:
            self.currentFilterItem = item
            self.lFiltKeys.Select(item)
        
            menu.Append(self.ID_FILT_DELETE, "Delete")
            menu.Append(self.ID_FILT_EDIT, "Edit")

        # Popup the menu.  If an item is selected then its handler
        # will be called before PopupMenu returns.
        self.PopupMenu(menu)
        menu.Destroy()

    def OnFilterItemSelected(self, event):
        self.currentFilterItem = event.m_itemIndex

        event.Skip()

    def OnFilterItemDeselected(self, event):
        self.currentFilterItem = None

        event.Skip()

    def OnFilterClipToSelection(self, event):
        x0, y0 = self.glCanvas.selectionStart
        x1, y1 = self.glCanvas.selectionFinish

        if not 'x' in self.filterKeys.keys():
            indx = self.lFiltKeys.InsertStringItem(sys.maxint, 'x')
        else:
            indx = [self.lFiltKeys.GetItemText(i) for i in range(self.lFiltKeys.GetItemCount())].index('x')

        if not 'y' in self.filterKeys.keys():
            indy = self.lFiltKeys.InsertStringItem(sys.maxint, 'y')
        else:
            indy = [self.lFiltKeys.GetItemText(i) for i in range(self.lFiltKeys.GetItemCount())].index('y')


        self.filterKeys['x'] = (min(x0, x1), max(x0, x1))
        self.filterKeys['y'] = (min(y0, y1), max(y0,y1))

        self.lFiltKeys.SetStringItem(indx,1, '%3.2f' % min(x0, x1))
        self.lFiltKeys.SetStringItem(indx,2, '%3.2f' % max(x0, x1))

        self.lFiltKeys.SetStringItem(indy,1, '%3.2f' % min(y0, y1))
        self.lFiltKeys.SetStringItem(indy,2, '%3.2f' % max(y0, y1))

        self.RegenFilter()

    def OnFilterAdd(self, event):
        #key = self.lFiltKeys.GetItem(self.currentFilterItem).GetText()

        possibleKeys = []
        if not self.selectedDataSource == None:
            possibleKeys = self.selectedDataSource.keys()

        dlg = editFilterDialog.FilterEditDialog(self, mode='new', possibleKeys=possibleKeys)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            minVal = float(dlg.tMin.GetValue())
            maxVal = float(dlg.tMax.GetValue())

            key = dlg.cbKey.GetValue().encode()

            if key == "":
                return

            self.filterKeys[key] = (minVal, maxVal)

            ind = self.lFiltKeys.InsertStringItem(sys.maxint, key)
            self.lFiltKeys.SetStringItem(ind,1, '%3.2f' % minVal)
            self.lFiltKeys.SetStringItem(ind,2, '%3.2f' % maxVal)

        dlg.Destroy()

        self.RegenFilter()

    def OnFilterDelete(self, event):
        it = self.lFiltKeys.GetItem(self.currentFilterItem)
        self.lFiltKeys.DeleteItem(self.currentFilterItem)
        self.filterKeys.pop(it.GetText())

        self.RegenFilter()
        
    def OnFilterEdit(self, event):
        key = self.lFiltKeys.GetItem(self.currentFilterItem).GetText()

        #dlg = editFilterDialog.FilterEditDialog(self, mode='edit', possibleKeys=[], key=key, minVal=self.filterKeys[key][0], maxVal=self.filterKeys[key][1])
        dlg = histLimits.HistLimitDialog(self, self.selectedDataSource[key], self.filterKeys[key][0], self.filterKeys[key][1], title=key)
        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            #minVal = float(dlg.tMin.GetValue())
            #maxVal = float(dlg.tMax.GetValue())
            minVal, maxVal = dlg.GetLimits()

            self.filterKeys[key] = (minVal, maxVal)

            self.lFiltKeys.SetStringItem(self.currentFilterItem,1, '%3.2f' % minVal)
            self.lFiltKeys.SetStringItem(self.currentFilterItem,2, '%3.2f' % maxVal)

        dlg.Destroy()
        self.RegenFilter()

    
    def GenQuadTreePanel(self):
        item = self._pnl.AddFoldPanel("QuadTree", collapsed=False,
                                      foldIcons=self.Images)

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

        
        self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        self.tQTLeafSize.Bind(wx.EVT_TEXT, self.OnQTLeafChange)

    

    def OnQTLeafChange(self, event):
        leafSize = int(self.tQTLeafSize.GetValue())
        if not leafSize >= 1:
            raise 'QuadTree leaves must be able to contain at least 1 item'

        pointQT.QT_MAXRECORDS = leafSize
        self.stQTSNR.SetLabel('Effective SNR = %3.2f' % pylab.sqrt(pointQT.QT_MAXRECORDS/2.0))

        self.Quads = None
        self.RefreshView()


    def GenBlobPanel(self):
        item = self._pnl.AddFoldPanel("Objects", collapsed=False,
                                      foldIcons=self.Images)

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

        self.cBlobColour = wx.Choice(pan, -1, choices=['Index'])
        self.cBlobColour.SetSelection(0)
        self.cBlobColour.Bind(wx.EVT_CHOICE, self.OnSetBlobColour)

        hsizer.Add(self.cBlobColour, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        pan.SetSizer(bsizer)
        bsizer.Fit(pan)


        self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        self.bApplyThreshold.Bind(wx.EVT_BUTTON, self.OnObjApplyThreshold)
        self.bObjMeasure.Bind(wx.EVT_BUTTON, self.OnObjMeasure)

    def OnSetBlobColour(self, event):
        bcolour = self.cBlobColour.GetStringSelection()

        if bcolour == 'Index':
            c = self.objCInd.astype('f')
        else:
            c = self.objectMeasures[bcolour][self.objCInd.astype('i')]

        self.glCanvas.c = c
        self.glCanvas.setColour()
        self.OnGLViewChanged()
        
        self.hlCLim.SetData(self.glCanvas.c, self.glCanvas.clim[0], self.glCanvas.clim[1])

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
            self.rav = recArrayView.recArrayPanel(self.notebook, self.objectMeasures)
            self.notebook.AddPage(self.rav, 'Measurements')
        else:
            self.rav.grid.SetData(self.objectMeasures)

        self.cBlobColour.Clear()
        self.cBlobColour.Append('Index')

        for n in self.objectMeasures.dtype.names:
            self.cBlobColour.Append(n)

        self.RefreshView()


    def GenPointsPanel(self, title='Points'):
        item = self._pnl.AddFoldPanel(title, collapsed=False,
                                      foldIcons=self.Images)

        pan = wx.Panel(item, -1)
        bsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Size [nm]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPointSize = wx.TextCtrl(pan, -1, '%3.2f' % self.glCanvas.pointSize)
        hsizer.Add(self.tPointSize, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        
        colData = ['<None>']

        if not self.filter == None:
            colData += self.filter.keys()

        colData += self.GeneratedMeasures.keys()

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Colour:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.chPointColour = wx.Choice(pan, -1, choices=colData, size=(100, -1))
        hsizer.Add(self.chPointColour, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)
        
        pan.SetSizer(bsizer)
        bsizer.Fit(pan)

        
        self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        self.tPointSize.Bind(wx.EVT_TEXT, self.OnPointSizeChange)
        self.chPointColour.Bind(wx.EVT_CHOICE, self.OnChangePointColour)

    def OnPointSizeChange(self, event):
        self.glCanvas.pointSize = float(self.tPointSize.GetValue())
        self.glCanvas.Refresh()

    def OnChangePointColour(self, event):
        colData = event.GetString()

        if colData == '<None>':
            self.pointColour = None
        elif not self.mapping == None:
            if colData in self.mapping.keys():
                self.pointColour = self.mapping[colData]
            elif colData in self.GeneratedMeasures.keys():
                self.pointColour = self.GeneratedMeasures[colData]
            else:
                self.pointColour = None
        
        self.RefreshView()

    def GenDriftPanel(self):
        item = self._pnl.AddFoldPanel("Drift Correction", collapsed=True,
                                      foldIcons=self.Images)

        pan = wx.Panel(item, -1)
        bsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, "x' = "), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tXExpr = wx.TextCtrl(pan, -1, self.driftExprX, size=(130, -1))
        hsizer.Add(self.tXExpr, 2,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, "y' = "), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tYExpr = wx.TextCtrl(pan, -1, self.driftExprY, size=(130,-1))
        hsizer.Add(self.tYExpr, 2,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        pan.SetSizer(bsizer)
        bsizer.Fit(pan)


        self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        self.tXExpr.Bind(wx.EVT_TEXT, self.OnDriftExprChange)
        self.tYExpr.Bind(wx.EVT_TEXT, self.OnDriftExprChange)


        self.lDriftParams = editList.EditListCtrl(item, -1, style=wx.LC_REPORT|wx.LC_SINGLE_SEL|wx.SUNKEN_BORDER, size=(-1, 100))

        self._pnl.AddFoldPanelWindow(item, self.lDriftParams, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

        self.lDriftParams.InsertColumn(0, 'Parameter')
        self.lDriftParams.InsertColumn(1, 'Value')

        self.lDriftParams.makeColumnEditable(1)

        #self.RefreshDriftParameters()

        self.OnDriftExprChange()

        self.lDriftParams.Bind(wx.EVT_LIST_END_LABEL_EDIT, self.OnDriftParameterChange)

        pan = wx.Panel(item, -1)
        #bsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        bFit = wx.Button(pan, -1, 'Fit', size=(30,-1))
        hsizer.Add(bFit, 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        bPlot = wx.Button(pan, -1, 'Plt', size=(30,-1))
        hsizer.Add(bPlot, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bApply = wx.Button(pan, -1, 'Apply', size=(50,-1))
        hsizer.Add(bApply, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bRevert = wx.Button(pan, -1, 'Revert', size=(50,-1))
        hsizer.Add(bRevert, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        pan.SetSizer(hsizer)
        hsizer.Fit(pan)


        self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        bFit.Bind(wx.EVT_BUTTON, self.OnDriftFit)
        bApply.Bind(wx.EVT_BUTTON, self.OnDriftApply)
        bRevert.Bind(wx.EVT_BUTTON, self.OnDriftRevert)
        bPlot.Bind(wx.EVT_BUTTON, self.OnDriftPlot)

    def OnDriftFit(self, event):
        self.driftCorrParams = intelliFit.doFitT(self.driftCorrFcn, self.driftCorrParams, self.filter, self.optimiseFcn)
        self.RefreshDriftParameters()

    def OnDriftApply(self, event):
        self.mapping.setMapping('x', self.driftCorrFcn[2])
        self.mapping.setMapping('y', self.driftCorrFcn[3])
        self.mapping.__dict__.update(self.driftCorrParams)

        self.Triangles = None
        self.GeneratedMeasures = {}
        self.Quads = None
        
        self.RefreshView()

    def OnDriftRevert(self, event):
        self.mapping.mappings.pop('x')
        self.mapping.mappings.pop('y')

        self.Triangles = None
        self.GeneratedMeasures = {}
        self.Quads = None

        self.RefreshView()

    def OnDriftPlot(self, event):
        intelliFit.plotDriftResultT(self.driftCorrFcn, self.driftCorrParams, self.filter)

    def OnDriftExprChange(self, event=None):
        self.driftExprX = self.tXExpr.GetValue()
        self.driftExprY = self.tYExpr.GetValue()
        if self.filter == None:
            filtKeys = []
        else:
            filtKeys = self.filter.keys()

        self.driftCorrFcn = intelliFit.genFcnCodeT(self.driftExprX,self.driftExprY, filtKeys)

        #self.driftCorrParams = {}
        for p in self.driftCorrFcn[0]:
            if not p in self.driftCorrParams.keys():
                self.driftCorrParams[p] = 0

        self.RefreshDriftParameters()

    def OnDriftParameterChange(self, event=None):
        parameterNames = self.driftCorrFcn[0]

        pn = parameterNames[event.m_itemIndex]

        self.driftCorrParams[pn] = float(event.m_item.GetText())

    def RefreshDriftParameters(self):
        parameterNames = self.driftCorrFcn[0]

        self.lDriftParams.DeleteAllItems()

        for pn in parameterNames:
            ind = self.lDriftParams.InsertStringItem(sys.maxint, pn)
            self.lDriftParams.SetStringItem(ind,1, '%1.3g' % self.driftCorrParams[pn])

        self.lDriftParams.SetColumnWidth(0, 80)
        self.lDriftParams.SetColumnWidth(1, 80)

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

        ID_VIEW_FIT = wx.NewId()
        
        ID_GEN_JIT_TRI = wx.NewId()
        ID_GEN_QUADS = wx.NewId()

        ID_GEN_GAUSS = wx.NewId()
        ID_GEN_HIST = wx.NewId()

        ID_GEN_CURRENT = wx.NewId()

        ID_TOGGLE_SETTINGS = wx.NewId()

        ID_ABOUT = wx.ID_ABOUT
        
        
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
        except:
            self.view_menu.Append(ID_VIEW_POINTS, '&Points')
            self.view_menu.Append(ID_VIEW_TRIANGS, '&Triangles')
            self.view_menu.Append(ID_VIEW_QUADS, '&Quad Tree')
            self.view_menu.Append(ID_VIEW_VORONOI, '&Voronoi')
            self.view_menu.Append(ID_VIEW_INTERP_TRIANGS, '&Interpolated Triangles')
            self.view_menu.Append(ID_VIEW_BLOBS, '&Blobs')

        self.view_menu.Check(ID_VIEW_POINTS, True)
        #self.view_menu.Enable(ID_VIEW_QUADS, False)

        self.view_menu.AppendSeparator()
        self.view_menu.Append(ID_VIEW_FIT, '&Fit')

        self.view_menu.AppendSeparator()
        self.view_menu.AppendCheckItem(ID_TOGGLE_SETTINGS, "Show Settings")
        self.view_menu.Check(ID_TOGGLE_SETTINGS, True)

        gen_menu = wx.Menu()
        gen_menu.Append(ID_GEN_CURRENT, "&Current")
        
        gen_menu.AppendSeparator()
        gen_menu.Append(ID_GEN_GAUSS, "&Gaussian")
        gen_menu.Append(ID_GEN_HIST, "&Histogram")

        gen_menu.AppendSeparator()
        gen_menu.Append(ID_GEN_JIT_TRI, "&Triangulation")
        gen_menu.Append(ID_GEN_QUADS, "&QuadTree")
        

        help_menu = wx.Menu()
        help_menu.Append(ID_ABOUT, "&About")

        menu_bar = wx.MenuBar()

        menu_bar.Append(file_menu, "&File")
        menu_bar.Append(self.view_menu, "&View")
        menu_bar.Append(gen_menu, "&Generate Image")
       
        

            
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

        self.Bind(wx.EVT_MENU, self.SetFit, id=ID_VIEW_FIT)

        self.Bind(wx.EVT_MENU, self.OnGenCurrent, id=ID_GEN_CURRENT)
        self.Bind(wx.EVT_MENU, self.OnGenTriangles, id=ID_GEN_JIT_TRI)
        self.Bind(wx.EVT_MENU, self.OnGenGaussian, id=ID_GEN_GAUSS)
        self.Bind(wx.EVT_MENU, self.OnGenHistogram, id=ID_GEN_HIST)
        self.Bind(wx.EVT_MENU, self.OnGenQuadTree, id=ID_GEN_QUADS)

        return menu_bar

    def OnViewPoints(self,event):
        self.viewMode = 'points'
        #self.glCanvas.cmap = pylab.cm.hsv
        self.RefreshView()
        self.CreateFoldPanel()
        self.OnPercentileCLim(None)

    def OnViewBlobs(self,event):
        self.viewMode = 'blobs'
        self.RefreshView()
        self.CreateFoldPanel()
        #self.OnPercentileCLim(None)

    def OnViewTriangles(self,event):
        self.viewMode = 'triangles'
        self.RefreshView()
        self.CreateFoldPanel()
        self.OnPercentileCLim(None)

    def OnViewQuads(self,event):
        self.viewMode = 'quads'
        self.RefreshView()
        self.CreateFoldPanel()
        self.OnPercentileCLim(None)

    def OnViewVoronoi(self,event):
        self.viewMode = 'voronoi'
        self.RefreshView()
        self.CreateFoldPanel()
        self.OnPercentileCLim(None)

    def OnViewInterpTriangles(self,event):
        self.viewMode = 'interp_triangles'
        self.RefreshView()
        self.CreateFoldPanel()
        self.OnPercentileCLim(None)

    def OnGenCurrent(self, event):
        dlg = genImageDialog.GenImageDialog(self, mode='current')

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            pixelSize = dlg.getPixelSize()
            
            bCurr = wx.BusyCursor()

            oldcmap = self.glCanvas.cmap 
            self.glCanvas.setCMap(pylab.cm.gray)

            
            im = self.glCanvas.getIm(pixelSize)

            imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)

            img = GeneratedImage(im,imb, pixelSize )
            imf = imageView.ImageViewFrame(self,img, self.glCanvas)
            self.generatedImages.append(imf)
            imf.Show()

            self.glCanvas.setCMap(oldcmap)
            self.RefreshView()

        dlg.Destroy()

    def genNeighbourDists(self):
        bCurr = wx.BusyCursor()

        if self.Triangles == None:
                statTri = statusLog.StatusLogger("Generating Triangulation ...")
                self.Triangles = delaunay.Triangulation(self.mapping['x'], self.mapping['y'])

        statNeigh = statusLog.StatusLogger("Calculating mean neighbour distances ...")
        self.GeneratedMeasures['neighbourDistances'] = pylab.array(visHelpers.calcNeighbourDists(self.Triangles))
        

    def OnGenTriangles(self, event): 
        jitVars = ['1.0']

        if not 'neighbourDistances' in self.GeneratedMeasures.keys():
            self.genNeighbourDists()

        genMeas = self.GeneratedMeasures.keys()

        jitVars += genMeas
        jitVars += self.mapping.keys()
        
        dlg = genImageDialog.GenImageDialog(self, mode='triangles', jitterVariables = jitVars, jitterVarDefault=genMeas.index('neighbourDistances')+1)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            bCurr = wx.BusyCursor()
            pixelSize = dlg.getPixelSize()
            jitParamName = dlg.getJitterVariable()
            jitScale = dlg.getJitterScale()
            
            if jitParamName == '1.0':
                jitVals = 1.0
            elif jitParamName in self.mapping.keys():
                jitVals = self.mapping[jitParamName]
            elif jitParamName in self.GeneratedMeasures.keys():
                jitVals = self.GeneratedMeasures[jitParamName]
        
            #print jitScale
            #print jitVals
            jitVals = jitScale*jitVals

            oldcmap = self.glCanvas.cmap 
            self.glCanvas.setCMap(pylab.cm.gray)

            imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)

            status = statusLog.StatusLogger('Generating Triangulated Image ...')
            im = self.glCanvas.genJitTim(dlg.getNumSamples(),self.mapping['x'],self.mapping['y'], jitVals, dlg.getMCProbability(),pixelSize)
            

            img = GeneratedImage(im,imb, pixelSize )
            imf = imageView.ImageViewFrame(self,img, self.glCanvas)
            self.generatedImages.append(imf)
            imf.Show()

            self.glCanvas.setCMap(oldcmap)
            self.RefreshView()

        dlg.Destroy()

    def OnGenGaussian(self, event):
        bCurr = wx.BusyCursor()
        jitVars = ['1.0']

        jitVars += self.mapping.keys()
        jitVars += self.GeneratedMeasures.keys()
        
        dlg = genImageDialog.GenImageDialog(self, mode='gaussian', jitterVariables = jitVars, jitterVarDefault=self.mapping.keys().index('error_x')+1)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            pixelSize = dlg.getPixelSize()
            jitParamName = dlg.getJitterVariable()
            jitScale = dlg.getJitterScale()
            
            if jitParamName == '1.0':
                jitVals = 1.0
            elif jitParamName in self.mapping.keys():
                jitVals = self.mapping[jitParamName]
            elif jitParamName in self.GeneratedMeasures.keys():
                jitVals = self.GeneratedMeasures[jitParamName]
        
            #print jitScale
            #print jitVals
            jitVals = jitScale*jitVals

            status = statusLog.StatusLogger('Generating Gaussian Image ...')

            imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)

            im = visHelpers.rendGauss(self.mapping['x'],self.mapping['y'], jitVals, imb, pixelSize)

            img = GeneratedImage(im,imb, pixelSize )
            imf = imageView.ImageViewFrame(self,img, self.glCanvas)
            self.generatedImages.append(imf)
            imf.Show()
            

        dlg.Destroy()

    def OnGenHistogram(self, event): 
        bCurr = wx.BusyCursor()
        dlg = genImageDialog.GenImageDialog(self, mode='histogram')

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            pixelSize = dlg.getPixelSize()

            status = statusLog.StatusLogger('Generating Histogram Image ...')
            
            imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)

            im = visHelpers.rendHist(self.mapping['x'],self.mapping['y'], imb, pixelSize)
            
            img = GeneratedImage(im,imb, pixelSize )
            imf = imageView.ImageViewFrame(self,img, self.glCanvas)
            self.generatedImages.append(imf)
            imf.Show()
            

        dlg.Destroy()

    def OnGenQuadTree(self, event):
        bCurr = wx.BusyCursor() 
        dlg = genImageDialog.GenImageDialog(self, mode='quadtree')

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            pixelSize = dlg.getPixelSize()

            status = statusLog.StatusLogger('Generating QuadTree Image ...')
            
            imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)

            if not pylab.mod(pylab.log2(pixelSize/self.QTGoalPixelSize), 1) == 0:#recalculate QuadTree to get right pixel size
                self.QTGoalPixelSize = pixelSize
                self.Quads = None
            
            if self.Quads == None:
                self.GenQuads()

            qtWidth = self.Quads.x1 - self.Quads.x0

            qtWidthPixels = pylab.ceil(qtWidth/pixelSize)

            im = pylab.zeros((qtWidthPixels, qtWidthPixels))

            QTrend.rendQTa(im, self.Quads)

            im = im[(imb.x0/pixelSize):(imb.x1/pixelSize),(imb.y0/pixelSize):(imb.y1/pixelSize)]
            
            img = GeneratedImage(im,imb, pixelSize )
            imf = imageView.ImageViewFrame(self,img, self.glCanvas)
            self.generatedImages.append(imf)
            imf.Show()

        dlg.Destroy()

    def OnSaveMeasurements(self, event):
        fdialog = wx.FileDialog(None, 'Save measurements ...',
            wildcard='Numpy array|*.npy', style=wx.SAVE|wx.HIDE_READONLY)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            outFilename = fdialog.GetPath().encode()
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

                #once we get around to storing the some metadata with the results
                if 'MetaData' in self.selectedDataSource.h5f.root:
                    self.mdh = MetaDataHandler.HDFMDHandler(self.selectedDataSource.h5f)

                    if 'Camera.ROIWidth' in self.mdh.getEntryNames():
                        x0 = 0
                        y0 = 0

                        x1 = self.mdh.getEntry('Camera.ROIWidth')*1e3*self.mdh.getEntry('voxelsize.x')
                        y1 = self.mdh.getEntry('Camera.ROIHeight')*1e3*self.mdh.getEntry('voxelsize.y')

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

                if 'Events' in self.selectedDataSource.h5f.root:
                    self.events = self.selectedDataSource.h5f.root.Events[:]

                    self.elv = eventLogViewer.eventLogPanel(self.notebook, self.events, self.mdh, [0, self.selectedDataSource['tIndex'].max()]);
                    self.notebook.AddPage(self.elv, 'Events')

                    evKeyNames = set()
                    for e in self.events:
                        evKeyNames.add(e['EventName'])

                    if 'ProtocolFocus' in evKeyNames:
                        self.zm = piecewiseMapping.GeneratePMFromEventList(self.events, self.mdh.getEntry('Camera.CycleTime'), self.mdh.getEntry('StartTime'), self.mdh.getEntry('Protocol.PiezoStartPos'))
                        self.elv.SetCharts([('Focus [um]', self.zm, 'ProtocolFocus'),])
                        
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
            dlg = importTextDialog.ImportTextDialog(self)

            ret = dlg.ShowModal()

            if not ret == wx.ID_OK:
                return #we cancelled

            #try:
            print dlg.GetFieldNames()
            ds = inpFilt.textfileSource(filename, dlg.GetFieldNames())
            self.selectedDataSource = ds
            self.dataSources.append(ds)

            self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)

        self.SetTitle('PYME Visualise - ' + filename)
        #for k in self.filterKeys.keys():

        #if we've done a 3d fit
        print self.selectedDataSource.keys()
        if not 'sig' in self.selectedDataSource.keys():
            self.filterKeys.pop('sig')

        print self.filterKeys
        self.RegenFilter()
        self.CreateFoldPanel()
        self.SetFit()

    def OnOpenChannel(self, event):
        filename = wx.FileSelector("Choose a file to open", nameUtils.genResultDirectoryPath(), default_extension='h5r', wildcard='PYME Results Files (*.h5r)|*.h5r|Tab Formatted Text (*.txt)|*.txt')

        #print filename
        if not filename == '':
            self.OpenChannel(filename)

    def OpenChannel(self, filename):
        self.filter = None
        self.mapping = None
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

                #once we get around to storing the some metadata with the results
#                if 'MetaData' in self.selectedDataSource.h5f.root:
#                    self.mdh = MetaDataHandler.HDFMDHandler(self.selectedDataSource.h5f)
#
#                    if 'Camera.ROIWidth' in self.mdh.getEntryNames():
#                        x0 = 0
#                        y0 = 0
#
#                        x1 = self.mdh.getEntry('Camera.ROIWidth')*1e3*self.mdh.getEntry('voxelsize.x')
#                        y1 = self.mdh.getEntry('Camera.ROIHeight')*1e3*self.mdh.getEntry('voxelsize.y')
#
#                        self.imageBounds = ImageBounds(x0, y0, x1, y1)
#                    else:
#                        self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)
#
#                else:
#                    self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)
#
#                if not self.elv == None: #remove previous event viewer
#                    i = 0
#                    found = False
#                    while not found and i < self.notebook.GetPageCount():
#                        if self.notebook.GetPage(i) == self.elv:
#                            self.notebook.DeletePage(i)
#                            found = True
#                        else:
#                            i += 1
#
#                if 'Events' in self.selectedDataSource.h5f.root:
#                    self.events = self.selectedDataSource.h5f.root.Events[:]
#
#                    self.elv = eventLogViewer.eventLogPanel(self.notebook, self.events, self.mdh, [0, self.selectedDataSource['tIndex'].max()]);
#                    self.notebook.AddPage(self.elv, 'Events')
#
#                    evKeyNames = set()
#                    for e in self.events:
#                        evKeyNames.add(e['EventName'])
#
#                    if 'ProtocolFocus' in evKeyNames:
#                        self.zm = piecewiseMapping.GeneratePMFromEventList(self.events, self.mdh.getEntry('Camera.CycleTime'), self.mdh.getEntry('StartTime'), self.mdh.getEntry('Protocol.PiezoStartPos'))
#                        self.elv.SetCharts([('Focus [um]', self.zm, 'ProtocolFocus'),])

        else: #assume it's a text file
            dlg = importTextDialog.ImportTextDialog(self)

            ret = dlg.ShowModal()

            if not ret == wx.ID_OK:
                return #we cancelled

            #try:
            print dlg.GetFieldNames()
            ds = inpFilt.textfileSource(filename, dlg.GetFieldNames())
            self.selectedDataSource = ds
            self.dataSources.append(ds)

            #self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)

        #self.SetTitle('PYME Visualise - ' + filename)
        #for k in self.filterKeys.keys():

        #if we've done a 3d fit
#        print self.selectedDataSource.keys()
#        if 'fitResults_z0' in self.selectedDataSource.keys():
#            self.filterKeys.pop('sig')

        print self.filterKeys
        self.RegenFilter()
        self.CreateFoldPanel()
        self.SetFit()

    def OnOpenRaw(self, event):
        filename = wx.FileSelector("Choose a file to open", nameUtils.genResultDirectoryPath(), default_extension='h5', wildcard='PYME Spool Files (*.h5)|*.h5|Khoros Data Format (*.kdf)|*.kdf')

        #print filename
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
            h5f = tables.openFile(filename)

            md = MetaData.genMetaDataFromHDF(h5f)

            #im = h5f.root.ImageData[min(md.EstimatedLaserOnFrameNo+10,(h5f.root.ImageData.shape[0]-1)) , :,:].squeeze().astype('f')
            im = h5f.root.ImageData
            #im = im - min(md.CCD.ADOffset, im.min())

            #h5f.close()

            self.filesToClose.append(h5f)
            
            pixelSize = md.voxelsize.x*1e3

            imb = ImageBounds(0,0,pixelSize*im.shape[1],pixelSize*im.shape[2])

            img = GeneratedImage(im,imb, pixelSize )
            imf = imageView.ImageViewFrame(self,img, self.glCanvas, title=filename,zp=min(md.EstimatedLaserOnFrameNo+10,(h5f.root.ImageData.shape[0]-1)))
            self.generatedImages.append(imf)
            imf.Show()
        else:
            raise 'Unrecognised Data Format'




    def RegenFilter(self):
        if not self.selectedDataSource == None:
            self.filter = inpFilt.resultsFilter(self.selectedDataSource, **self.filterKeys)
            self.mapping = inpFilt.mappingFilter(self.filter)

        self.stFilterNumPoints.SetLabel('%d of %d events' % (len(self.filter['x']), len(self.selectedDataSource['x'])))

        self.Triangles = None
        self.objects = None

        self.GeneratedMeasures = {}
        if 'zm' in dir(self):
            self.GeneratedMeasures['focusPos'] = self.zm(self.mapping['tIndex'].astype('f'))
        self.Quads = None

        self.RefreshView()


    def RefreshView(self):
        if self.mapping == None:
            return #get out of here

        if len(self.mapping['x']) == 0:
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
            self.glCanvas.setPoints(self.mapping['x'], self.mapping['y'], self.pointColour)
        elif self.viewMode == 'triangles':
            if self.Triangles == None:
                status = statusLog.StatusLogger("Generating Triangulation ...")
                self.Triangles = delaunay.Triangulation(self.mapping['x'], self.mapping['y'])
                
            self.glCanvas.setTriang(self.Triangles)

        elif self.viewMode == 'voronoi':
            if self.Triangles == None:
                status = statusLog.StatusLogger("Generating Triangulation ...")
                self.Triangles = delaunay.Triangulation(self.mapping['x'], self.mapping['y'])
                

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
                self.Triangles = delaunay.Triangulation(self.mapping['x'], self.mapping['y'])

            self.glCanvas.setIntTriang(self.Triangles, self.pointColour)

        elif self.viewMode == 'blobs':
            if self.objects == None:
                #check to see that we don't have too many points
                if len(self.mapping['x']) > 10e3:
                    goAhead = wx.MessageBox('You have %d events in the selected ROI;\nThis could take a LONG time ...' % len(self.mapping['x']), 'Continue with blob detection', wx.YES_NO|wx.ICON_EXCLAMATION)

                    if not goAhead == wx.YES:
                        return
                if self.blobJitter == 0:
                    T = delny.Triangulation(pylab.array([self.mapping['x'] + 0.1*pylab.randn(len(self.mapping['x'])), self.mapping['y']+ 0.1*pylab.randn(len(self.mapping['x']))]).T)
                    self.objects = gen3DTriangs.segment(T, self.objThreshold, self.objMinSize)
                else:
                    if not 'neighbourDistances' in self.GeneratedMeasures.keys():
                        self.genNeighbourDists()
                    x_ = pylab.hstack([self.mapping['x'] + 0.5*self.GeneratedMeasures['neighbourDistances']*pylab.randn(len(self.mapping['x'])) for i in range(self.blobJitter)])
                    y_ = pylab.hstack([self.mapping['y'] + 0.5*self.GeneratedMeasures['neighbourDistances']*pylab.randn(len(self.mapping['x'])) for i in range(self.blobJitter)])

                    T = delny.Triangulation(pylab.array([x_, y_]).T)
                    self.objects = gen3DTriangs.segment(T, self.objThreshold, self.objMinSize)

#                if 'bObjMeasure' in dir(self):
#                    self.bObjMeasure.Enable(True)

            self.glCanvas.setBlobs(self.objects, self.objThreshold)
            self.objCInd = self.glCanvas.c

        self.hlCLim.SetData(self.glCanvas.c, self.glCanvas.clim[0], self.glCanvas.clim[1])


    def GenQuads(self):
        di = max(self.imageBounds.x1 - self.imageBounds.x0, self.imageBounds.y1 - self.imageBounds.y0)

        np = di/self.QTGoalPixelSize

        di = self.QTGoalPixelSize*2**pylab.ceil(pylab.log2(np))

        
        self.Quads = pointQT.qtRoot(self.imageBounds.x0, self.imageBounds.x0+di, self.imageBounds.y0, self.imageBounds.y0 + di)

        for xi, yi in zip(self.mapping['x'],self.mapping['y']):
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
    #from optparse import OptionParser

    #parser = OptionParser()
    #parser.add_option("-i", "--init-file", dest="initFile", help="Read initialisation from file [defaults to init.py]", metavar="FILE")
        
    #(options, args) = parser.parse_args()

    
    
    application = VisGuiApp(filename, 0)
    application.MainLoop()

if __name__ == '__main__':

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

