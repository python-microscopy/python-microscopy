import os.path
import wx
import wx.lib.foldpanelbar as fpb
import gl_render
import sys
import inpFilt
import editFilterDialog
import pylab
from PYME.FileUtils import nameUtils
import os
from scikits import delaunay

from PYME.Analysis.QuadTree import pointQT, QTrend
import Image

import genImageDialog
import importTextDialog
import visHelpers
import imageView
import histLimits
import time

import tables
from PYME.Analysis import MetaData

import threading

import statusLog


# ----------------------------------------------------------------------------
# Visualisation of analysed localisation microscopy data
#
# David Baddeley 2009
#
# Some of the code in this file borrowed from the wxPython examples
# ----------------------------------------------------------------------------

class ImageBounds:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @classmethod
    def estimateFromSource(cls, ds):
        return cls(ds['x'].min(),ds['y'].min(),ds['x'].max(), ds['y'].max() )

    def width(self):
        return self.x1 - self.x0

    def height(self):
        return self.y1 - self.y0

class dummy:
    pass

class GeneratedImage:
    def __init__(self, img, imgBounds, pixelSize):
        self.img = img
        self.imgBounds = imgBounds

        self.pixelSize = pixelSize

    def save(self, filename):
        #save using PIL - because we're using float pretty much only tif will work
        im = Image.fromarray(self.img.astype('f'), 'F')
        
        im.tag = dummy()
        #set up resolution data - unfortunately in cm as TIFF standard only supports cm and inches
        res_ = int(1e-2/(self.pixelSize*1e-9))
        im.tag.tagdata={296:(3,), 282:(res_,1), 283:(res_,1)}

        im.save(filename)
        

def GetCollapsedIconData():
    return \
'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\
\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\
\x00\x01\x8eIDAT8\x8d\xa5\x93-n\xe4@\x10\x85?g\x03\n6lh)\xc4\xd2\x12\xc3\x81\
\xd6\xa2I\x90\x154\xb9\x81\x8f1G\xc8\x11\x16\x86\xcd\xa0\x99F\xb3A\x91\xa1\
\xc9J&\x96L"5lX\xcc\x0bl\xf7v\xb2\x7fZ\xa5\x98\xebU\xbdz\xf5\\\x9deW\x9f\xf8\
H\\\xbfO|{y\x9dT\x15P\x04\x01\x01UPUD\x84\xdb/7YZ\x9f\xa5\n\xce\x97aRU\x8a\
\xdc`\xacA\x00\x04P\xf0!0\xf6\x81\xa0\xf0p\xff9\xfb\x85\xe0|\x19&T)K\x8b\x18\
\xf9\xa3\xe4\xbe\xf3\x8c^#\xc9\xd5\n\xa8*\xc5?\x9a\x01\x8a\xd2b\r\x1cN\xc3\
\x14\t\xce\x97a\xb2F0Ks\xd58\xaa\xc6\xc5\xa6\xf7\xdfya\xe7\xbdR\x13M2\xf9\
\xf9qKQ\x1fi\xf6-\x00~T\xfac\x1dq#\x82,\xe5q\x05\x91D\xba@\xefj\xba1\xf0\xdc\
zzW\xcff&\xb8,\x89\xa8@Q\xd6\xaaf\xdfRm,\xee\xb1BDxr#\xae\xf5|\xddo\xd6\xe2H\
\x18\x15\x84\xa0q@]\xe54\x8d\xa3\xedf\x05M\xe3\xd8Uy\xc4\x15\x8d\xf5\xd7\x8b\
~\x82\x0fh\x0e"\xb0\xad,\xee\xb8c\xbb\x18\xe7\x8e;6\xa5\x89\x04\xde\xff\x1c\
\x16\xef\xe0p\xfa>\x19\x11\xca\x8d\x8d\xe0\x93\x1b\x01\xd8m\xf3(;x\xa5\xef=\
\xb7w\xf3\x1d$\x7f\xc1\xe0\xbd\xa7\xeb\xa0(,"Kc\x12\xc1+\xfd\xe8\tI\xee\xed)\
\xbf\xbcN\xc1{D\x04k\x05#\x12\xfd\xf2a\xde[\x81\x87\xbb\xdf\x9cr\x1a\x87\xd3\
0)\xba>\x83\xd5\xb97o\xe0\xaf\x04\xff\x13?\x00\xd2\xfb\xa9`z\xac\x80w\x00\
\x00\x00\x00IEND\xaeB`\x82' 

def GetCollapsedIconBitmap():
    return wx.BitmapFromImage(GetCollapsedIconImage())

def GetCollapsedIconImage():
    import cStringIO
    stream = cStringIO.StringIO(GetCollapsedIconData())
    return wx.ImageFromStream(stream)

#----------------------------------------------------------------------
def GetExpandedIconData():
    return \
'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\
\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\
\x00\x01\x9fIDAT8\x8d\x95\x93\xa1\x8e\xdc0\x14EO\xb2\xc4\xd0\xd2\x12\xb7(mI\
\xa4%V\xd1lQT4[4-\x9a\xfe\xc1\xc2|\xc6\xc2~BY\x83:A3E\xd3\xa0*\xa4\xd2\x90H!\
\x95\x0c\r\r\x1fK\x81g\xb2\x99\x84\xb4\x0fY\xd6\xbb\xc7\xf7>=\'Iz\xc3\xbcv\
\xfbn\xb8\x9c\x15 \xe7\xf3\xc7\x0fw\xc9\xbc7\x99\x03\x0e\xfbn0\x99F+\x85R\
\x80RH\x10\x82\x08\xde\x05\x1ef\x90+\xc0\xe1\xd8\ryn\xd0Z-\\A\xb4\xd2\xf7\
\x9e\xfbwoF\xc8\x088\x1c\xbbae\xb3\xe8y&\x9a\xdf\xf5\xbd\xe7\xfem\x84\xa4\
\x97\xccYf\x16\x8d\xdb\xb2a]\xfeX\x18\xc9s\xc3\xe1\x18\xe7\x94\x12cb\xcc\xb5\
\xfa\xb1l8\xf5\x01\xe7\x84\xc7\xb2Y@\xb2\xcc0\x02\xb4\x9a\x88%\xbe\xdc\xb4\
\x9e\xb6Zs\xaa74\xadg[6\x88<\xb7]\xc6\x14\x1dL\x86\xe6\x83\xa0\x81\xba\xda\
\x10\x02x/\xd4\xd5\x06\r\x840!\x9c\x1fM\x92\xf4\x86\x9f\xbf\xfe\x0c\xd6\x9ae\
\xd6u\x8d \xf4\xf5\x165\x9b\x8f\x04\xe1\xc5\xcb\xdb$\x05\x90\xa97@\x04lQas\
\xcd*7\x14\xdb\x9aY\xcb\xb8\\\xe9E\x10|\xbc\xf2^\xb0E\x85\xc95_\x9f\n\xaa/\
\x05\x10\x81\xce\xc9\xa8\xf6><G\xd8\xed\xbbA)X\xd9\x0c\x01\x9a\xc6Q\x14\xd9h\
[\x04\xda\xd6c\xadFkE\xf0\xc2\xab\xd7\xb7\xc9\x08\x00\xf8\xf6\xbd\x1b\x8cQ\
\xd8|\xb9\x0f\xd3\x9a\x8a\xc7\x08\x00\x9f?\xdd%\xde\x07\xda\x93\xc3{\x19C\
\x8a\x9c\x03\x0b8\x17\xe8\x9d\xbf\x02.>\x13\xc0n\xff{PJ\xc5\xfdP\x11""<\xbc\
\xff\x87\xdf\xf8\xbf\xf5\x17FF\xaf\x8f\x8b\xd3\xe6K\x00\x00\x00\x00IEND\xaeB\
`\x82' 

def GetExpandedIconBitmap():
    return wx.BitmapFromImage(GetExpandedIconImage())

def GetExpandedIconImage():
    import cStringIO
    stream = cStringIO.StringIO(GetExpandedIconData())
    return wx.ImageFromStream(stream)



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
        self.glCanvas = gl_render.LMGLCanvas(self)
        self.glCanvas.cmap = pylab.cm.hot

        self.ID_WINDOW_TOP = 100
        self.ID_WINDOW_LEFT1 = 101
        self.ID_WINDOW_RIGHT1 = 102
        self.ID_WINDOW_BOTTOM = 103
    
        self._leftWindow1.Bind(wx.EVT_SASH_DRAGGED_RANGE, self.OnFoldPanelBarDrag,
                               id=100, id2=103)
        self.Bind(wx.EVT_SIZE, self.OnSize)

        
        self._pc_clim_change = False

        self.filesToClose = []
        self.generatedImages = []

        self.dataSources = []
        self.selectedDataSource = None
        self.filterKeys = {'error_x': (0,30), 'A':(5,200), 'sig' : (150/2.35, 350/2.35)}

        self.filter = None
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
        

    def OnSize(self, event):

        wx.LayoutAlgorithm().LayoutWindow(self, self.glCanvas)
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
        wx.LayoutAlgorithm().LayoutWindow(self, self.glCanvas)
        self.glCanvas.Refresh()

        event.Skip()
        

    def OnFoldPanelBarDrag(self, event):

        if event.GetDragStatus() == wx.SASH_STATUS_OUT_OF_RANGE:
            return

        if event.GetId() == self.ID_WINDOW_LEFT1:
            self._leftWindow1.SetDefaultSize(wx.Size(event.GetDragRect().width, 1000))


        # Leaves bits of itself behind sometimes
        wx.LayoutAlgorithm().LayoutWindow(self, self.glCanvas)
        self.glCanvas.Refresh()

        event.Skip()
        

    def CreateFoldPanel(self):

        # delete earlier panel
        self._leftWindow1.DestroyChildren()

        # recreate the foldpanelbar

        self._pnl = fpb.FoldPanelBar(self._leftWindow1, -1, wx.DefaultPosition,
                                     wx.Size(-1,-1), fpb.FPB_DEFAULT_STYLE,0)

        self.Images = wx.ImageList(16,16)
        self.Images.Add(GetExpandedIconBitmap())
        self.Images.Add(GetCollapsedIconBitmap())
            
        self.GenDataSourcePanel()
        self.GenFilterPanel()

        self.GenDisplayPanel()
        
        if self.viewMode == 'quads':
            self.GenQuadTreePanel()

        if self.viewMode == 'points':
            self.GenPointsPanel()

       

        #item = self._pnl.AddFoldPanel("Filters", False, foldIcons=self.Images)
        #item = self._pnl.AddFoldPanel("Visualisation", False, foldIcons=self.Images)
        wx.LayoutAlgorithm().LayoutWindow(self, self.glCanvas)
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


    def GenPointsPanel(self):
        item = self._pnl.AddFoldPanel("Points", collapsed=False,
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
        elif not self.filter == None:
            if colData in self.filter.keys():
                self.pointColour = self.filter[colData]
        elif colData in self.GeneratedMeasures.keys():
            self.pointColour = self.GeneratedMeasures[colData]
        else:
            self.pointColour = None
        
        self.RefreshView()

    def CreateMenuBar(self):

        # Make a menubar
        file_menu = wx.Menu()

        ID_OPEN = wx.ID_OPEN
        ID_QUIT = wx.ID_EXIT

        ID_OPEN_RAW = wx.NewId()

        ID_VIEW_POINTS = wx.NewId()
        ID_VIEW_TRIANGS = wx.NewId()
        ID_VIEW_QUADS = wx.NewId()

        ID_VIEW_VORONOI = wx.NewId()

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
        
        file_menu.AppendSeparator()
        
        file_menu.Append(ID_QUIT, "&Exit")

        self.view_menu = wx.Menu()

        try: #stop us bombing on Mac
            self.view_menu.AppendRadioItem(ID_VIEW_POINTS, '&Points')
            self.view_menu.AppendRadioItem(ID_VIEW_TRIANGS, '&Triangles')
            self.view_menu.AppendRadioItem(ID_VIEW_QUADS, '&Quad Tree')
            self.view_menu.AppendRadioItem(ID_VIEW_VORONOI, '&Voronoi')
        except:
            self.view_menu.Append(ID_VIEW_POINTS, '&Points')
            self.view_menu.Append(ID_VIEW_TRIANGS, '&Triangles')
            self.view_menu.Append(ID_VIEW_QUADS, '&Quad Tree')
            self.view_menu.Append(ID_VIEW_VORONOI, '&Voronoi')

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
        self.Bind(wx.EVT_MENU, self.OnOpenRaw, id=ID_OPEN_RAW)

        self.Bind(wx.EVT_MENU, self.OnViewPoints, id=ID_VIEW_POINTS)
        self.Bind(wx.EVT_MENU, self.OnViewTriangles, id=ID_VIEW_TRIANGS)
        self.Bind(wx.EVT_MENU, self.OnViewQuads, id=ID_VIEW_QUADS)
        self.Bind(wx.EVT_MENU, self.OnViewVoronoi, id=ID_VIEW_VORONOI)

        self.Bind(wx.EVT_MENU, self.SetFit, id=ID_VIEW_FIT)

        self.Bind(wx.EVT_MENU, self.OnGenCurrent, id=ID_GEN_CURRENT)
        self.Bind(wx.EVT_MENU, self.OnGenTriangles, id=ID_GEN_JIT_TRI)
        self.Bind(wx.EVT_MENU, self.OnGenGaussian, id=ID_GEN_GAUSS)
        self.Bind(wx.EVT_MENU, self.OnGenHistogram, id=ID_GEN_HIST)
        self.Bind(wx.EVT_MENU, self.OnGenQuadTree, id=ID_GEN_QUADS)

        return menu_bar

    def OnViewPoints(self,event):
        self.viewMode = 'points'
        self.RefreshView()
        self.CreateFoldPanel()
        self.OnPercentileCLim(None)

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
                self.Triangles = delaunay.Triangulation(self.filter['x'], self.filter['y'])

        statNeigh = statusLog.StatusLogger("Calculating mean neighbour distances ...")
        self.GeneratedMeasures['neighbourDistances'] = pylab.array(visHelpers.calcNeighbourDists(self.Triangles))
        

    def OnGenTriangles(self, event): 
        jitVars = ['1.0']

        if not 'neighbourDistances' in self.GeneratedMeasures.keys():
            self.genNeighbourDists()

        genMeas = self.GeneratedMeasures.keys()

        jitVars += genMeas
        jitVars += self.filter.keys()
        
        dlg = genImageDialog.GenImageDialog(self, mode='triangles', jitterVariables = jitVars, jitterVarDefault=genMeas.index('neighbourDistances')+1)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            bCurr = wx.BusyCursor()
            pixelSize = dlg.getPixelSize()
            jitParamName = dlg.getJitterVariable()
            jitScale = dlg.getJitterScale()
            
            if jitParamName == '1.0':
                jitVals = 1.0
            elif jitParamName in self.filter.keys():
                jitVals = self.filter[jitParamName]
            elif jitParamName in self.GeneratedMeasures.keys():
                jitVals = self.GeneratedMeasures[jitParamName]
        
            #print jitScale
            #print jitVals
            jitVals = jitScale*jitVals

            oldcmap = self.glCanvas.cmap 
            self.glCanvas.setCMap(pylab.cm.gray)

            imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)

            status = statusLog.StatusLogger('Generating Triangulated Image ...')
            im = self.glCanvas.genJitTim(dlg.getNumSamples(),self.filter['x'],self.filter['y'], jitVals, dlg.getMCProbability(),pixelSize)
            

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

        jitVars += self.filter.keys()        
        jitVars += self.GeneratedMeasures.keys()
        
        dlg = genImageDialog.GenImageDialog(self, mode='gaussian', jitterVariables = jitVars, jitterVarDefault=self.filter.keys().index('error_x')+1)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            pixelSize = dlg.getPixelSize()
            jitParamName = dlg.getJitterVariable()
            jitScale = dlg.getJitterScale()
            
            if jitParamName == '1.0':
                jitVals = 1.0
            elif jitParamName in self.filter.keys():
                jitVals = self.filter[jitParamName]
            elif jitParamName in self.GeneratedMeasures.keys():
                jitVals = self.GeneratedMeasures[jitParamName]
        
            #print jitScale
            #print jitVals
            jitVals = jitScale*jitVals

            status = statusLog.StatusLogger('Generating Gaussian Image ...')

            imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)

            im = visHelpers.rendGauss(self.filter['x'],self.filter['y'], jitVals, imb, pixelSize)

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

            im = visHelpers.rendHist(self.filter['x'],self.filter['y'], imb, pixelSize)
            
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


    def OnOpenFile(self, event):
        filename = wx.FileSelector("Choose a file to open", nameUtils.genResultDirectoryPath(), default_extension='h5r', wildcard='PYME Results Files (*.h5r)|*.h5r|Tab Formatted Text (*.txt)|*.txt')

        #print filename
        if not filename == '':
            self.OpenFile(filename)

    def OpenFile(self, filename):
        while len(self.filesToClose) > 0:
            self.filesToClose.pop().close()
        
        self.dataSources = []
        self.filter = None
        print os.path.splitext(filename)[1]
        if os.path.splitext(filename)[1] == '.h5r':
                self.selectedDataSource = inpFilt.h5rSource(filename)
                self.dataSources.append(self.selectedDataSource)

                self.filesToClose.append(self.selectedDataSource.h5f)
                
                if 'DriftResults' in self.selectedDataSource.h5f.root:
                    self.dataSources.append(inpFilt.h5rDSource(self.selectedDataSource.h5f))

                #once we get around to storing the some metadata with the results
                if 'MetaData' in self.selectedDataSource.h5f.root: 
                    mdh = MetaDataHandler(self.selectedDataSource.h5f)
                    x0 = mdh.getEntry('ImageShape.x0_nm')
                    y0 = mdh.getEntry('ImageShape.y0_nm')
                    x1 = mdh.getEntry('ImageShape.x1_nm')
                    y1 = mdh.getEntry('ImageShape.y1_nm')

                    self.imageBounds = ImageBounds(x0, y0, x1, y1)

                else:
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
            imf = imageView.ImageViewFrame(self,img, self.glCanvas, title=filename)
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

        self.Triangles = None
        self.GeneratedMeasures = {}
        self.Quads = None

        self.RefreshView()


    def RefreshView(self):
        if self.filter == None:
            return #get out of here

        if len(self.filter['x']) == 0:
            wx.MessageBox('No data points - try adjusting the filter', "len(filter['x']) ==0")
            return

        if self.glCanvas.init == 0: #glcanvas is not initialised
            return

        bCurr = wx.BusyCursor()

        if self.viewMode == 'points':
            self.glCanvas.setPoints(self.filter['x'], self.filter['y'], self.pointColour)
        elif self.viewMode == 'triangles':
            if self.Triangles == None:
                status = statusLog.StatusLogger("Generating Triangulation ...")
                self.Triangles = delaunay.Triangulation(self.filter['x'], self.filter['y'])
                
            self.glCanvas.setTriang(self.Triangles)

        elif self.viewMode == 'voronoi':
            if self.Triangles == None:
                status = statusLog.StatusLogger("Generating Triangulation ...")
                self.Triangles = delaunay.Triangulation(self.filter['x'], self.filter['y'])
                

            status = statusLog.StatusLogger("Generating Voronoi Diagram ... ")
            self.glCanvas.setVoronoi(self.Triangles)
            

        elif self.viewMode == 'quads':
            if self.Quads == None:
                status = statusLog.StatusLogger("Generating QuadTree ...")
                self.GenQuads()
                

            self.glCanvas.setQuads(self.Quads)

        self.hlCLim.SetData(self.glCanvas.c, self.glCanvas.clim[0], self.glCanvas.clim[1])


    def GenQuads(self):
        di = max(self.imageBounds.x1 - self.imageBounds.x0, self.imageBounds.y1 - self.imageBounds.y0)

        np = di/self.QTGoalPixelSize

        di = self.QTGoalPixelSize*2**pylab.ceil(pylab.log2(np))

        
        self.Quads = pointQT.qtRoot(self.imageBounds.x0, self.imageBounds.x0+di, self.imageBounds.y0, self.imageBounds.y0 + di)

        for xi, yi in zip(self.filter['x'],self.filter['y']):
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

