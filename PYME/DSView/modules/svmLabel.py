#!/usr/bin/python
##################
# profilePlotting.py
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

#from PYME.Acquire.mytimer import mytimer
# import pylab
import matplotlib.cm
from scipy import ndimage
import numpy as np
import wx.lib.agw.aui as aui

try:
    from shapely import speedups
    if speedups.available():
        speedups.enable()
except:
    pass

from PYME.DSView.dsviewer import ViewIm3D, ImageStack

import wx.lib.mixins.listctrl as listmix

class myListCtrl(wx.ListCtrl, listmix.ListCtrlAutoWidthMixin):#, listmix.TextEditMixin):
    def __init__(self, parent, ID, pos=wx.DefaultPosition, size=wx.DefaultSize, style=0):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        listmix.ListCtrlAutoWidthMixin.__init__(self)
        #listmix.TextEditMixin.__init__(self)
        #self.Bind(wx.EVT_LIST_BEGIN_LABEL_EDIT, self.OnBeginLabelEdit)
        self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnLabelActivate)
        

    def OnBeginLabelEdit(self, event):
        if event.m_col == 0:
            event.Veto()
        else:
            event.Skip()
            
    def OnLabelActivate(self, event):
        newLabel = wx.GetTextFromUser("Enter new category name", "Rename")
        if not newLabel == '':
            self.SetStringItem(event.m_itemIndex, 1, newLabel)
        
            
        

class LabelPanel(wx.Panel):
    def __init__(self, parent, labeler, **kwargs):
        kwargs['style'] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, parent, **kwargs)

        #self.parent = parent
        self.labeler = labeler

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.lLabels = myListCtrl(self, -1, style=wx.LC_REPORT)
        self.lLabels.InsertColumn(0, 'Label')
        self.lLabels.InsertColumn(1, 'Structure')
        
        for i in range(10):
            self.lLabels.InsertStringItem(i, '%d'%i)
            self.lLabels.SetStringItem(i, 1, 'Structure %d' % i)
            
        self.lLabels.SetStringItem(0, 1, 'No label')
        
        self.lLabels.SetItemState(1, wx.LIST_STATE_SELECTED, wx.LIST_STATE_SELECTED)
        self.lLabels.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        self.lLabels.SetColumnWidth(1, wx.LIST_AUTOSIZE)
        
        self.lLabels.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnChangeStructure)
        
        vsizer.Add(self.lLabels, 1, wx.ALL|wx.EXPAND, 5)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Line width:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tLineWidth = wx.TextCtrl(self, -1, '1')
        self.tLineWidth.Bind(wx.EVT_TEXT, self.OnChangeLineWidth)
        hsizer.Add(self.tLineWidth, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        vsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 5)

        self.SetSizer(vsizer)
        
    def OnChangeStructure(self, event):
        self.labeler.curLabel = event.m_itemIndex
        
    def OnChangeLineWidth(self, event):
        self.labeler.lineWidth = float(self.tLineWidth.GetValue())

class manualLabel:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        #self.view = dsviewer.view
        self.do = dsviewer.do
        self.image = dsviewer.image
        self.mask = None
        #self.rois = []
        self.curLabel = 1
        self.lineWidth = 1
        
        
        #dsviewer.do.overlays.append(self.DrawOverlays)

        dsviewer.AddMenuItem('Segmentation', 'Create mask', self.OnCreateMask)
        dsviewer.AddMenuItem('Segmentation', "Fill selection\tCtrl-F", self.FillSelection)
        dsviewer.AddMenuItem('Segmentation', "Draw line\tCtrl-L", self.DrawLine)
        dsviewer.AddMenuItem('Segmentation', "Train Classifier", self.OnSVMTrain)
        dsviewer.AddMenuItem('Segmentation', "SVM Segmentation", self.OnSVMSegment)
        
        dsviewer.AddMenuItem('Segmentation', 'Save Classifier', self.OnSaveClassifier)
        dsviewer.AddMenuItem('Segmentation', 'Load Classifier', self.OnLoadClassifier)
        

        
        #accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL,  ord('k'), PLOT_PROFILE )])
        #self.dsviewer.SetAcceleratorTable(accel_tbl)

        self.labelPanel = LabelPanel(dsviewer, self)
        self.labelPanel.SetSize(self.labelPanel.GetBestSize())
        
        pinfo2 = aui.AuiPaneInfo().Name("labelPanel").Right().Caption('Labels').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        dsviewer._mgr.AddPane(self.labelPanel, pinfo2)


    def OnCreateMask(self, event=None):
        #lx, ly, hx, hy = self.do.GetSliceSelection()
        import numpy as np
        #from scipy import ndimage
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D

        #sp = self.image.data.shape[:3]
        #if len(sp)        
        self.mask = np.atleast_3d(np.zeros(self.image.data.shape[:3], 'uint16'))
        #self.vmax = 0
        self.image.labels = self.mask
        
        im = ImageStack(self.mask, titleStub = 'Manual labels')
        im.mdh.copyEntriesFrom(self.image.mdh)

        #im.mdh['Processing.CropROI'] = roi

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        self.labv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))
        
        self.rois = []

        #set scaling to (0,1)
        #for i in range(im.data.shape[3]):
        #    self.dv.do.Gains[i] = 1.0
            
        for i in range(im.data.shape[3]):
            self.labv.do.Gains[i] = .1
            self.labv.do.cmaps[i] = matplotlib.cm.labeled
            
    def OnSaveClassifier(self, event=None):
        filename = wx.FileSelector("Save classifier as:", wildcard="*.pkl", flags=wx.FD_SAVE)
        if not filename == '':
            self.cf.save(filename)
            
    def OnLoadClassifier(self, event=None):
        from PYME.Analysis import svmSegment
        filename = wx.FileSelector("Load Classifier:", wildcard="*.pkl", flags=wx.FD_OPEN)
        if not filename == '':
            self.cf = svmSegment.svmClassifier(filename=filename)


            
    def OnKeyPress(self, event):
        key = event.GetKeyCode()
        
        if key == ord('F'):
            self.FillSelection()
        
        event.Skip()
        
    def FillSelection(self, event=None):
        if self.mask is None:
            print('Create a mask first')
            return
        
#        import Image, ImageDraw
#        import pylab
#        x, y = np.array(self.do.selection_trace).T
#        x0 = x.min()
#        x1 =  x.max()
#        y0 = y.min()
#        y1 = y.max()
#        
#        w = x1 - x0
#        h = y1 - y0
#        
#        img = Image.new('L', (h, w))
#        ImageDraw.Draw(img).polygon(np.hstack([y - y0,x-x0]).T, outline=1, fill=1)
#        pylab.imshow(np.array(img))
#        self.mask[x0:x1, y0:y1, self.do.zp] = np.array(img)
        
            
        #from shapely.geometry import Polygon, Point
        from skimage import draw
        
        #p = Polygon(self.do.selection_trace)
        #x0, y0, x1, y1 = p.bounds

        #self.vmax += 1

        rr, cc = draw.polygon(*np.array(self.do.selection_trace).T)
        self.mask[rr, cc, self.do.zp] = self.curLabel
        
        # for x in range(int(x0), int(x1)+1):
        #     for y in range (int(y0), int(y1)+1):
        #         if Point(x, y).within(p):
        #             self.mask[x,y,self.do.zp] = self.curLabel
        #self.rois.append(p)            
        self.labv.update()
        #try:
        self.labv.do.Gains = [.1]
        self.labv.do.cmaps = [matplotlib.cm.labeled]          
        self.labv.Refresh()
        self.labv.Update()
        
    def DrawLine(self, event=None):
        if self.mask is None:
            print('Create a mask first')
            return
        
#        import Image, ImageDraw
#        import pylab
#        x, y = np.array(self.do.selection_trace).T
#        x0 = x.min()
#        x1 =  x.max()
#        y0 = y.min()
#        y1 = y.max()
#        
#        w = x1 - x0
#        h = y1 - y0
#        
#        img = Image.new('L', (h, w))
#        ImageDraw.Draw(img).polygon(np.hstack([y - y0,x-x0]).T, outline=1, fill=1)
#        pylab.imshow(np.array(img))
#        self.mask[x0:x1, y0:y1, self.do.zp] = np.array(img)
        
            
        from shapely.geometry import LineString, Point
        
        p = LineString(self.do.selection_trace)
        x0, y0, x1, y1 = p.bounds

        #self.vmax += 1        
        
        for x in range(int(x0), int(x1)+1):
            for y in range (int(y0), int(y1)+1):
                if Point(x, y).distance(p) < self.lineWidth:
                    self.mask[x,y,self.do.zp] = self.curLabel
        #self.rois.append(p)            
        self.labv.update()
        #try:
        self.labv.do.Gains = [.1]
        self.labv.do.cmaps = [matplotlib.cm.labeled]          
        self.labv.Refresh()
        self.labv.Update()
        
    def OnSVMTrain(self, event):
        from PYME.Analysis import svmSegment
        
        #from PYME.IO.image import ImageStack
        #from PYME.DSView import ViewIm3D
        
        if not 'cf' in dir(self):
            self.cf = svmSegment.svmClassifier()
        
        self.cf.train(self.image.data[:,:,self.do.zp, 0].squeeze(), self.image.labels[:,:,self.do.zp])
        self.OnSVMSegment(None)
        
    def OnSVMSegment(self, event):
        # import pylab
        #sp = self.image.data.shape[:3]
        #if len(sp)        
        lab2 = self.cf.classify(self.image.data[:,:,self.do.zp, 0].squeeze())#, self.image.labels[:,:,self.do.zp])
        #self.vmax = 0
        #self.image.labels = self.mask
        
        im = ImageStack(lab2, titleStub = 'Segmentation')
        im.mdh.copyEntriesFrom(self.image.mdh)

        #im.mdh['Processing.CropROI'] = roi

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        self.dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))
        
        self.rois = []

        #set scaling to (0,10)
        for i in range(im.data.shape[3]):
            self.dv.do.Gains[i] = .1
            self.dv.do.cmaps[i] = matplotlib.cm.labeled
            
        self.dv.Refresh()
        self.dv.Update()
        
        
    
        #pylab.legend(names)
        
    def DrawOverlays(self, view, dc):
        import wx
        col = wx.TheColourDatabase.FindColour('WHITE')
        dc.SetPen(wx.Pen(col,1))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        
        for roi in self.rois:
            x, y = roi.exterior.xy
            pts = np.vstack(view._PixelToScreenCoordinates(np.array(x), np.array(y))).T
            dc.DrawSpline(pts)
            
        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.NullBrush)


        #dsviewer.paneHooks.append(self.GenProfilePanel)

#    def GenProfilePanel(self, _pnl):
#        item = afp.foldingPane(_pnl, -1, caption="Intensity Profile", pinned = True)
##        item = self._pnl.AddFoldPanel("Intensity Profile", collapsed=False,
##                                      foldIcons=self.Images)
#
#        bPlotProfile = wx.Button(item, -1, 'Plot')
#
#        bPlotProfile.Bind(wx.EVT_BUTTON, self.OnPlotProfile)
#        #self._pnl.AddFoldPanelWindow(item, bPlotProfile, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
#        item.AddNewElement(bPlotProfile)
#        _pnl.AddPane(item)


   


def Plug(dsviewer):
    dsviewer.mLabel = manualLabel(dsviewer)
    dsviewer.mLabel.OnCreateMask(None)
    
