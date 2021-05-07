#!/usr/bin/python
##################
# vis3D.py
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
#import numpy
from __future__ import print_function
import wx
import wx.lib.mixins.listctrl  as  listmix
import os
#import pylab
from PYME.IO import MetaDataHandler
from PYME.IO import image
from PYME.DSView import View3D, ViewIm3D
from PYME.IO import dataWrap

import wx.lib.agw.aui as aui

from PYME.Analysis.composite import common_prefix, make_composite

class ShiftPanel(wx.Panel):
    def __init__(self, dsviewer):
        wx.Panel.__init__(self, dsviewer)
        self.dsviewer = dsviewer
        self.image = dsviewer.image
        
        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.xctls = []
        self.yctls = []
        self.zctls = [] 
        
        self.rbs = []
        
        chanList = self.image.data.dataList
        
        #iterate over colours
        for i in range(self.image.data.shape[3]):
            shifts = chanList[i].shifts
            #vsizer.Add(wx.StaticText(self, -1, 'Chan %d:' % i), 0, wx.EXPAND|wx.ALL, 2)
            rb = wx.RadioButton(self, -1, 'Chan %d:' % i)
            if i == 0:
                rb.SetValue(True)
            vsizer.Add(rb, 0, wx.EXPAND|wx.ALL, 2)
            self.rbs.append(rb)
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            #hsizer.Add(wx.StaticText(self, -1, 'dx'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
            iD = wx.NewId()
            x1 = wx.TextCtrl(self, iD, str(int(shifts[0])), size=[30,-1])
            self.xctls.append(x1)
            hsizer.Add(x1, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
            
            #hsizer.Add(wx.StaticText(self, -1, 'dy'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
            iD = wx.NewId()
            y1 = wx.TextCtrl(self, iD, str(int(shifts[1])), size=[30,-1])
            self.yctls.append(y1)
            hsizer.Add(y1, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
            
            #hsizer.Add(wx.StaticText(self, -1, 'dz'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
            iD = wx.NewId()
            y1 = wx.TextCtrl(self, iD, str(int(shifts[2])), size=[30,-1])
            self.zctls.append(y1)
            hsizer.Add(y1, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
            
            vsizer.Add(hsizer, 0, wx.EXPAND, 0)
            
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        bUpdate = wx.Button(self, -1, 'Update', size=[40, -1])
        bUpdate.Bind(wx.EVT_BUTTON, self.OnApply)
        hsizer.Add(bUpdate, 1, wx.ALL, 2)
        bZero = wx.Button(self, -1, 'Zero', size=[40, -1])
        bZero.Bind(wx.EVT_BUTTON, self.OnZero) 
        hsizer.Add(bZero, 0, wx.ALL, 2)
        
        vsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)
        
        
        
        bEstimate = wx.Button(self, -1, 'Estimate (slow)')
        vsizer.Add(bEstimate, 0, wx.ALL|wx.EXPAND, 2)
        
        bEstimate.Bind(wx.EVT_BUTTON, self.OnCorrelate)
        
        self.SetSizerAndFit(vsizer)
        
    def OnApply(self, event):
        chanList = self.image.data.dataList
            
        for i in range(self.image.data.shape[3]):
            dx = float(self.xctls[i].GetValue())
            dy = float(self.yctls[i].GetValue())
            dz = float(self.zctls[i].GetValue())
            
            chanList[i].SetShifts([dx,dy,dz])
            
        self.dsviewer.do.OnChange()
        
    def OnCorrelate(self, event):
        from numpy.fft import fftn, ifftn, fftshift, ifftshift
        # from pylab import fftshift, ifftshift
        import numpy as np
        #ch0 = self.image.data[:,:,:,0]
        chanList = self.image.data.dataList
        
        for i in range(len(self.rbs)):
            if self.rbs[i].GetValue():
                ch0 = self.image.data[:,:,:,i]
                
        ch0 = np.maximum(ch0 - ch0.mean(), 0)
        
        F0 = fftn(ch0)
        
        for i in range(self.image.data.shape[3]):
            if not self.rbs[i].GetValue():
                ch0 = self.image.data[:,:,:,i]
                ch0 = np.maximum(ch0 - ch0.mean(), 0)
                Fi = ifftn(ch0)
                
                corr = abs(fftshift(ifftn(F0*Fi)))
                
                corr -= corr.min()
                
                corr = np.maximum(corr - corr.max()*.75, 0)
                
                xi, yi, zi = np.where(corr)
                
                corr_s = corr[corr>0]
                corr_s/= corr_s.sum()
                
                dxi =  ((xi*corr_s).sum() - corr.shape[0]/2.)*chanList[i].voxelsize[0]
                dyi =  ((yi*corr_s).sum() - corr.shape[1]/2.)*chanList[i].voxelsize[1]
                dzi =  ((zi*corr_s).sum() - corr.shape[2]/2.)*chanList[i].voxelsize[2]
                
                self.xctls[i].SetValue(str(int(dxi)))
                self.yctls[i].SetValue(str(int(dyi)))
                self.zctls[i].SetValue(str(int(dzi)))
            
        self.OnApply(None)
        
    def OnZero(self, event):
        for i in range(self.image.data.shape[3]):
            self.xctls[i].SetValue('0')
            self.yctls[i].SetValue('0')
            self.zctls[i].SetValue('0')
            
        self.OnApply(None)
        #View3D(corr)
        
class ShiftmapSelectionDialog(wx.Dialog):
    def __init__(self, parent, image):
        wx.Dialog.__init__(self, parent, title='Deconvolution')
        nChans = image.data.shape[3]

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        
        sizer2 = wx.GridSizer(nChans, 2, 5, 5)
        
        sizer2.Add(wx.StaticText(self, -1, 'Channel:'))
        sizer2.Add(wx.StaticText(self, -1, 'Shiftmap:'))
        
        self.filectrls = {}
        
        for chan in range(nChans):
            sizer2.Add(wx.StaticText(self, -1, image.names[chan]), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            fp = wx.FilePickerCtrl(self, -1, wildcard='*.sm', style=wx.FLP_OPEN|wx.FLP_FILE_MUST_EXIST|wx.FLP_USE_TEXTCTRL)
            sizer2.Add(fp, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            self.filectrls[chan] = fp
            
        sizer1.Add(sizer2, 1, wx.ALL|wx.EXPAND, 5)
            
        btSizer = wx.StdDialogButtonSizer()

        self.bOK = wx.Button(self, wx.ID_OK)
        self.bOK.SetDefault()

        btSizer.AddButton(self.bOK)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)
        
    def GetChanFilename(self, chan):
        return self.filectrls[chan].GetPath()
        
def _getImage(name):
    if name in image.openImages.keys():
        img = image.openImages[name]
        chan = 0
    else:
        n_, chan_, cn_ = name.split('$')
        img = image.openImages[n_]
        chan = int(chan_)
        
    return img, chan
        
class ImageList(wx.ListCtrl):
    def __init__(self, parent, id=-1, imageNames=[], size=(300,200)):
        wx.ListCtrl.__init__(self, parent, id, 
            style=wx.LC_REPORT|wx.LC_VIRTUAL|wx.LC_VRULES, size=size)
        #listmix.ListCtrlAutoWidthMixin.__init__(self)
            
        self.imageNames = list(imageNames)
        self.shiftmaps = {}
        self.channelNames = {}
        
        self.InsertColumn(0, "Name")
        self.InsertColumn(1, "dx")
        self.InsertColumn(2, "dy")
        
        self.InsertColumn(3, "width")
        self.InsertColumn(4, "height")
        self.InsertColumn(5, "Shiftmap")
        
        self.SetColumnWidth(0, 300)
        self.SetColumnWidth(1, 50)
        self.SetColumnWidth(2, 50)
        self.SetColumnWidth(3, 50)
        self.SetColumnWidth(4, 50)
        self.SetColumnWidth(5, 200)
        #self.InsertColumn(5, "x0")
        #self.InsertColumn(6, "y0")
        
        self.SetItemCount(len(self.imageNames))
        
   
        
    def OnGetItemText(self, item, col):
        name = self.imageNames[item]
        #print item, col, name
        if col == 0:
            #item name
            return self.channelNames[name]
        else:
            img, chan = _getImage(name)
            if col == 1:
                return '%3.2f' % img.mdh['voxelsize.x']
            elif col == 2:
                return '%3.2f' % img.mdh['voxelsize.y']
            elif col == 3:
                return '%3.2f' % (img.imgBounds.width() / 1e3)
            elif col == 4:
                return '%3.2f' % (img.imgBounds.height() / 1e3)
            elif col == 5:
                try:
                    return self.shiftmaps[name]
                except KeyError:
                    return ''
                
                
    def Append(self, name):
        self.imageNames.append(name)
        self.channelNames[name] = os.path.split(name)[-1]
        self.SetItemCount(len(self.imageNames))
        #self.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        
    def SetShiftmap(self, name):
        filename = wx.FileSelector('Choose a shiftmap', default_extension='*.sm', wildcard="Shiftmaps (*.sm)|*.sm")
        #if filename:
        self.shiftmaps[name] = filename
            
        self.Refresh()
        
    def DeleteItem(self, index):
        self.imageNames.pop(index)
        self.SetItemCount(len(self.imageNames))
        
    def ShiftItem(self, index, direction='up'):
        tmp = self.imageNames[index]
        if direction == 'up':
            if index > 0:
                self.imageNames[index] = self.imageNames[index - 1]
                self.imageNames[index -1] = tmp
                self.SetItemState(-1, 0, wx.LIST_STATE_SELECTED)
                self.Select(index-1)
        else:
            if index < (len(self.imageNames) - 1):
                self.imageNames[index] = self.imageNames[index + 1]
                self.imageNames[index +1] = tmp
                self.SetItemState(-1, 0, wx.LIST_STATE_SELECTED)
                self.Select(index+1)
        self.Refresh()
        
    def GetName(self, index):
        return self.imageNames[index]
            
        

class CompositeDialog(wx.Dialog):
    def __init__(self, parent, img):
        wx.Dialog.__init__(self, parent, title='Composite')
        #nChans = image.data.shape[3]
        
        self.imageNames =  image.openImages.keys()
        
        self.dispNames = []
        
        for iN in self.imageNames:
            im = image.openImages[iN]
            if im.data.shape[3] == 1: #single colour
                self.dispNames.append(iN)
            else:
                for i in range(im.data.shape[3]):
                    self.dispNames.append('%s$%d$%s' % (iN, i, im.names[i]))
                    
        #for testing
        #self.dispNames = ['a', 'b', 'c']

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        vsizer.Add(wx.StaticText(self, -1, 'Available Channels:'), 0, wx.ALL, 2)
        self.lAvail = wx.ListBox(self, -1, choices=self.dispNames, size=(300, 200), style=wx.LB_SORT|wx.LB_NEEDED_SB|wx.LB_EXTENDED)
        
        vsizer.Add(self.lAvail, 0, wx.ALL|wx.EXPAND, 2)
        hsizer.Add(vsizer, 0, wx.ALL|wx.EXPAND, 2)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        bmp = wx.ArtProvider.GetBitmap(wx.ART_GO_FORWARD, wx.ART_TOOLBAR, (16,16))
        self.bAddChan = wx.BitmapButton(self, -1, bmp)
        self.bAddChan.Bind(wx.EVT_BUTTON, self.OnAddChan)
        vsizer.Add(self.bAddChan, 0, wx.ALL, 2)
        
        bmp = wx.ArtProvider.GetBitmap(wx.ART_GO_BACK, wx.ART_TOOLBAR, (16,16))
        self.bRemoveChan = wx.BitmapButton(self, -1, bmp)
        self.bRemoveChan.Bind(wx.EVT_BUTTON, self.OnRemoveChan)
        vsizer.Add(self.bRemoveChan, 0, wx.ALL, 2)
        

        hsizer.Add(vsizer, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        vsizer.Add(wx.StaticText(self, -1, 'Selected Channels:'), 0, wx.ALL, 2)
        self.lSelected = ImageList(self, -1, size=(600, 200))
        
        vsizer.Add(self.lSelected, 1, wx.ALL|wx.EXPAND, 2)
        hsizer.Add(vsizer, 1, wx.ALL|wx.EXPAND, 2)
        
        vsizer = wx.BoxSizer(wx.VERTICAL) 
        bmp = wx.ArtProvider.GetBitmap(wx.ART_GO_UP, wx.ART_TOOLBAR, (16,16))
        self.bChanUp = wx.BitmapButton(self, -1, bmp)
        self.bChanUp.Bind(wx.EVT_BUTTON, self.OnChanUp)
        vsizer.Add(self.bChanUp, 0, wx.ALL, 2)
        
        bmp = wx.ArtProvider.GetBitmap(wx.ART_GO_DOWN, wx.ART_TOOLBAR, (16,16))
        self.bChanDown = wx.BitmapButton(self, -1, bmp)
        self.bChanDown.Bind(wx.EVT_BUTTON, self.OnChanDown)
        vsizer.Add(self.bChanDown, 0, wx.ALL, 2)
        
        vsizer.AddSpacer(10)
        
        bmp = wx.ArtProvider.GetBitmap(wx.ART_HELP_SETTINGS, wx.ART_TOOLBAR, (16,16))
        self.bSetShiftmap = wx.BitmapButton(self, -1, bmp)
        self.bSetShiftmap.Bind(wx.EVT_BUTTON, self.OnSetShiftmap)
        self.bSetShiftmap.SetToolTipString("Specify a shiftmap for the selected channel")
        vsizer.Add(self.bSetShiftmap, 0, wx.ALL, 2)
        
        hsizer.Add(vsizer, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        sizer1.Add(hsizer, 1, wx.ALL|wx.EXPAND, 5)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Shape:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)        
        
        self.tShapeX = wx.TextCtrl(self, -1)
        hsizer.Add(self.tShapeX, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tShapeY = wx.TextCtrl(self, -1)
        hsizer.Add(self.tShapeY, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tShapeZ = wx.TextCtrl(self, -1)
        hsizer.Add(self.tShapeZ, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        hsizer.Add(wx.StaticText(self, -1, 'Voxelsize [nm]:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)        
        
        self.tVoxX = wx.TextCtrl(self, -1)
        hsizer.Add(self.tVoxX, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tVoxY = wx.TextCtrl(self, -1)
        hsizer.Add(self.tVoxY, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tVoxZ = wx.TextCtrl(self, -1)
        hsizer.Add(self.tVoxZ, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        hsizer.Add(wx.StaticText(self, -1, 'Origin [nm]:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)        
        
        self.tOriginX = wx.TextCtrl(self, -1)
        hsizer.Add(self.tOriginX, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tOriginY = wx.TextCtrl(self, -1)
        hsizer.Add(self.tOriginY, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tOriginZ = wx.TextCtrl(self, -1)
        hsizer.Add(self.tOriginZ, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        sizer1.Add(hsizer, 0, wx.ALL|wx.EXPAND, 5)
                
        
        self.stOutput = wx.StaticText(self, -1, 'Pixel Size: 0,0\t Shape: 0,0')
        sizer1.Add(self.stOutput, 0, wx.ALL, 2) 
        
        self.cbIgnoreZ = wx.CheckBox(self, -1, 'Ignore Z Origin')
        self.cbIgnoreZ.SetValue(True)
        sizer1.Add(self.cbIgnoreZ, 0, wx.ALL, 2) 
        
        self.cbInterp = wx.CheckBox(self, -1, 'Interpolate')
        self.cbInterp.SetValue(True)
        sizer1.Add(self.cbInterp, 0, wx.ALL, 2) 
            
        btSizer = wx.StdDialogButtonSizer()

        self.bOK = wx.Button(self, wx.ID_OK)
        self.bOK.SetDefault()

        btSizer.AddButton(self.bOK)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)
        
    def SetMaster(self, master):
        self.master = master
        
        ms, ch = _getImage(master)
        
        #self.shape = list(ms.data.shape[:3])
        #self.voxelsize = ms.voxelsize
        #self.origin = ms.origin
        
        self.tShapeX.SetValue('%d' % ms.data.shape[0])
        self.tShapeY.SetValue('%d' % ms.data.shape[1])
        self.tShapeZ.SetValue('%d' % ms.data.shape[2])
        
        self.tVoxX.SetValue('%3.3f' % ms.voxelsize[0])
        self.tVoxY.SetValue('%3.3f' % ms.voxelsize[1])
        self.tVoxZ.SetValue('%3.3f' % ms.voxelsize[2])
        
        self.tOriginX.SetValue('%d' % ms.origin[0])
        self.tOriginY.SetValue('%d' % ms.origin[1])
        self.tOriginZ.SetValue('%d' % ms.origin[2])
        
        sh = self.shape + [self.lSelected.GetItemCount()]
        
        self.stOutput.SetLabel('Output Shape: %s\t Output Voxel Size: %s' % (sh, ms.voxelsize))
        
    @property
    def voxelsize(self):
        return float(self.tVoxX.GetValue()), float(self.tVoxY.GetValue()),float(self.tVoxZ.GetValue())
        
    @property
    def origin(self):
        return float(self.tOriginX.GetValue()), float(self.tOriginY.GetValue()),float(self.tOriginZ.GetValue())

    @property
    def shape(self):
        return [int(self.tShapeX.GetValue()), int(self.tShapeY.GetValue()),int(self.tShapeZ.GetValue())]
        
    def OnAddChan(self, event):
        chans = list(self.lAvail.GetSelections())
        
        for chan in chans:
            self.lSelected.Append(self.lAvail.GetString(chan))
            if self.lSelected.GetItemCount() == 1:
                #first channel is master
                self.SetMaster(self.lAvail.GetString(chan))
        
        chans.sort()
        chans.reverse()
        
        for chan in chans:
            #print chan
            self.lAvail.Delete(chan)
            
    def OnRemoveChan(self, event):
        chan = self.lSelected.GetFirstSelected()
        
        if not chan == wx.NOT_FOUND:
            self.lAvail.Append(self.lSelected.GetName(chan))
            self.lSelected.DeleteItem(chan)
        
    def OnChanUp(self, event):
        chan = self.lSelected.GetFirstSelected()
        
        if not chan == wx.NOT_FOUND:
            self.lSelected.ShiftItem(chan, 'up')
#            if chan > 0: #chan is not already at top
#                text = self.lSelected.GetName(chan)
#                self.lSelected.DeleteItem(chan)
#                self.lSelected.InsertStringItem(chan - 1, text)
#                #self.lSelected.Insert(text, chan - 1)
#                self.lSelected.Select(chan -1)
                
    def OnChanDown(self, event):
        chan = self.lSelected.GetFirstSelected()
        
        if not chan == wx.NOT_FOUND:
            self.lSelected.ShiftItem(chan, 'down')
#            if chan < (self.lSelected.GetItemCount() - 1): #chan is not already at top
#                text = self.lSelected.GetName(chan)
#                self.lSelected.DeleteItem(chan)
#                self.lSelected.InsertStringItem(chan + 1, text)
#                #self.lSelected.Insert(text, chan + 1)
#                #print ch
#                self.lSelected.Select(chan + 1)
                
    def OnSetShiftmap(self, event):
        chan = self.lSelected.GetFirstSelected()
        if not chan == wx.NOT_FOUND:
            text = self.lSelected.GetName(chan)
            self.lSelected.SetShiftmap(text)
                
    def GetSelections(self):
        return [self.lSelected.GetName(ch) for ch in range(self.lSelected.GetItemCount())]
        
    def GetMaster(self):
        return self.master
        
    def GetShiftmap(self, chan):
        try:
            return self.lSelected.shiftmaps[chan]
        except KeyError:
            return ''
        
    def GetIgnoreZ(self):
        return self.cbIgnoreZ.GetValue()
        
    def GetInterp(self):
        return self.cbInterp.GetValue()
        

from ._base import Plugin
class Compositor(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)

        #self.compMenu = wx.Menu()

        dsviewer.mProcessing.AppendSeparator()
        
        dsviewer.AddMenuItem("Processing", "Make Composite", self.OnMakeComposites)
        dsviewer.AddMenuItem("Processing", "Split Channels", self.OnSplitChannels)
        dsviewer.AddMenuItem("Processing", "Apply vector shifts", self.OnApplyShiftmap)
        dsviewer.AddMenuItem("Processing", "Align Channels", self.OnAlignChannels)
        
        dsviewer.mProcessing.AppendSeparator()

        
    def OnMakeComposites(self, event):
        dlg = CompositeDialog(self.dsviewer, self.image)
        
        if dlg.ShowModal() == wx.ID_OK:
            imgs = []
            for im_name in dlg.GetSelections():
                im, chan = _getImage(im_name)
                shiftfield = dlg.GetShiftmap(im_name)
                
                imgs.append((im, chan, shiftfield))
            
            if len(imgs) > 0:
                out = make_composite(imgs,
                                     ignoreZ=dlg.GetIgnoreZ(),
                                     interp=dlg.GetInterp(),
                                     shape=dlg.shape,
                                     origin=dlg.origin,
                                     voxelsize=dlg.voxelsize)
                
                ViewIm3D(out, 'Composite', mode=self.dsviewer.mode,
                       parent=wx.GetTopLevelParent(self.dsviewer), glCanvas=self.dsviewer.glCanvas)
            
        dlg.Destroy()
    

    def OnSplitChannels(self, event):
        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['%d' % d for d in range(self.image.data.shape[3])]

        for i in range(self.image.data.shape[3]):
            mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
            mdh.setEntry('ChannelNames', [names[i]])

            View3D(self.image.data[:,:,:,i], '%s - %s' % (self.image.filename, names[i]), mdh=mdh, parent=wx.GetTopLevelParent(self.dsviewer))
        
        
    def OnApplyShiftmap(self, event):
        """apply a vectorial correction for chromatic shift to an image - this
        is a generic vectorial shift compensation, rather than the secial case 
        correction used with the splitter."""
        from scipy import ndimage
        import numpy as np
        from PYME.DSView import ImageStack, ViewIm3D
        from PYME.IO.MetaDataHandler import get_camera_roi_origin
        
        dlg = ShiftmapSelectionDialog(self.dsviewer, self.image)
        succ = dlg.ShowModal()
        if (succ == wx.ID_OK):
            #self.ds = example.CDataStack(fdialog.GetPath().encode())
            #self.ds =
            ds = []
            shiftFiles = {}
            X, Y, Z = np.mgrid[0:self.image.data.shape[0], 0:self.image.data.shape[1], 0:self.image.data.shape[2]]
            
            vx, vy, vz = self.image.voxelsize_nm
            
            roi_x0, roi_y0 = get_camera_roi_origin(self.image.mdh)
            
            for ch in range(self.image.data.shape[3]):
                sfFilename = dlg.GetChanFilename(ch)
                shiftFiles[ch] = sfFilename
                
                data = self.image.data[:,:,:, ch]
                
                if os.path.exists(sfFilename):
                    spx, spy, dz = np.load(sfFilename)
                    
                    dx = spx.ev(vx*(X+roi_x0), vy*(Y+roi_y0))/vx
                    dy = spy.ev(vx*(X+roi_x0), vy*(Y+roi_y0))/vy
                    dz = dz/vz
                    
                    ds.append(ndimage.map_coordinates(data, [X+dx, Y+dy, Z+dz], mode='nearest'))
                else:
                    ds.append(data)
                
            
            fns = os.path.split(self.image.filename)[1]
            im = ImageStack(ds, titleStub = '%s - corrected' % fns)
            im.mdh.copyEntriesFrom(self.image.mdh)
            im.mdh['Parent'] = self.image.filename
            im.mdh.setEntry('ChromaCorrection.ShiftFilenames', shiftFiles)
            
            if 'fitResults' in dir(self.image):
                im.fitResults = self.image.fitResults
            #im.mdh['Processing.GaussianFilter'] = sigmas
    
            if self.dsviewer.mode == 'visGUI':
                mode = 'visGUI'
            else:
                mode = 'lite'
    
            dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))
            
        dlg.Destroy()
            
            
            
    def OnAlignChannels(self, event):
        #try:
        #    names = self.image.mdh.getEntry('ChannelNames')
        #except:
        #    names = ['%d' % d for d in range(self.image.data.shape[3])]
        from PYME.IO.DataSources import AlignDataSource
        from  PYME.IO import dataWrap
        
        if isinstance(self.image.data, dataWrap.ListWrapper):
            nd = [AlignDataSource.DataSource(ds) for ds in self.image.data.wrapList]
        else:
            nd = [AlignDataSource.DataSource(dataWrap.Wrap(self.image.data[:,:,:,i])) for i in range(self.image.data.shape[3])]
            
        mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
            

        res = View3D(nd, '%s - %s' % (self.image.filename, 'align'), mdh=mdh, parent=wx.GetTopLevelParent(self.dsviewer))
        
        res.panAlign = ShiftPanel(res)

        pinfo1 = aui.AuiPaneInfo().Name("shiftPanel").Right().Caption('Alignment').DestroyOnClose(True).CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        #pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        res._mgr.AddPane(res.panAlign, pinfo1)
        res._mgr.Update()





       
    


def Plug(dsviewer):
    return Compositor(dsviewer)
    #if 'chanList' in dir(dsviewer.image.data) and 'shifts' in dir(dsviewer.image.data.chanList[0]):
        #we have shiftable channels
        


