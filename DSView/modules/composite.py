#!/usr/bin/python
##################
# vis3D.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
#import numpy
import wx
import os
#import pylab
from PYME.Acquire import MetaDataHandler
from PYME.DSView import image, View3D
from PYME.DSView import dataWrap

import wx.lib.agw.aui as aui

def common_prefix(strings):
    """ Find the longest string that is a prefix of all the strings.
    """
    if not strings:
        return ''
    prefix = strings[0]
    for s in strings:
        if len(s) < len(prefix):
            prefix = prefix[:len(s)]
        if not prefix:
            return ''
        for i in range(len(prefix)):
            if prefix[i] != s[i]:
                prefix = prefix[:i]
                break
    return prefix

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
        from scipy.fftpack import fftn, ifftn
        from pylab import fftshift, ifftshift
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

class compositor:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image

        self.compMenu = wx.Menu()

        dsviewer.mProcessing.AppendSeparator()
        MAKE_COMPOSITE = wx.NewId()
        dsviewer.mProcessing.Append(MAKE_COMPOSITE, "Make Composite", "", wx.ITEM_NORMAL)

        SPLIT_CHANNELS = wx.NewId()
        dsviewer.mProcessing.Append(SPLIT_CHANNELS, "Split Channels", "", wx.ITEM_NORMAL)
        
        ALIGN_CHANNELS = wx.NewId()
        dsviewer.mProcessing.Append(ALIGN_CHANNELS, "Align Channels", "", wx.ITEM_NORMAL)

        dsviewer.mProcessing.AppendSeparator()

        dsviewer.Bind(wx.EVT_MENU, self.OnMakeComposites, id=MAKE_COMPOSITE)
        dsviewer.Bind(wx.EVT_MENU, self.OnSplitChannels, id=SPLIT_CHANNELS)
        dsviewer.Bind(wx.EVT_MENU, self.OnAlignChannels, id=ALIGN_CHANNELS)



#    def OnMakeComposite(self, event):
#        dlg = wx.SingleChoiceDialog(
#                self.dsviewer, 'choose the image to composite with', 'Make Composite',
#                image.openImages.keys(),
#                wx.CHOICEDLG_STYLE
#                )
#
#        if dlg.ShowModal() == wx.ID_OK:
#            other = image.openImages[dlg.GetStringSelection()]
#
#            ###TODO - Put checks on image size, voxel size etc ...
#
#            try:
#                names = ['%s - %s' % (os.path.split(self.image.filename)[1], cn) for cn in  self.image.mdh.getEntry('ChannelNames')]
#            except:
#                names = ['%s -  %d' % (os.path.split(self.image.filename)[1], d) for d in range(self.image.data.shape[3])]
#
#            try:
#                otherNames = ['%s - %s' % (os.path.split(self.other.filename)[1], cn) for cn in  self.image.mdh.getEntry('ChannelNames')]
#            except:
#                otherNames = ['%s -  %d' % (os.path.split(other.filename)[1], d) for d in range(other.data.shape[3])]
#
#            newNames = names + otherNames
#
#            newData = []
#            if isinstance(self.image.data, dataWrap.ListWrap):
#                newData += self.image.data.dataList
#            else:
#                newData += [self.image.data]
#
#            if isinstance(other.data, dataWrap.ListWrap):
#                newData += other.data.dataList
#            else:
#                newData += [other.data]
#                
#            pre = common_prefix(newNames)
#            print pre
#            lPre = len(pre)
#            newNames = [n[lPre:] for n in newNames]
#
#            mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
#            mdh.setEntry('ChannelNames', newNames)
#
#            View3D(dataWrap.ListWrap(newData, 3), 'Composite', mdh=mdh, mode = self.dsviewer.mode, parent=self.dsviewer, glCanvas=self.dsviewer.glCanvas)
#
#        dlg.Destroy()
        
    def OnMakeComposites(self, event):
        imageNames =  image.openImages.keys()
        
        dispNames = []
        
        for iN in imageNames:
            im = image.openImages[iN]
            if im.data.shape[3] == 1: #single colour
                dispNames.append(iN)
            else:
                for i in range(im.data.shape[3]):
                    dispNames.append('%s$%d$%s' % (iN, i, im.names[i]))
        
        dlg = wx.MultiChoiceDialog(
                self.dsviewer, 'choose the images to composite with', 'Make Composite',
                dispNames,
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            others = [dispNames[n] for n in dlg.GetSelections()]
            
            if len(others) > 0:    
                ###TODO - Put checks on image size, voxel size etc ...
    
                #try:
                #    names = ['%s - %s' % (os.path.split(self.image.filename)[1], cn) for cn in  self.image.mdh.getEntry('ChannelNames')]
                #except:
                #    if self.image.data.shape[3] == 1:
                #        names = [os.path.split(self.image.filename)[1]]
                #    else:
                #        names = ['%s -  %d' % (os.path.split(self.image.filename)[1], d) for d in range(self.image.data.shape[3])]
                    
                newNames = []
                newData = []
                
                #if isinstance(self.image.data, dataWrap.ListWrap):
                #    newData += self.image.data.dataList
                #else:
                #    newData += [self.image.data]
                
                for otherN in others:
                    if otherN in imageNames:
                        other = image.openImages[otherN]
                        chan = 0
                    else:
                        n_, chan_, cn_ = otherN.split('$')
                        other = image.openImages[n_]
                        chan = int(chan_)
                        
                    try:
                        cn = other.mdh.getEntry('ChannelNames')[chan]
                        otherName = '%s - %s' % (os.path.split(other.filename)[1], cn) 
                    except:
                        if other.data.shape[3] == 1:
                            otherName = os.path.split(other.filename)[1]
                        else:
                            otherName = '%s -  %d' % (os.path.split(other.filename)[1], chan)
        
                    newNames.append(otherName)
        
                    if isinstance(other.data, dataWrap.ListWrap):
                        newData += [other.data.dataList[chan]]
                    else:
                        newData += [other.data]
    
                pre = common_prefix(newNames)
                print pre
                lPre = len(pre)
                newNames = [n[lPre:] for n in newNames]                
                
                mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
                mdh.setEntry('ChannelNames', newNames)
    
                View3D(dataWrap.ListWrap(newData, 3), 'Composite', mdh=mdh, mode = self.dsviewer.mode, parent=wx.GetTopLevelParent(self.dsviewer), glCanvas=self.dsviewer.glCanvas)

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
            
    def OnAlignChannels(self, event):
        #try:
        #    names = self.image.mdh.getEntry('ChannelNames')
        #except:
        #    names = ['%d' % d for d in range(self.image.data.shape[3])]
        from PYME.Analysis.DataSources import AlignDataSource
        import PYME.DSView.dataWrap
        
        if isinstance(self.image.data, PYME.DSView.dataWrap.ListWrap):
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
    dsviewer.compos = compositor(dsviewer)
    #if 'chanList' in dir(dsviewer.image.data) and 'shifts' in dir(dsviewer.image.data.chanList[0]):
        #we have shiftable channels
        



