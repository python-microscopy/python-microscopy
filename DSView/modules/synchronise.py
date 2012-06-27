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
#from PYME.Acquire import MetaDataHandler
#from PYME.DSView import image, View3D
#from PYME.DSView import dataWrap
from PYME.DSView import dsviewer_npy_nb


class syncer:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image

        self.compMenu = wx.Menu()

        dsviewer.mProcessing.AppendSeparator()
        SYNCHRONISE = wx.NewId()
        dsviewer.mProcessing.Append(SYNCHRONISE, "Sync Windows", "", wx.ITEM_NORMAL)

        #SPLIT_CHANNELS = wx.NewId()
        #dsviewer.mProcessing.Append(SPLIT_CHANNELS, "Split Channels", "", wx.ITEM_NORMAL)

        #dsviewer.mProcessing.AppendSeparator()

        dsviewer.Bind(wx.EVT_MENU, self.OnSynchronise, id=SYNCHRONISE)
        #dsviewer.Bind(wx.EVT_MENU, self.OnSplitChannels, id=SPLIT_CHANNELS)



    def OnSynchronise(self, event):
        dlg = wx.SingleChoiceDialog(
                self.dsviewer, 'choose the image to composite with', 'Make Composite',
                dsviewer_npy_nb.openViewers.keys(),
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            other = dsviewer_npy_nb.openViewers[dlg.GetStringSelection()]

            other.do.syncedWith.append(self.do)

        dlg.Destroy()

    

       
    


def Plug(dsviewer):
    dsviewer.compos = syncer(dsviewer)



