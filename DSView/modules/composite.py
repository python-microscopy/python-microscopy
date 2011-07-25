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
#import pylab
from PYME.Acquire import MetaDataHandler
from PYME.DSView import image, View3D
from PYME.DSView import dataWrap


class compositor:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image

        self.compMenu = wx.Menu()

        MAKE_COMPOSITE = wx.NewId()
        dsviewer.mProcessing.Append(MAKE_COMPOSITE, "Make Composite", "", wx.ITEM_NORMAL)

        dsviewer.Bind(wx.EVT_MENU, self.OnMakeComposite, id=MAKE_COMPOSITE)



    def OnMakeComposite(self, event):
        dlg = wx.SingleChoiceDialog(
                self.dsviewer, 'choose the image to composite with', 'Make Composite',
                image.openImages.keys(),
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            other = image.openImages[dlg.GetStringSelection()]

            ###TODO - Put checks on image size, voxel size etc ...

            try:
                names = self.image.mdh.getEntry('ChannelNames')
            except:
                names = ['%s -  %d' % (self.image.filename, d) for d in range(self.image.data.shape[3])]

            try:
                otherNames = other.mdh.getEntry('ChannelNames')
            except:
                otherNames = ['%s -  %d' % (self.image.filename, d) for d in range(other.data.shape[3])]

            newNames = names + otherNames

            newData = []
            if isinstance(self.image.data, dataWrap.ListWrap):
                newData += self.image.data.dataList
            else:
                newData += [self.image.data]

            if isinstance(other.data, dataWrap.ListWrap):
                newData += other.data.dataList
            else:
                newData += [other.data]

            mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
            mdh.setEntry('ChannelNames', newNames)

            View3D(dataWrap.ListWrap(newData, 3), '< Composite >', mdh=mdh, parent=self.dsviewer)

        dlg.Destroy()

       
    


def Plug(dsviewer):
    dsviewer.compos = compositor(dsviewer)



