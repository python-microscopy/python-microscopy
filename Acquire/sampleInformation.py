#!/usr/bin/python

##################
# sampleInformation.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
import wx.grid

from PYME.FileUtils import nameUtils

lastCreator = nameUtils.getUsername()
lastSlideRef = ''


class SampleInfoDialog(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, -1, 'Sample Information')

        #self.mdh = mdh

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        sizer2.Add(wx.StaticText(self, -1, 'Slide Creator:'), 0, wx.ALL, 5)

        self.tCreator = wx.TextCtrl(self, -1, lastCreator)
        self.tCreator.SetToolTip(wx.ToolTip('This should be the person who mounted the slide & should have details about the slide ref in their lab book'))
        sizer2.Add(self.tCreator, 1, wx.ALL, 5)

        sizer2.Add(wx.StaticText(self, -1, 'Slide Ref:'), 0, wx.ALL, 5)

        self.tSlideRef = wx.TextCtrl(self, -1, lastSlideRef)
        self.tSlideRef.SetToolTip(wx.ToolTip('This should be the reference #/code which is on the slide and in lab book'))
        sizer2.Add(self.tSlideRef, 0, wx.ALL, 5)

        sizer1.Add(sizer2, 1, wx.ALL|wx.EXPAND, 5)


        sizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Labelling:'), wx.HORIZONTAL)
        self.gLabelling = wx.grid.Grid(self, -1, size=(250, 100))
        self.gLabelling.SetDefaultColSize(235/2)
        self.gLabelling.CreateGrid(1, 2)
        self.gLabelling.SetRowLabelSize(0)
        self.gLabelling.SetColLabelValue(0,'Structure')
        self.gLabelling.SetColLabelValue(1,'Dye')

        
        sizer2.Add(self.gLabelling, 1, wx.ALL|wx.EXPAND, 5)

        self.bAddLabel = wx.Button(self, -1, 'Add')
        self.bAddLabel.Bind(wx.EVT_BUTTON, self.OnAddLabel)
        sizer2.Add(self.bAddLabel, 0, wx.ALL|wx.ALIGN_BOTTOM, 5)

        sizer1.Add(sizer2, 0, wx.ALL|wx.EXPAND, 5)


        sizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Notes:'), wx.VERTICAL)
        self.tNotes = wx.TextCtrl(self, -1, '', size=(350, 150), style=wx.TE_MULTILINE|wx.TE_PROCESS_ENTER)
        sizer2.Add(self.tNotes, 0, wx.ALL|wx.EXPAND, 5)

        sizer1.Add(sizer2, 0, wx.ALL|wx.EXPAND, 5)

        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        #btn = wx.Button(self, wx.ID_CANCEL)

        #btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizerAndFit(sizer1)

    def OnAddLabel(self, event):
        self.gLabelling.AppendRows(1)

    def PopulateMetadata(self, mdh):
        global lastCreator, lastSlideRef

        creator = self.tCreator.GetValue()
        slideRef = self.tSlideRef.GetValue()
        notes = self.tNotes.GetValue()

        lastCreator = creator
        lastSlideRef = slideRef

        if len(creator) == 0:
            creator = '<none>'

        if len(slideRef) == 0:
            slideRef = '<none>'

        if len(notes) == 0:
            notes = '<none>'
        
        mdh.setEntry('Sample.Creator', creator)
        mdh.setEntry('Sample.SlideRef', slideRef)
        mdh.setEntry('Sample.Notes', notes)

        labels = []
        for i in range(self.gLabelling.GetNumberRows()):
            labels.append((self.gLabelling.GetCellValue(i, 0),self.gLabelling.GetCellValue(i, 1)))

        mdh.setEntry('Sample.Labelling', labels)


def getSampleData(parent, mdh):
    dlg = SampleInfoDialog(parent)

    if dlg.ShowModal() == wx.ID_OK:
        dlg.PopulateMetadata(mdh)

    dlg.Destroy()


