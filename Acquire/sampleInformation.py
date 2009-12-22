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
#import wx.grid
import  wx.lib.mixins.listctrl  as  listmix
import sys

from PYME.FileUtils import nameUtils
from PYME.misc import TextCtrlAutoComplete

from PYME.SampleDB import populate #just to setup the Django environment
from PYME.SampleDB.samples import models

lastCreator = nameUtils.getUsername()
lastSlideRef = ''

class AutoWidthListCtrl(wx.ListCtrl, listmix.ListCtrlAutoWidthMixin):
    def __init__(self, parent, ID, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        listmix.ListCtrlAutoWidthMixin.__init__(self)


class SampleInfoDialog(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, -1, 'Sample Information')

        #self.mdh = mdh
        self.labels = []
        self.slideExists = False

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        sizer1a = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Slide:'), wx.VERTICAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        sizer2.Add(wx.StaticText(self, -1, 'Slide Creator:'), 0, wx.ALL, 5)

        #self.tCreator = wx.TextCtrl(self, -1, lastCreator)
        self.tCreator = TextCtrlAutoComplete.TextCtrlAutoComplete(self, value=lastCreator, choices=[lastCreator])
        self.tCreator.SetEntryCallback(self.setCreatorChoices)
        self.tCreator.SetToolTip(wx.ToolTip('This should be the person who mounted the slide & should have details about the slide ref in their lab book'))
        sizer2.Add(self.tCreator, 1, wx.ALL, 5)

        sizer2.Add(wx.StaticText(self, -1, 'Slide Ref:'), 0, wx.ALL, 5)

        #self.tSlideRef = wx.TextCtrl(self, -1, lastSlideRef)
        self.tSlideRef = TextCtrlAutoComplete.TextCtrlAutoComplete(self, value=lastSlideRef, choices=[lastSlideRef])
        self.tSlideRef.SetEntryCallback(self.setSlideChoices)
        self.tSlideRef.SetToolTip(wx.ToolTip('This should be the reference #/code which is on the slide and in lab book'))
        sizer2.Add(self.tSlideRef, 0, wx.ALL, 5)

        sizer1a.Add(sizer2, 1, wx.ALL|wx.EXPAND, 5)


        sizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Labelling:'), wx.VERTICAL)
#        self.gLabelling = wx.grid.Grid(self, -1, size=(250, 100))
#        self.gLabelling.SetDefaultColSize(235/2)
#        self.gLabelling.CreateGrid(1, 2)
#        self.gLabelling.SetRowLabelSize(0)
#        self.gLabelling.SetColLabelValue(0,'Structure')
#        self.gLabelling.SetColLabelValue(1,'Dye')
        self.lLabelling = AutoWidthListCtrl(self, -1, style=wx.LC_REPORT| wx.BORDER_NONE)
        self.lLabelling.InsertColumn(0, "Structure")
        self.lLabelling.InsertColumn(1, "Dye")

        

        # for wxMSW
        self.lLabelling.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK, self.OnLabellingRightClick)

        self.ID_LABEL_DELETE = wx.NewId()
        self.Bind(wx.EVT_MENU, self.OnLabelDelete, id=self.ID_LABEL_DELETE)
#
#        # for wxGTK
#        self.lLabelling.Bind(wx.EVT_RIGHT_UP, self.OnLabellingRightClick)

        

        sizer2.Add(self.lLabelling, 1, wx.ALL|wx.EXPAND, 5)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)

        self.tStructure = TextCtrlAutoComplete.TextCtrlAutoComplete(self, value='', choices=[''], size=(150,-1))
        self.tStructure.SetEntryCallback(self.setStructureChoices)
        sizer3.Add(self.tStructure, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tDye = TextCtrlAutoComplete.TextCtrlAutoComplete(self, value='', choices=[''], size=(150,-1))
        self.tDye.SetEntryCallback(self.setDyeChoices)
        sizer3.Add(self.tDye, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        #sizer3.Add()

        self.bAddLabel = wx.Button(self, -1, 'Add')
        self.bAddLabel.Bind(wx.EVT_BUTTON, self.OnAddLabel)
        sizer3.Add(self.bAddLabel, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT, 5)
        sizer2.Add(sizer3, 0, wx.ALL|wx.EXPAND, 5)

        sizer1a.Add(sizer2, 0, wx.ALL|wx.EXPAND, 5)

        sizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Slide Notes:'), wx.VERTICAL)
        self.tSlideNotes = wx.TextCtrl(self, -1, '', size=(350, 50), style=wx.TE_MULTILINE|wx.TE_PROCESS_ENTER)
        sizer2.Add(self.tSlideNotes, 0, wx.ALL|wx.EXPAND, 5)

        sizer1a.Add(sizer2, 0, wx.ALL|wx.EXPAND, 5)

        sizer1.Add(sizer1a, 0, wx.ALL|wx.EXPAND, 5)


        sizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Acquisition Notes:'), wx.VERTICAL)
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

        self.GetSlideLabels()


    def OnLabellingRightClick(self, event):

#        x = event.GetX()
#        y = event.GetY()
#
#        item, flags = self.lFiltKeys.HitTest((x, y))

        if self.slideExists:
            return

        self.rcind = event.GetIndex()


        # make a menu
        menu = wx.Menu()
        # add some items
#        menu.Append(self.ID_FILT_ADD, "Add")
#
#        if item != wx.NOT_FOUND and flags & wx.LIST_HITTEST_ONITEM:
#            self.currentFilterItem = item
#            self.lFiltKeys.Select(item)

        menu.Append(self.ID_LABEL_DELETE, "Delete")
#            menu.Append(self.ID_FILT_EDIT, "Edit")

        # Popup the menu.  If an item is selected then its handler
        # will be called before PopupMenu returns.
        self.PopupMenu(menu)
        menu.Destroy()

        self.PopulateLabelList()

    def OnLabelDelete(self, event):
        self.labels.pop(self.rcind)


    def GetSlideLabels(self):
        cname = self.tCreator.GetValue()
        slref = self.tSlideRef.GetValue()

        if len(models.Slide.objects.filter(reference=slref, creator=cname)) > 0:
            slide = models.Slide.objects.get(reference=slref, creator=cname)

            self.labels = [(l.structure, l.label) for l in slide.labelling.all()]

            self.slideExists = True
            self.tStructure.Enable(False)
            self.tDye.Enable(False)
            self.bAddLabel.Enable(False)
            self.tSlideNotes.SetValue(slide.notes)
            self.tSlideNotes.Enable(False)
        else:
            self.slideExists = False
            self.tStructure.Enable(True)
            self.tDye.Enable(True)
            self.bAddLabel.Enable(True)
            self.tSlideNotes.Enable(True)

        self.PopulateLabelList()


    def PopulateLabelList(self):
        self.lLabelling.DeleteAllItems()

        for struct, label in self.labels:
            index = self.lLabelling.InsertStringItem(sys.maxint, struct)
            self.lLabelling.SetStringItem(index, 1, label)


        self.lLabelling.SetColumnWidth(0, 325/2)
        self.lLabelling.SetColumnWidth(1, 325/2)


    def OnAddLabel(self, event):
        struct = self.tStructure.GetValue()
        dye = self.tDye.GetValue()
        if not struct == '' and not dye == '':
            self.labels.append((struct, dye))
            self.PopulateLabelList()

        self.tStructure.SetValue('')
        self.tDye.SetValue('')

    def setCreatorChoices(self):
        cname = self.tCreator.GetValue()
        slref = self.tSlideRef.GetValue()
        current_choices = self.tCreator.GetChoices()

        if slref == '' or len(models.Slide.objects.filter(reference=slref)) ==0:
            choices = [e.creator for e in models.Slide.objects.filter(creator__startswith=cname)]
        else:
            choices = [e.creator for e in models.Slide.objects.filter(creator__startswith=cname, reference=slref)]
            
        if choices != current_choices:
            self.tCreator.SetChoices(choices)
            
        self.GetSlideLabels()

        

    def setSlideChoices(self):
        cname = self.tCreator.GetValue()
        slref = self.tSlideRef.GetValue()
        current_choices = self.tSlideRef.GetChoices()

        if cname == '' or (len(models.Slide.objects.filter(creator=cname)) == 0):
            choices = [e.reference for e in models.Slide.objects.filter(reference__startswith=slref)]
        else:
            choices = [e.reference for e in models.Slide.objects.filter(reference__startswith=slref, creator=cname)]

        if choices != current_choices:
            self.tSlideRef.SetChoices(choices)

        self.GetSlideLabels()

    def setStructureChoices(self):
        sname = self.tStructure.GetValue()

        current_choices = self.tStructure.GetChoices()

        choices = [e.structure for e in models.Labelling.objects.filter(structure__startswith=sname)]

        if choices != current_choices:
            self.tStructure.SetChoices(choices)

    def setDyeChoices(self):
        dname = self.tDye.GetValue()

        current_choices = self.tDye.GetChoices()

        choices = [e.label for e in models.Labelling.objects.filter(label__startswith=dname)]

        if choices != current_choices:
            self.tDye.SetChoices(choices)

    def PopulateMetadata(self, mdh):
        global lastCreator, lastSlideRef

        creator = self.tCreator.GetValue()
        slideRef = self.tSlideRef.GetValue()
        notes = self.tNotes.GetValue()
        sampleNotes = self.tSlideNotes.GetValue()

        lastCreator = creator
        lastSlideRef = slideRef

        if len(creator) == 0:
            creator = '<none>'

        if len(slideRef) == 0:
            slideRef = '<none>'

        if len(notes) == 0:
            notes = '<none>'

        if len(sampleNotes) == 0:
            sampleNotes = '<none>'
        
        mdh.setEntry('Sample.Creator', creator)
        mdh.setEntry('Sample.SlideRef', slideRef)
        mdh.setEntry('Sample.Notes', sampleNotes)
        mdh.setEntry('Notes', notes)

#        labels = []
#        for i in range(self.gLabelling.GetNumberRows()):
#            labels.append((self.gLabelling.GetCellValue(i, 0),self.gLabelling.GetCellValue(i, 1)))

        mdh.setEntry('Sample.Labelling', self.labels)

        mdh.setEntry('AcquiringUser', nameUtils.getUsername())

        slide = models.Slide.GetOrCreate(creator, slideRef)

        if not self.slideExists:
            for struct, label in self.labels:
                l = models.Labelling(slideID=slide, structure=struct, label=label)
                l.notes = sampleNotes
                l.save()

        im = models.Image.GetOrCreate(mdh.getEntry('imageID'), nameUtils.getUsername(), slide, mdh.getEntry('StartTime'))
        im.comments = notes
        im.save()



def getSampleData(parent, mdh):
    dlg = SampleInfoDialog(parent)

    if dlg.ShowModal() == wx.ID_OK:
        dlg.PopulateMetadata(mdh)

    dlg.Destroy()


