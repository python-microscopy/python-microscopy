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

currentSlide = [None]

WantSlideChangeNotification = []

from PYME.Acquire.MetaDataHandler import NestedClassMDHandler

slideMD = NestedClassMDHandler()

class AutoWidthListCtrl(wx.ListCtrl, listmix.ListCtrlAutoWidthMixin):
    def __init__(self, parent, ID, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        listmix.ListCtrlAutoWidthMixin.__init__(self)

class VirtList(wx.ListCtrl):
    def __init__(self, parent, ID, pos=wx.DefaultPosition,
                 size=(480, 300), style=wx.LC_REPORT|wx.LC_VIRTUAL):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        #listmix.ListCtrlAutoWidthMixin.__init__(self)
        
        #self.qs = qs

        self.InsertColumn(0, "Creator")
        self.InsertColumn(1, "Reference")
        self.InsertColumn(2, "Labelling")
        self.SetColumnWidth(0, 100)
        self.SetColumnWidth(1, 120)
        self.SetColumnWidth(2, 250)
        #r = qs[0]
        #self.SetItemCount(0)
        #self.SetFilter()

    def SetFilter(self, creator='', reference='', structure=''):
        self.qs = models.Slide.objects.filter(creator__contains=creator, reference__contains = reference, labelling__structure__contains= structure).order_by('-timestamp')
        self.SetItemCount(len(self.qs))

    def OnGetItemText(self, item, col):
        #print self.qs[item].desc()
        return self.qs[item].desc()[col]

    #def OnGetItemImage(self):



class SampleInfoDialog(wx.Dialog):
    def __init__(self, parent, init_filter=('', '', ''), acquiring=True):
        wx.Dialog.__init__(self, parent, -1, 'Select Slide')

        #self.mdh = mdh
        self.labels = []
        self.slideExists = False
        self.slide = None

        sizer1 = wx.BoxSizer(wx.VERTICAL)

        #qs = models.Slide.objects.all().order_by('-timestamp')
        self.lSlides = VirtList(self, -1)
        self.lSlides.SetFilter(*init_filter)
        self.lSlides.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSelectSlide)

        

        sizer1.Add(self.lSlides, 1, wx.ALL|wx.EXPAND, 5)

        #sizer3 = wx.BoxSizer(wx.HORIZONTAL)

        #self.tStructure = TextCtrlAutoComplete.TextCtrlAutoComplete(self, value='', choices=[''], size=(150,-1))
        #self.tStructure.SetEntryCallback(self.setStructureChoices)
        #sizer3.Add(self.tStructure, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        #self.tDye = TextCtrlAutoComplete.TextCtrlAutoComplete(self, value='', choices=[''], size=(150,-1))
        #self.tDye.SetEntryCallback(self.setDyeChoices)
        #sizer3.Add(self.tDye, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        #sizer3.Add()

        #self.bAddLabel = wx.Button(self, -1, 'Add')
        #self.bAddLabel.Bind(wx.EVT_BUTTON, self.OnAddLabel)
        #sizer3.Add(self.bAddLabel, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT, 5)
        #sizer2.Add(sizer3, 0, wx.ALL|wx.EXPAND, 5)

        #sizer1a.Add(sizer2, 0, wx.ALL|wx.EXPAND, 5)

        btAdd = wx.Button(self, wx.ID_ADD, 'New Slide')
        sizer1.Add(btAdd, 0, wx.ALL|wx.ALIGN_RIGHT, 5)
        btAdd.Bind(wx.EVT_BUTTON, self.OnAddSlide)

        sizer1a = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Filter by:'), wx.VERTICAL)

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        sizer2.Add(wx.StaticText(self, -1, 'Creator:'), 0, wx.LEFT|wx.RIGHT, 5)

        #self.tCreator = wx.TextCtrl(self, -1, lastCreator)
        self.tCreator = TextCtrlAutoComplete.TextCtrlAutoComplete(self, value=init_filter[0], choices=[lastCreator])
        self.tCreator.SetEntryCallback(self.setCreatorChoices)
        self.tCreator.SetToolTip(wx.ToolTip('This should be the person who mounted the slide & should have details about the slide ref in their lab book'))
        self.tCreator.Bind(wx.EVT_TEXT, self.OnFilterChange)
        sizer2.Add(self.tCreator, 1, wx.LEFT|wx.RIGHT, 5)

        sizer1a.Add(sizer2, 1, wx.ALL|wx.EXPAND, 5)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        sizer2.Add(wx.StaticText(self, -1, 'Reference:'), 0, wx.LEFT|wx.RIGHT, 5)

        #self.tSlideRef = wx.TextCtrl(self, -1, lastSlideRef)
        self.tSlideRef = TextCtrlAutoComplete.TextCtrlAutoComplete(self, value=init_filter[1], choices=[lastSlideRef])
        self.tSlideRef.SetEntryCallback(self.setSlideChoices)
        self.tSlideRef.SetToolTip(wx.ToolTip('This should be the reference #/code which is on the slide and in lab book'))
        self.tSlideRef.Bind(wx.EVT_TEXT, self.OnFilterChange)
        sizer2.Add(self.tSlideRef, 1, wx.LEFT|wx.RIGHT, 5)

        sizer1a.Add(sizer2, 1, wx.ALL|wx.EXPAND, 5)
        
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add(wx.StaticText(self, -1, 'Structure:'), 0, wx.LEFT|wx.RIGHT, 5)

        self.tStructure = TextCtrlAutoComplete.TextCtrlAutoComplete(self, value=init_filter[2], choices=[''], size=(150,-1))
        self.tStructure.SetEntryCallback(self.setStructureChoices)
        self.tStructure.Bind(wx.EVT_TEXT, self.OnFilterChange)
        sizer2.Add(self.tStructure, 1, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        sizer1a.Add(sizer2, 1, wx.ALL|wx.EXPAND, 5)


        #sizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Slide Notes:'), wx.VERTICAL)
        #self.tSlideNotes = wx.TextCtrl(self, -1, '', size=(350, 50), style=wx.TE_MULTILINE|wx.TE_PROCESS_ENTER)
        #sizer2.Add(self.tSlideNotes, 0, wx.ALL|wx.EXPAND, 5)

        #self.tSlideNotes.Enable(False)

        #sizer1a.Add(sizer2, 0, wx.ALL|wx.EXPAND, 5)

        sizer1.Add(sizer1a, 0, wx.ALL|wx.EXPAND, 5)

        if acquiring:
            sizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Acquisition Notes:'), wx.VERTICAL)
            self.tNotes = wx.TextCtrl(self, -1, '', size=(350, 150), style=wx.TE_MULTILINE|wx.TE_PROCESS_ENTER)
            sizer2.Add(self.tNotes, 0, wx.ALL|wx.EXPAND, 5)

            sizer1.Add(sizer2, 0, wx.ALL|wx.EXPAND, 5)

        btSizer = wx.StdDialogButtonSizer()

        self.bOK = wx.Button(self, wx.ID_OK)
        #self.bOK.SetDefault()
        self.bOK.Disable()

        btSizer.AddButton(self.bOK)

        #btn = wx.Button(self, wx.ID_CANCEL)

        #btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizerAndFit(sizer1)

        if acquiring and self.lSlides.GetItemCount() == 1:
            self.lSlides.Select(0)

        print 'foo;'

    def OnAddSlide(self, event):
        import webbrowser
        webbrowser.open('http://lmsrv1/admin/samples/slide/add/')


    def OnSelectSlide(self, event):
        i = event.GetIndex()
        print i
        self.slide = self.lSlides.qs[i]
        print self.slide
        self.bOK.Enable()

    def OnFilterChange(self, event):
        self.lSlides.SetFilter(self.tCreator.GetValue(),self.tSlideRef.GetValue(),self.tStructure.GetValue())
        event.Skip()

    def setCreatorChoices(self):
        cname = self.tCreator.GetValue()
        slref = self.tSlideRef.GetValue()
        current_choices = self.tCreator.GetChoices()

        if slref == '' or len(models.Slide.objects.filter(reference=slref)) ==0:
            choices = list(set([e.creator for e in models.Slide.objects.filter(creator__startswith=cname)]))
        else:
            choices = list(set([e.creator for e in models.Slide.objects.filter(creator__startswith=cname, reference=slref)]))
            
        if choices != current_choices:
            self.tCreator.SetChoices(choices)
            
        #self.GetSlideLabels()

        

    def setSlideChoices(self):
        cname = self.tCreator.GetValue()
        slref = self.tSlideRef.GetValue()
        current_choices = self.tSlideRef.GetChoices()

        if cname == '' or (len(models.Slide.objects.filter(creator=cname)) == 0):
            choices = list(set([e.reference for e in models.Slide.objects.filter(reference__startswith=slref)]))
        else:
            choices = list(set([e.reference for e in models.Slide.objects.filter(reference__startswith=slref, creator=cname)]))

        if choices != current_choices:
            self.tSlideRef.SetChoices(choices)

        #self.GetSlideLabels()

    def setStructureChoices(self):
        sname = self.tStructure.GetValue()

        current_choices = self.tStructure.GetChoices()

        choices = list(set([e.structure for e in models.Labelling.objects.filter(structure__startswith=sname)]))

        if choices != current_choices:
            self.tStructure.SetChoices(choices)

    def setDyeChoices(self):
        dname = self.tDye.GetValue()

        current_choices = self.tDye.GetChoices()

        choices = list(set([e.label for e in models.Labelling.objects.filter(label__startswith=dname)]))

        if choices != current_choices:
            self.tDye.SetChoices(choices)

    def PopulateMetadata(self, mdh, acquiring=True):
        global lastCreator, lastSlideRef
        currentSlide[0] = self.slide

        creator = self.slide.creator
        slideRef = self.slide.reference
        sampleNotes = self.slide.notes
        #notes = self.tNotes.GetValue()
        #sampleNotes = self.tSlideNotes.GetValue()

        if not slideRef == lastSlideRef or not creator == lastCreator:
            lastCreator = creator
            lastSlideRef = slideRef

            for f in WantSlideChangeNotification:
                f()

        #if len(creator) == 0:
        #    creator = '<none>'

        #if len(slideRef) == 0:
        #    slideRef = '<none>'

        #if len(notes) == 0:
        #    notes = '<none>'

        #if len(sampleNotes) == 0:
        #    sampleNotes = '<none>'
        
        mdh.setEntry('Sample.Creator', creator)
        mdh.setEntry('Sample.SlideRef', slideRef)
        mdh.setEntry('Sample.Notes', sampleNotes)
        #mdh.setEntry('AcquisitionNotes', notes)
        

#        labels = []
#        for i in range(self.gLabelling.GetNumberRows()):
#            labels.append((self.gLabelling.GetCellValue(i, 0),self.gLabelling.GetCellValue(i, 1)))

        mdh.setEntry('Sample.Labelling', [(l.structure, l.dye.shortName) for l in self.slide.labelling.all()])

        mdh.setEntry('AcquiringUser', nameUtils.getUsername())

        #slide = models.Slide.Get(creator, slideRef)

#        if not self.slideExists:
#            for struct, label in self.labels:
#                l = models.Labelling(slideID=slide, structure=struct, label=label)
#                l.notes = sampleNotes
#                l.save()


        if acquiring:
            notes = self.tNotes.GetValue()
            mdh.setEntry('Notes', notes)

            #createImage(mdh, self.slide, notes)
            #im = models.Image.GetOrCreate(mdh.getEntry('imageID'), nameUtils.getUsername(), self.slide, mdh.getEntry('StartTime'))
            #im.comments = notes
            #im.save()



class slidePanel(wx.Panel):
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.stSlideName = wx.StaticText(self, -1, '\n\n')
        hsizer.Add(self.stSlideName, 1, wx.EXPAND|wx.ALL, 2)

        self.bSetSlide = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        hsizer.Add(self.bSetSlide, 0, wx.EXPAND|wx.ALL, 2)

        self.bSetSlide.Bind(wx.EVT_BUTTON, self.OnSetSlide)

        self.SetSizerAndFit(hsizer)
        WantSlideChangeNotification.append(self.update)

    def OnSetSlide(self, event):
        prefillSampleData(self.GetParent())
        self.update()

    def update(self):
        cs = currentSlide[0]
        self.stSlideName.SetLabel('%s - %s\n%s' % (cs.creator, cs.reference, cs.labels()))




def prefillSampleData(parent):
    #global currentSlide

    dlg = SampleInfoDialog(parent, acquiring=False)

    if dlg.ShowModal() == wx.ID_OK:
        dlg.PopulateMetadata(slideMD, False)
        print 'bar'
        print dlg.slide
        currentSlide[0] = dlg.slide
        print currentSlide
    else:
        currentSlide[0] = None

    dlg.Destroy()


def getSampleData(parent, mdh):
    #global currentSlide

    print currentSlide
    cs = currentSlide[0]

    if cs:
        #dlg = SampleInfoDialog(parent, (cs.creator, cs.reference, ''))
        mdh.copyEntriesFrom(slideMD)
        #createImage(mdh, cs)
    else:
        dlg = SampleInfoDialog(parent)

        if dlg.ShowModal() == wx.ID_OK:
            dlg.PopulateMetadata(mdh)

            currentSlide[0] = dlg.slide
        else:
            currentSlide[0] = None

    dlg.Destroy()

def createImage(mdh, slide, comments=''):
    im = models.Image.GetOrCreate(mdh.getEntry('imageID'), nameUtils.getUsername(), slide, mdh.getEntry('StartTime'))
    im.comments = comments
    im.save()


