#!/usr/bin/python

##################
# colourPanel.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.misc import wxPlotPanel
from PYME.misc import editList
#from PYME.misc import facsPlot
import numpy
import wx
from pylab import cm
import sys

class colourPlotPanel(wxPlotPanel.PlotPanel):
    def __init__(self, parent, visFrame, **kwargs ):
        self.visFrame = visFrame
        self.parent = parent

        wxPlotPanel.PlotPanel.__init__( self, parent, **kwargs )

    def draw( self ):
            """Draw data."""
            if not self.visFrame.filter or len(self.visFrame.filter['x']) == 0:
                return
            
            if not hasattr( self, 'subplot1' ):
                self.subplot1 = self.figure.add_axes([.12, .12, .87, .87])
                #self.subplot2 = self.figure.add_subplot( 122 )



            self.subplot1.cla()

            x, y = self.visFrame.filter['fitResults_Ag'], self.visFrame.filter['fitResults_Ar']

            #facsPlot.facsPlotContour(x, y, 100)
            self.subplot1.scatter(x, y, s=1, c='b', edgecolor='none')

            n, xedge, yedge = numpy.histogram2d(x, y, bins = [100,100], range=[(x.min(), x.max()), (y.min(), y.max())])

            self.subplot1.contour((xedge[:-1] + xedge[1:])/2, (yedge[:-1] + yedge[1:])/2, numpy.log(n.T + .1), 5, cmap = cm.copper_r, linewidths=2)
            self.subplot1.axis('image')
            self.subplot1.set_xlabel('Channel 1')
            self.subplot1.set_ylabel('Channel 2')

            x.sort()
            y.sort()

            xl = x[.999*len(x)]
            yl = y[.999*len(x)]

            self.subplot1.set_xlim((0, xl))
            self.subplot1.set_ylim((0, yl))


            for k, v in self.visFrame.fluorSpecies.items():
                self.subplot1.plot([0,xl], [0, (v/(1-v))*xl], lw=2, c=cm.gist_rainbow(v))

#            self.subplot1.plot(ed[:-1], a/float(numpy.diff(ed[:2])), color='b' )
#            self.subplot1.set_xticks([0, ed.max()])
#            self.subplot1.set_yticks([0, numpy.floor(a.max()/float(numpy.diff(ed[:2])))])
#            self.subplot2.cla()
#            self.subplot2.plot(ed[:-1], numpy.cumsum(a), color='g' )
#            self.subplot2.set_xticks([0, ed.max()])
#            self.subplot2.set_yticks([0, a.sum()])

class colourPanel(wx.Panel):
    def __init__(self, parent, visFrame, id=-1):
        wx.Panel.__init__(self, parent, id)
        bsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.visFr = visFrame

        self.colPlotPan = colourPlotPanel(self, visFrame)
        bsizer.Add(self.colPlotPan, 4,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#        hsizer.Add(wx.StaticText(self, -1, "x' = "), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#
#        self.tXExpr = wx.TextCtrl(self, -1, self.driftExprX, size=(130, -1))
#        hsizer.Add(self.tXExpr, 2,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)
#
#        bsizer.Add(hsizer, 0, wx.ALL, 0)
#
#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#        hsizer.Add(wx.StaticText(self, -1, "y' = "), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#
#        self.tYExpr = wx.TextCtrl(self, -1, self.driftExprY, size=(130,-1))
#        hsizer.Add(self.tYExpr, 2,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)
#
#        bsizer.Add(hsizer, 0, wx.ALL, 0)

        vsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Fluorophores'), wx.VERTICAL)

        self.lFluorSpecies = editList.EditListCtrl(self, -1, style=wx.LC_REPORT|wx.LC_SINGLE_SEL|wx.SUNKEN_BORDER, size=(150, 150))
        vsizer.Add(self.lFluorSpecies, 0, wx.ALL, 5)

        self.lFluorSpecies.InsertColumn(0, 'Name', width=50)
        self.lFluorSpecies.InsertColumn(1, 'Ag/(Ag + Ar)')
        self.lFluorSpecies.makeColumnEditable(1)

        for key, value in self.visFr.fluorSpecies.items():
            ind = self.lFluorSpecies.InsertStringItem(sys.maxint, key)
            self.lFluorSpecies.SetStringItem(ind,1, '%3.2f' % value)
            

#        self.lFluorSpecies.SetColumnWidth(0, wx.LIST_AUTOSIZE)
#        self.lFluorSpecies.SetColumnWidth(1, wx.LIST_AUTOSIZE)

        ## only do this part the first time so the events are only bound once
        #if not hasattr(self, "ID_FILT_ADD"):
        self.ID_SPEC_ADD = wx.NewId()
        self.ID_SPEC_DELETE = wx.NewId()
        #self.ID_FILT_EDIT = wx.NewId()

        self.Bind(wx.EVT_MENU, self.OnSpecAdd, id=self.ID_SPEC_ADD)
        self.Bind(wx.EVT_MENU, self.OnSpecDelete, id=self.ID_SPEC_DELETE)
        #self.Bind(wx.EVT_MENU, self.OnFilterEdit, id=self.ID_FILT_EDIT)

        # for wxMSW
        self.lFluorSpecies.Bind(wx.EVT_COMMAND_RIGHT_CLICK, self.OnSpecListRightClick)

        # for wxGTK
        self.lFluorSpecies.Bind(wx.EVT_RIGHT_UP, self.OnSpecListRightClick)

#        self.lFluorSpecies.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSpecItemSelected)
#        self.lFluorSpecies.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.OnSpecItemDeselected)
        #self.lFluorSpecies.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnFilterEdit)

        self.lFluorSpecies.Bind(wx.EVT_LIST_END_LABEL_EDIT, self.OnSpecChange)

        bsizer.Add(vsizer, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.SetSizer(bsizer)
        bsizer.Fit(self)



    def OnSpecListRightClick(self, event):

        x = event.GetX()
        y = event.GetY()

        item, flags = self.lFluorSpecies.HitTest((x, y))


        # make a menu
        menu = wx.Menu()
        # add some items
        menu.Append(self.ID_SPEC_ADD, "Add")

        if item != wx.NOT_FOUND and flags & wx.LIST_HITTEST_ONITEM:
            self.currentFilterItem = item
            self.lFluorSpecies.Select(item)

            menu.Append(self.ID_SPEC_DELETE, "Delete")
            #menu.Append(self.ID_FILT_EDIT, "Edit")

        # Popup the menu.  If an item is selected then its handler
        # will be called before PopupMenu returns.
        self.PopupMenu(menu)
        menu.Destroy()



    def OnSpecAdd(self, event):
        #key = self.lFluorSpecies.GetItem(self.currentFilterItem).GetText()

        dlg = specAddDialog(self)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            val = float(dlg.tVal.GetValue())

            key = dlg.tKey.GetValue().encode()

            if key == "":
                return

            self.visFr.fluorSpecies[key] = val

            ind = self.lFluorSpecies.InsertStringItem(sys.maxint, key)
            self.lFluorSpecies.SetStringItem(ind,1, '%3.2f' % val)
            

        dlg.Destroy()
        self.refresh()


    def OnSpecDelete(self, event):
        it = self.lFluorSpecies.GetItem(self.currentFilterItem)
        self.lFluorSpecies.DeleteItem(self.currentFilterItem)
        self.visFr.fluorSpecies.pop(it.GetText())

        self.refresh()

    def OnSpecChange(self, event=None):
        it = self.lFluorSpecies.GetItem(event.m_itemIndex)

        self.visFr.fluorSpecies[it.GetText()] = float(event.m_item.GetText())

        self.refresh()

    def refresh(self):
        self.colPlotPan.draw()
        self.colPlotPan._SetSize()
        #self.colPlotPan.Refresh()




class specAddDialog(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, title='New signature')

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        sizer2.Add(wx.StaticText(self, -1, 'Name:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.tKey = wx.TextCtrl(self, -1, value='', size=(150, -1))

        sizer2.Add(self.tKey, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(wx.StaticText(self, -1, 'Ag/(Ag + Ar)'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.tVal = wx.TextCtrl(self, -1, '.5', size=(60, -1))


        sizer2.Add(self.tVal, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)


        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)


