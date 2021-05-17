#!/usr/bin/python

##################
# colourPanel.py
#
# Copyright David Baddeley, 2009
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

from PYME.contrib import wxPlotPanel
from PYME.ui import editList
#from PYME.misc import facsPlot
import numpy
import wx
# from pylab import cm
from matplotlib import cm
import sys
from . import dyeRatios

class colourPlotPanel(wxPlotPanel.PlotPanel):
    def __init__(self, parent, pipeline, visFrame, **kwargs ):
        self.pipeline = pipeline
        self.visFrame = visFrame
        self.parent = parent

        wxPlotPanel.PlotPanel.__init__( self, parent, **kwargs )

    def draw( self ):
        if self.IsShownOnScreen():
            #print self.IsShownOnScreen()
            print('d')
            """Draw data."""
            if self.visFrame.refv and not self.pipeline.ready or len(self.pipeline.filter['x']) == 0:
                return
            
            print('vd')            
            if not hasattr( self, 'subplot1' ):
                self.subplot1 = self.figure.add_axes([.1, .1, .89, .89])
                #self.subplot2 = self.figure.add_subplot( 122 )



            self.subplot1.cla()

            x, y = self.pipeline.filter['fitResults_Ag'], self.pipeline.filter['fitResults_Ar']
            n, xedge, yedge = numpy.histogram2d(x, y, bins = [100,100], range=[(x.min(), x.max()), (y.min(), y.max())])

            l_x = len(x)
            x = x[::int(max(l_x/1e4, 1))]
            y = y[::int(max(l_x/1e4, 1))]

            #facsPlot.facsPlotContour(x, y, 100)

            c = -5e3*numpy.ones(x.shape)
#            print (c < -1).sum(), c.min()

            cf = self.pipeline.colourFilter

            for k, v in self.pipeline.colour_mapper.species_ratios.items():
                p_dye = self.pipeline.filter['p_%s' % k][::int(max(l_x/1e4, 1))]

                p_other = numpy.zeros(x.shape)
                #p_tot = numpy.zeros(p_dye.shape)
                p_tot = cf.t_p_background*self.pipeline.filter['ColourNorm'][::int(max(l_x/1e4, 1))]

                for k2 in self.pipeline.colour_mapper.species_ratios.keys():
                    p_tot  += self.pipeline.filter['p_%s' % k2][::int(max(l_x/1e4, 1))]
                    if not k2 ==k:
                        p_other = numpy.maximum(p_other, self.pipeline.filter['p_%s' % k2][::int(max(l_x/1e4, 1))])

                p_dye = p_dye/p_tot
                p_other = p_other/p_tot

                c[(p_dye > cf.t_p_dye)*(p_other < cf.t_p_other)] = v

            cs = 0.75*cm.jet_r(c.copy())[:,:3]
            
            cs[c < -1, :] = (0.3*numpy.ones(cs.shape))[c < -1, :]

#            print (c < -1).sum(), c.min()
#            print (c < -1).sum(), cs[c < -1, :].shape




            self.subplot1.scatter(x, y, s=2, c=cs, edgecolor='none')
            #self.subplot1.plot(x, y, '.', ms=1, c=cs)

            #n, xedge, yedge = numpy.histogram2d(x, y, bins = [100,100], range=[(x.min(), x.max()), (y.min(), y.max())])

            self.subplot1.contour((xedge[:-1] + xedge[1:])/2, (yedge[:-1] + yedge[1:])/2, numpy.log2(n.T + .5), 7, cmap = cm.copper_r, linewidths=2)
            self.subplot1.axis('image')
            self.subplot1.set_xlabel('Channel 1')
            self.subplot1.set_ylabel('Channel 2')

            x.sort()
            y.sort()

            xl = x[int(.997*len(x))]
            yl = y[int(.997*len(x))]

            self.subplot1.set_xlim((0, xl))
            self.subplot1.set_ylim((0, yl))


            for k, v in self.pipeline.colour_mapper.species_ratios.items():
                self.subplot1.plot([0,xl], [0, ((1-v)/v)*xl], lw=2, c=cm.jet_r(v))

            self.canvas.draw()

#            self.subplot1.plot(ed[:-1], a/float(numpy.diff(ed[:2])), color='b' )
#            self.subplot1.set_xticks([0, ed.max()])
#            self.subplot1.set_yticks([0, numpy.floor(a.max()/float(numpy.diff(ed[:2])))])
#            self.subplot2.cla()
#            self.subplot2.plot(ed[:-1], numpy.cumsum(a), color='g' )
#            self.subplot2.set_xticks([0, ed.max()])
#            self.subplot2.set_yticks([0, a.sum()])

from PYME.ui import UI_MAXSIZE #hack for sys.maxsize bug

class colourPanel(wx.Panel):
    def __init__(self, parent, pipeline,visFrame, id=-1):
        wx.Panel.__init__(self, parent, id)
        bsizer = wx.BoxSizer(wx.VERTICAL)

        self.visFr = visFrame
        self.pipeline = pipeline

        self.colPlotPan = colourPlotPanel(self, pipeline, visFrame)
        bsizer.Add(self.colPlotPan, 4,wx.ALL|wx.EXPAND, 5)

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

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        vsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Fluorophores'), wx.VERTICAL)

        self.lFluorSpecies = editList.EditListCtrl(self, -1, style=wx.LC_REPORT|wx.LC_SINGLE_SEL|wx.SUNKEN_BORDER, size=(450, 100))
        vsizer.Add(self.lFluorSpecies, 0, wx.ALL, 5)

        self.lFluorSpecies.InsertColumn(0, 'Name')
        self.lFluorSpecies.InsertColumn(1, 'Ag/(Ag + Ar)')
        self.lFluorSpecies.InsertColumn(2, '# Events')
        self.lFluorSpecies.InsertColumn(3, 'dx')
        self.lFluorSpecies.InsertColumn(4, 'dy')
        self.lFluorSpecies.InsertColumn(5, 'dz')
        self.lFluorSpecies.makeColumnEditable(1)
        self.lFluorSpecies.makeColumnEditable(3)
        self.lFluorSpecies.makeColumnEditable(4)
        self.lFluorSpecies.makeColumnEditable(5)

        for key, value in self.pipeline.colour_mapper.species_ratios.items():
            ind = self.lFluorSpecies.InsertItem(UI_MAXSIZE, key)
            self.lFluorSpecies.SetItem(ind,1, '%3.2f' % value)
            self.lFluorSpecies.SetItemTextColour(ind, wx.Colour(*((128*numpy.array(cm.jet_r(value)))[:3])))
            
            

        self.lFluorSpecies.SetColumnWidth(3, 60)
        self.lFluorSpecies.SetColumnWidth(4, 60)
        self.lFluorSpecies.SetColumnWidth(5, 60)


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

        self.bGuess = wx.Button(self, -1, 'Guess')
        self.bGuess.Bind(wx.EVT_BUTTON, self.OnSpecGuess)
        vsizer.Add(self.bGuess, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)

        hsizer.Add(vsizer, 0, wx.ALL, 5)

        vsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Channel Assignment '), wx.VERTICAL)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2.Add(wx.StaticText(self, -1, 'p_dye:   '), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPBelong = wx.TextCtrl(self, -1, '%3.3f' % self.pipeline.colourFilter.t_p_dye)
        hsizer2.Add(self.tPBelong, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tPBelong.Bind(wx.EVT_TEXT, self.OnChangePDye)

        vsizer.Add(hsizer2, 0, wx.ALL, 0)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2.Add(wx.StaticText(self, -1, 'p_other:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPOther = wx.TextCtrl(self, -1, '%3.3f' % self.pipeline.colourFilter.t_p_other)
        hsizer2.Add(self.tPOther, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tPOther.Bind(wx.EVT_TEXT, self.OnChangePOther)

        vsizer.Add(hsizer2, 0, wx.ALL, 0)

        hsizer.Add(vsizer, 0, wx.ALL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 5)

        self.SetSizer(bsizer)
        bsizer.Fit(self)

        self.Bind(wx.EVT_SHOW, self.OnShow)

    def OnChangePDye(self, event):
        self.pipeline.colourFilter.t_p_dye = float(self.tPBelong.GetValue())
        self.refresh()

    def OnChangePOther(self, event):
        self.pipeline.colourFilter.t_p_other = float(self.tPOther.GetValue())
        self.refresh()

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

            key = str(dlg.tKey.GetValue())  # unicode builds of wx3 return a unicode str, which breaks wx.ListCtrl.InsertStringItem (on py2)

            if key == "":
                return

            #update dict ex-situ and assign to ensure that trait handlers get triggered and UI updates.
            ratios = dict(self.pipeline.colour_mapper.species_ratios)
            ratios[key] = val
            self.pipeline.colour_mapper.ratios_from_metadata = False
            self.pipeline.colour_mapper.species_ratios = ratios
            
            self.pipeline.Rebuild()

        dlg.Destroy()
        self.refresh()


    def OnSpecDelete(self, event):
        it = self.lFluorSpecies.GetItem(self.currentFilterItem)
        self.lFluorSpecies.DeleteItem(self.currentFilterItem)
        
        #update dict ex-situ and assign to ensure that trait handlers get triggered and UI updates.
        ratios = dict(self.pipeline.colour_mapper.species_ratios)
        ratios.pop(str(it.GetText()))
        self.pipeline.colour_mapper.ratios_from_metadata = False
        self.pipeline.colour_mapper.species_ratios = ratios

        
        self.pipeline.Rebuild()

        #self.visFr.UpdatePointColourChoices()
        #self.visFr.colourFilterPane.UpdateColourFilterChoices()

        self.refresh()

    def OnSpecChange(self, event=None):
        it = self.lFluorSpecies.GetItem(event.m_itemIndex)

        val = float(event.m_item.GetText())

        col = event.GetColumn()

        if col == 1: #frac
            #update dict ex-situ and assign to ensure that trait handlers get triggered and UI updates.
            ratios = dict(self.pipeline.colour_mapper.species_ratios)
            ratios[str(it.GetText())] = val
            self.pipeline.colour_mapper.ratios_from_metadata = False
            self.pipeline.colour_mapper.species_ratios = ratios

            self.pipeline.Rebuild()

            self.refresh()
        else: #shift
            axis = ['x', 'y', 'z'][col-3]
            specName = str(it.GetText())
            if not specName in self.pipeline.chromaticShifts.keys():
                self.pipeline.chromaticShifts[specName] = {}

            self.pipeline.chromaticShifts[specName][axis] = val
            self.visFr.RefreshView()

    def OnSpecGuess(self, event):
        import scipy.cluster
        
        #count number of peaks in gFrac histogram
        n = (numpy.diff(numpy.sign(numpy.diff(numpy.histogram(self.pipeline.filter['gFrac'], numpy.linspace(0, 1, 20))[0]))) < 0).sum()

        guesses = scipy.cluster.vq.kmeans(self.pipeline.filter['gFrac'], n)[0]
 
        ratios = {}
        for g, i in zip(guesses, range(n)):
            key = '%c' % (65 + i)
            ratios[key] = float(g)

        self.pipeline.colour_mapper.ratios_from_metadata = False
        self.pipeline.colour_mapper.species_ratios=ratios
        self.pipeline.Rebuild()

        try:
            self.visFr.UpdatePointColourChoices()
        
            self.visFr.colourFilterPane.UpdateColourFilterChoices()
        except AttributeError:
            pass

        self.refresh()


    def SpecFromMetadata(self, mdh):
        labels = mdh.getEntry('Sample.Labelling')

        self.pipeline.colour_mapper.ratios_from_metadata = False
        
        #TODO - do we need the following, or can the ProcessColour module handle this internally?
        structures = []
        ratios = {}
        for structure, dye in labels:
            if structure in structures: #duplicate structure
                structure += ' A'
            structures.append(structure)
            ratio = dyeRatios.getRatio(dye, mdh)

            if not ratio is None:
                ratios[structure] = float(ratio)

        self.pipeline.colour_mapper.species_ratios = ratios
        self.pipeline.Rebuild()

        try:
            self.visFr.UpdatePointColourChoices()
            self.visFr.colourFilterPane.UpdateColourFilterChoices()
        except AttributeError:
            pass

        self.refresh()

        

    def OnShow(self, event):
        #print event.IsShown()
        if event.IsShown() and not self.visFr.adding_panes:
            self.refresh()

    def refresh(self):
        if not self.IsShown():
            return
        
        self.colPlotPan.draw()
        
        self.lFluorSpecies.DeleteAllItems()

        for key in self.pipeline.colour_mapper.species_ratios.keys():
            ind = self.lFluorSpecies.InsertStringItem(UI_MAXSIZE, key)

            ratio = self.pipeline.colour_mapper.species_ratios[key]
            self.lFluorSpecies.SetStringItem(ind, 1, '%3.3f' % ratio)
            self.lFluorSpecies.SetItemTextColour(ind, wx.Colour(*((128 * numpy.array(cm.jet_r(ratio)))[:3])))
            
            num_dyes = sum(self.pipeline.colourFilter._index(key))

            self.lFluorSpecies.SetStringItem(ind,2, '%d' % num_dyes)


        #self.colPlotPan._SetSize()
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


