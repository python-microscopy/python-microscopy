#!/usr/bin/python
##################
# filterPane.py
#
# Copyright David Baddeley, 2010
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
from PYME.ui import UI_MAXSIZE #hack for sys.maxsize bug
import sys
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
import numpy as np
from PYME.contrib import dispatch

from PYME.ui import histLimits
from PYME.LMVis import editFilterDialog

def CreateFilterPane(panel, mapping, pipeline, visFr):
    pane = FilterPane(panel, mapping, pipeline, visFr)
    panel.AddPane(pane)
    return pane

class FilterPanel(wx.Panel):
    def __init__(self, parent, filterKeys, dataSource=None):
        """

        Parameters
        ----------
        parent : wx.Window

        filterKeys : dict
            Dictionary keys

        dataSource : function
            function to call to get the current data source
        """
        wx.Panel.__init__(self, parent)

        self.filterKeys = filterKeys
        self._dataSource = dataSource

        self.on_filter_changed=dispatch.Signal()

        #GUI stuff
        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.lFiltKeys = wx.ListCtrl(self, -1, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.SUNKEN_BORDER, size=(-1, 25*(max(len(self.filterKeys.keys())+1, 5))))

        self.lFiltKeys.InsertColumn(0, 'Key')
        self.lFiltKeys.InsertColumn(1, 'Min')
        self.lFiltKeys.InsertColumn(2, 'Max')

        self.populate()

        self.lFiltKeys.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        self.lFiltKeys.SetColumnWidth(1, wx.LIST_AUTOSIZE)
        self.lFiltKeys.SetColumnWidth(2, wx.LIST_AUTOSIZE)

        # only do this part the first time so the events are only bound once
        if not hasattr(self, "ID_FILT_ADD"):
            self.ID_FILT_ADD = wx.NewId()
            self.ID_FILT_DELETE = wx.NewId()
            self.ID_FILT_EDIT = wx.NewId()

            self.Bind(wx.EVT_MENU, self.OnFilterAdd, id=self.ID_FILT_ADD)
            self.Bind(wx.EVT_MENU, self.OnFilterDelete, id=self.ID_FILT_DELETE)
            self.Bind(wx.EVT_MENU, self.OnFilterEdit, id=self.ID_FILT_EDIT)

        # for wxMSW
        self.lFiltKeys.Bind(wx.EVT_COMMAND_RIGHT_CLICK, self.OnFilterListRightClick)

        # for wxGTK
        self.lFiltKeys.Bind(wx.EVT_RIGHT_UP, self.OnFilterListRightClick)

        self.lFiltKeys.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnFilterItemSelected)
        self.lFiltKeys.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.OnFilterItemDeselected)
        self.lFiltKeys.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnFilterEdit)
        

        vsizer.Add(self.lFiltKeys, 1, wx.ALL|wx.EXPAND, 0)
        #self.stNumFiltered = wx.StaticText(self, -1, '')
        #vsizer.Add(self.stNumFiltered, 0, wx.ALL | wx.EXPAND, 2)
        self.SetSizerAndFit(vsizer)
        
    def update(self, filter_keys, data_source):
        self.filterKeys = filter_keys
        self._dataSource = data_source
        
        self.populate()

    def populate(self):
        self.lFiltKeys.DeleteAllItems()
        ind = 0
        for key, value in self.filterKeys.items():
            ind = self.lFiltKeys.InsertItem(ind+1, key)
            self.lFiltKeys.SetItem(ind, 1, '%3.2f' % value[0])
            self.lFiltKeys.SetItem(ind, 2, '%3.2f' % value[1])

    @property
    def dataSource(self):
        if self._dataSource is None:
            return None
        elif callable(self._dataSource):
            #support passing data source as a callable
            return self._dataSource()
        else:
            return self._dataSource

    def OnFilterListRightClick(self, event):

        x = event.GetX()
        y = event.GetY()

        item, flags = self.lFiltKeys.HitTest((x, y))

        # make a menu
        menu = wx.Menu()
        # add some items
        menu.Append(self.ID_FILT_ADD, "Add")

        if item != wx.NOT_FOUND and flags & wx.LIST_HITTEST_ONITEM:
            self.currentFilterItem = item
            self.lFiltKeys.Select(item)

            menu.Append(self.ID_FILT_DELETE, "Delete")
            menu.Append(self.ID_FILT_EDIT, "Edit")

        # Popup the menu.  If an item is selected then its handler
        # will be called before PopupMenu returns.
        self.PopupMenu(menu)
        menu.Destroy()

    def OnFilterItemSelected(self, event):
        self.currentFilterItem = event.GetIndex()
        event.Skip()

    def OnFilterItemDeselected(self, event):
        self.currentFilterItem = None
        event.Skip()

    def OnFilterAdd(self, event):
        #key = self.lFiltKeys.GetItem(self.currentFilterItem).GetText()

        try:
            possibleKeys = list(self.dataSource.keys())
        except:
            possibleKeys = []

        dlg = editFilterDialog.FilterEditDialog(self, mode='new', possibleKeys=possibleKeys)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            minVal = float(dlg.tMin.GetValue())
            maxVal = float(dlg.tMax.GetValue())

            key = str(dlg.cbKey.GetValue())

            if key == "":
                return

            self.filterKeys[key] = [minVal, maxVal]

            ind = self.lFiltKeys.InsertItem(UI_MAXSIZE, key)
            self.lFiltKeys.SetItem(ind, 1, '%3.2f' % minVal)
            self.lFiltKeys.SetItem(ind, 2, '%3.2f' % maxVal)

        dlg.Destroy()

        self.on_filter_changed.send(self)

    def OnFilterDelete(self, event):
        it = self.lFiltKeys.GetItem(self.currentFilterItem)
        self.lFiltKeys.DeleteItem(self.currentFilterItem)
        self.filterKeys.pop(it.GetText())

        self.on_filter_changed.send(self)

    def OnFilterEdit(self, event):
        key = str(self.lFiltKeys.GetItem(self.currentFilterItem).GetText())
        minVal, maxVal = self.filterKeys[key]
        #dlg = editFilterDialog.FilterEditDialog(self, mode='edit', possibleKeys=[], key=key, minVal=self.filterKeys[key][0], maxVal=self.filterKeys[key][1])
        try:
            data = np.array(self.dataSource[key])

            dlg = histLimits.HistLimitDialog(self, data, minVal, maxVal, title=key, action=self.notify, key=key)
            ret = dlg.ShowModal()

            if ret == wx.ID_OK:
                #minVal = float(dlg.tMin.GetValue())
                #maxVal = float(dlg.tMax.GetValue())
                minVal, maxVal = dlg.GetLimits()

                self.filterKeys[key] = [minVal, maxVal]

                self.lFiltKeys.SetItem(self.currentFilterItem, 1, '%3.2f' % minVal)
                self.lFiltKeys.SetItem(self.currentFilterItem, 2, '%3.2f' % maxVal)

            dlg.Destroy()
        except (RuntimeError, KeyError, TypeError):
            dlg = editFilterDialog.FilterEditDialog(self, mode='edit', key=key,minVal=minVal, maxVal=maxVal)

            ret = dlg.ShowModal()

            if ret == wx.ID_OK:
                minVal = float(dlg.tMin.GetValue())
                maxVal = float(dlg.tMax.GetValue())

                self.filterKeys[key] = [minVal, maxVal]

                self.lFiltKeys.SetItem(self.currentFilterItem, 1, '%3.2f' % minVal)
                self.lFiltKeys.SetItem(self.currentFilterItem, 2, '%3.2f' % maxVal)

            dlg.Destroy()

        self.notify(key)

    def notify(self, key, limits=None):
        if limits is not None:
            minVal, maxVal = limits

            self.filterKeys[key] = [minVal, maxVal]

            self.lFiltKeys.SetItem(self.currentFilterItem, 1, '%3.2f' % minVal)
            self.lFiltKeys.SetItem(self.currentFilterItem, 2, '%3.2f' % maxVal)

        self.on_filter_changed.send(self)

class FilterPane(afp.foldingPane):
    def __init__(self, panel, filterKeys, pipeline, visFr):
        afp.foldingPane.__init__(self, panel, -1, caption="Output Filter", pinned = False)

        #self.filterKeys = filterKeys
        self.pipeline = pipeline
        self.visFr = visFr

        #self.lFiltKeys = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.LC_SINGLE_SEL|wx.SUNKEN_BORDER, size=(-1, 200))

        self.pFilter = FilterPanel(self, filterKeys, pipeline.selectedDataSource)
        self.pFilter.on_filter_changed.connect(pipeline.Rebuild)

        self.AddNewElement(self.pFilter)

        self.stFilterNumPoints = wx.StaticText(self, -1, '')

        self.setEventNumbers()
        self.pipeline.onRebuild.connect(self.setEventNumbers)
        
        self.AddNewElement(self.stFilterNumPoints)
        #self._pnl.AddFoldPanelWindow(self, self.stFilterNumPoints, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

        self.bClipToSelection = wx.Button(self, -1, 'Clip to selection')
        self.AddNewElement(self.bClipToSelection)
        #self._pnl.AddFoldPanelWindow(self, self.bClipToSelection, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)


        self.bClipToSelection.Bind(wx.EVT_BUTTON, self.OnFilterClipToSelection)
        
        pipeline.onRebuild.connect(self.update)
        
        
        try:        
            visFr.Bind(wx.EVT_MENU, self.OnFilterClipToSelection, id=visFr.ID_VIEW_CLIP_ROI)
        except AttributeError:
            pass
        
    @property
    def filterKeys(self):
        return self.pipeline.filterKeys
        
    def update(self, *args, **kwargs):#, filter_keys):
        #self.filterKeys = filter_keys
        self.pFilter.update(self.filterKeys, self.pipeline.selectedDataSource)

    def setEventNumbers(self,**kwargs):
        if not self.pipeline.filter is None:
            self.stFilterNumPoints.SetLabel('%d of %d events' % (len(self.pipeline.filter['x']), len(self.pipeline.selectedDataSource['x'])))
        
    def OnFilterClipToSelection(self, event):
        if 'x' in self.filterKeys.keys() or 'y' in self.filterKeys.keys():
            if 'x' in self.filterKeys.keys():
                i = 0
                while not self.pFilter.lFiltKeys.GetItemText(i) == 'x':
                    i +=1
                self.pFilter.lFiltKeys.DeleteItem(i)
                self.filterKeys.pop('x')
            if 'y' in self.filterKeys.keys():
                i = 0
                while not self.pFilter.lFiltKeys.GetItemText(i) == 'y':
                    i +=1
                self.pFilter.lFiltKeys.DeleteItem(i)
                self.filterKeys.pop('y')

            self.bClipToSelection.SetLabel('Clip to Selection')
        else:
            try:
                #old glcanvas
                x0, y0 = self.visFr.glCanvas.selectionStart
                x1, y1 = self.visFr.glCanvas.selectionFinish
            except AttributeError:
                #new glcanvas
                x0, y0 = self.visFr.glCanvas.selectionSettings.start
                x1, y1 = self.visFr.glCanvas.selectionSettings.finish

            if not 'x' in self.filterKeys.keys():
                indx = self.pFilter.lFiltKeys.InsertItem(UI_MAXSIZE, 'x')
            else:
                indx = [self.pFilter.lFiltKeys.GetItemText(i) for i in range(self.pFilter.lFiltKeys.GetItemCount())].index('x')

            if not 'y' in self.filterKeys.keys():
                indy = self.pFilter.lFiltKeys.InsertItem(UI_MAXSIZE, 'y')
            else:
                indy = [self.pFilter.lFiltKeys.GetItemText(i) for i in range(self.pFilter.lFiltKeys.GetItemCount())].index('y')


            self.filterKeys['x'] = (min(x0, x1), max(x0, x1))
            self.filterKeys['y'] = (min(y0, y1), max(y0,y1))

            self.pFilter.lFiltKeys.SetItem(indx,1, '%3.2f' % min(x0, x1))
            self.pFilter.lFiltKeys.SetItem(indx,2, '%3.2f' % max(x0, x1))

            self.pFilter.lFiltKeys.SetItem(indy,1, '%3.2f' % min(y0, y1))
            self.pFilter.lFiltKeys.SetItem(indy,2, '%3.2f' % max(y0, y1))

            self.bClipToSelection.SetLabel('Clear Clipping ROI')

        self.pipeline.Rebuild()

