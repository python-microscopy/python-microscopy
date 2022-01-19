#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
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
import sys
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp

from PYME.ui import histLimits
from PYME.ui import editFilterDialog

def CreateFilterPane(panel, mapping, pipeline, visFr):
    pane = FilterPane(panel, mapping, pipeline, visFr)
    panel.AddPane(pane)
    return pane

from PYME.ui import UI_MAXSIZE #hack for sys.maxsize bug

class FilterPane(wx.Panel):
    def __init__(self, panel, filterKeys, pipeline, visFr):
        afp.foldingPane.__init__(self, panel, -1, caption="Filter", pinned = False)

        self.filterKeys = filterKeys
        self.pipeline = pipeline
        self.visFr = visFr

        self.lFiltKeys = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.LC_SINGLE_SEL|wx.SUNKEN_BORDER, size=(-1, 200))

        self.AddNewElement(self.lFiltKeys)

        self.lFiltKeys.InsertColumn(0, 'Key')
        self.lFiltKeys.InsertColumn(1, 'Min')
        self.lFiltKeys.InsertColumn(2, 'Max')

        for key, value in self.filterKeys.items():
            ind = self.lFiltKeys.InsertStringItem(UI_MAXSIZE, key)
            self.lFiltKeys.SetStringItem(ind,1, '%3.2f' % value[0])
            self.lFiltKeys.SetStringItem(ind,2, '%3.2f' % value[1])

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

        self.stFilterNumPoints = wx.StaticText(self, -1, '')

        if not self.pipeline.filter is None:
            self.stFilterNumPoints.SetLabel('%d of %d events' % (len(self.pipeline.filter['x']), len(self.pipeline.selectedDataSource['x'])))

        self.AddNewElement(self.stFilterNumPoints)
        #self._pnl.AddFoldPanelWindow(self, self.stFilterNumPoints, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

        self.bClipToSelection = wx.Button(self, -1, 'Clip to selection')
        self.AddNewElement(self.bClipToSelection)
        #self._pnl.AddFoldPanelWindow(self, self.bClipToSelection, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)


        self.bClipToSelection.Bind(wx.EVT_BUTTON, self.OnFilterClipToSelection)
        
        
        visFr.Bind(wx.EVT_MENU, self.OnFilterClipToSelection, id=visFr.ID_VIEW_CLIP_ROI)



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
        self.currentFilterItem = event.m_itemIndex

        event.Skip()

    def OnFilterItemDeselected(self, event):
        self.currentFilterItem = None

        event.Skip()

    def OnFilterClipToSelection(self, event):
        if 'x' in self.filterKeys.keys() or 'y' in self.filterKeys.keys():
            if 'x' in self.filterKeys.keys():
                i = 0
                while not self.lFiltKeys.GetItemText(i) == 'x':
                    i +=1
                self.lFiltKeys.DeleteItem(i)
                self.filterKeys.pop('x')
            if 'y' in self.filterKeys.keys():
                i = 0
                while not self.lFiltKeys.GetItemText(i) == 'y':
                    i +=1
                self.lFiltKeys.DeleteItem(i)
                self.filterKeys.pop('y')

            self.bClipToSelection.SetLabel('Clip to Selection')
        else:
            x0, y0 = self.visFr.glCanvas.selectionStart
            x1, y1 = self.visFr.glCanvas.selectionFinish

            if not 'x' in self.filterKeys.keys():
                indx = self.lFiltKeys.InsertStringItem(UI_MAXSIZE, 'x')
            else:
                indx = [self.lFiltKeys.GetItemText(i) for i in range(self.lFiltKeys.GetItemCount())].index('x')

            if not 'y' in self.filterKeys.keys():
                indy = self.lFiltKeys.InsertStringItem(UI_MAXSIZE, 'y')
            else:
                indy = [self.lFiltKeys.GetItemText(i) for i in range(self.lFiltKeys.GetItemCount())].index('y')


            self.filterKeys['x'] = (min(x0, x1), max(x0, x1))
            self.filterKeys['y'] = (min(y0, y1), max(y0,y1))

            self.lFiltKeys.SetStringItem(indx,1, '%3.2f' % min(x0, x1))
            self.lFiltKeys.SetStringItem(indx,2, '%3.2f' % max(x0, x1))

            self.lFiltKeys.SetStringItem(indy,1, '%3.2f' % min(y0, y1))
            self.lFiltKeys.SetStringItem(indy,2, '%3.2f' % max(y0, y1))

            self.bClipToSelection.SetLabel('Clear Clipping ROI')

        self.pipeline.Rebuild()

    def OnFilterAdd(self, event):
        #key = self.lFiltKeys.GetItem(self.currentFilterItem).GetText()

        possibleKeys = []
        if not self.pipeline.selectedDataSource is None:
            possibleKeys = self.pipeline.selectedDataSource.keys()

        dlg = editFilterDialog.FilterEditDialog(self, mode='new', possibleKeys=possibleKeys)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            minVal = float(dlg.tMin.GetValue())
            maxVal = float(dlg.tMax.GetValue())

            key = dlg.cbKey.GetValue().encode()

            if key == "":
                return

            self.filterKeys[key] = (minVal, maxVal)

            ind = self.lFiltKeys.InsertStringItem(UI_MAXSIZE, key)
            self.lFiltKeys.SetStringItem(ind,1, '%3.2f' % minVal)
            self.lFiltKeys.SetStringItem(ind,2, '%3.2f' % maxVal)

        dlg.Destroy()

        self.pipeline.Rebuild()

    def OnFilterDelete(self, event):
        it = self.lFiltKeys.GetItem(self.currentFilterItem)
        self.lFiltKeys.DeleteItem(self.currentFilterItem)
        self.filterKeys.pop(it.GetText())

        self.pipeline.Rebuild()

    def OnFilterEdit(self, event):
        key = self.lFiltKeys.GetItem(self.currentFilterItem).GetText()

        #dlg = editFilterDialog.FilterEditDialog(self, mode='edit', possibleKeys=[], key=key, minVal=self.filterKeys[key][0], maxVal=self.filterKeys[key][1])
        dlg = histLimits.HistLimitDialog(self, self.pipeline.selectedDataSource[key], self.filterKeys[key][0], self.filterKeys[key][1], title=key)
        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            #minVal = float(dlg.tMin.GetValue())
            #maxVal = float(dlg.tMax.GetValue())
            minVal, maxVal = dlg.GetLimits()

            self.filterKeys[key] = (minVal, maxVal)

            self.lFiltKeys.SetStringItem(self.currentFilterItem,1, '%3.2f' % minVal)
            self.lFiltKeys.SetStringItem(self.currentFilterItem,2, '%3.2f' % maxVal)

        dlg.Destroy()
        self.pipeline.Rebuild()
