#!/usr/bin/python

##################
# recArrayView.py
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

import wx
import wx.grid as gridlib
import numpy as np
from PYME.IO import tabular

class RecArrayTable(gridlib.PyGridTableBase):
    def __init__(self, recarray):
        gridlib.PyGridTableBase.__init__(self)
        self.recarray = recarray

    def GetNumberRows(self):
        return len(self.recarray)

    def GetNumberCols(self):
        return len(self.recarray[0])

    def IsEmptyCell(self, row, col):
        return False

    def GetValue(self, row, col):
        return str(self.recarray[row][col] )

    def SetValue(self, row, col, value):
        pass

    def GetColLabelValue(self, col):
        return self.recarray.dtype.names[col]


class TabularTable(gridlib.PyGridTableBase):
    def __init__(self, tabular):
        gridlib.PyGridTableBase.__init__(self)
        self._tabular = tabular

    def GetNumberRows(self):
        return len(self._tabular)

    def GetNumberCols(self):
        return len(self._tabular.keys())

    def IsEmptyCell(self, row, col):
        return False

    def GetValue(self, row, col):
        return str(self._tabular[self._tabular.keys()[col]][row])

    def SetValue(self, row, col, value):
        pass

    def GetColLabelValue(self, col):
        return self._tabular.keys()[col]


class ArrayTableGrid(gridlib.Grid):
    def __init__(self, parent, data):
        gridlib.Grid.__init__(self, parent, -1, size = (-1,-1))
        
        self.SetData(data)

    def SetData(self, data):
        if isinstance(data, tabular.TabularBase):
            table = TabularTable(tabular.CachingResultsFilter(data))
        else:
            table = RecArrayTable(data)

        # The second parameter means that the grid is to take ownership of the
        # table and will destroy it when done.  Otherwise you would need to keep
        # a reference to it and call it's Destroy method later.
        self.SetTable(table, True)


class ArrayPanel(wx.Panel):
    def __init__(self, parent, recarray):
        wx.Panel.__init__(self, parent)

        self.data = recarray

        sizer = wx.BoxSizer(wx.VERTICAL)
        
        tool_sizer = wx.BoxSizer(wx.HORIZONTAL)
        bSave = wx.BitmapButton(self, -1, wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE), style=wx.NO_BORDER | wx.BU_AUTODRAW, name='Save')
        bSave.Bind(wx.EVT_BUTTON, self.OnSave)
        tool_sizer.Add(bSave)
        
        sizer.Add(tool_sizer)

        self.grid = ArrayTableGrid(self, recarray)
        self.grid.SetSize((10,10))

        sizer.Add(self.grid, 1, wx.EXPAND, 0)
        self.SetSizerAndFit(sizer)
        #self.SetAutoLayout(True)

        wx.EVT_SIZE(self, self.OnSize)

    def OnSize(self, event):
        self.grid.SetSize(self.GetClientSize())
        #self.Refresh()
        event.Skip()
        
    def OnSave(self, event):
        filename = wx.SaveFileSelector("Save data as ...", 'HDF (*.hdf)|*.hdf|Comma separated text (*.csv)|*.csv')
        if not filename == '':
            if isinstance(self.data, tabular.TabularBase):
                data = self.data
            else:
                data = tabular.RecArraySource(self.data)
                
            if filename.endswith('.hdf'):
                data.to_hdf(filename)
            else:
                data.to_csv(filename)

                
class ArrayFrame(wx.Frame):
    def __init__(self, data, title='Data table', parent=-1):
        wx.Frame.__init__(self, parent, title=title, size=(800,600))
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        self._array_pan = ArrayPanel(self, data)
        sizer.Add(self._array_pan, 1, wx.EXPAND, 0)
        self.SetSizer(sizer)