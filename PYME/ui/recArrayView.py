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
import  wx.grid as  gridlib

class RecArrayTable(gridlib.PyGridTableBase):

    def __init__(self, recarray):
        gridlib.PyGridTableBase.__init__(self)
        self.recarray = recarray

#        self.odd=gridlib.GridCellAttr()
#        self.odd.SetBackgroundColour("sky blue")
#        self.even=gridlib.GridCellAttr()
#        self.even.SetBackgroundColour("sea green")
#
#    def GetAttr(self, row, col, kind):
#        attr = [self.even, self.odd][row % 2]
#        attr.IncRef()
#        return attr



    # This is all it takes to make a custom data table to plug into a
    # wxGrid.  There are many more methods that can be overridden, but
    # the ones shown below are the required ones.  This table simply
    # provides strings containing the row and column values.

    def GetNumberRows(self):
        return len(self.recarray)

    def GetNumberCols(self):
        return len(self.recarray[0])

    def IsEmptyCell(self, row, col):
        return False

    def GetValue(self, row, col):
        return str( self.recarray[row][col] )

    def SetValue(self, row, col, value):
        pass
        #self.log.write('SetValue(%d, %d, "%s") ignored.\n' % (row, col, value))

    def GetColLabelValue(self, col):
        return self.recarray.dtype.names[col]




class RecarrayTableGrid(gridlib.Grid):
    def __init__(self, parent, recarray):
        gridlib.Grid.__init__(self, parent, -1, size = (-1,-1))

        table = RecArrayTable(recarray)

        # The second parameter means that the grid is to take ownership of the
        # table and will destroy it when done.  Otherwise you would need to keep
        # a reference to it and call it's Destroy method later.
        self.SetTable(table, True)

    def SetData(self, recarray):
        table = RecArrayTable(recarray)

        # The second parameter means that the grid is to take ownership of the
        # table and will destroy it when done.  Otherwise you would need to keep
        # a reference to it and call it's Destroy method later.
        self.SetTable(table, True)
    

class recArrayPanel(wx.Panel):
    def __init__(self, parent, recarray):
        wx.Panel.__init__(self, parent)

        self.recarray = recarray

        #sizer = wx.BoxSizer(wx.VERTICAL)

        self.grid = RecarrayTableGrid(self, recarray)

        self.grid.SetSize((10,10))

        #sizer.Add(self.grid, 1, wx.EXPAND, 0)

        #self.SetSizerAndFit(sizer)
        #self.SetAutoLayout(True)

        wx.EVT_SIZE(self, self.OnSize)

    def OnSize(self, event):
        self.grid.SetSize(self.GetClientSize())
        #self.Refresh()
        event.Skip()