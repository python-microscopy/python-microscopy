#!/usr/bin/python

##################
# MetadataTree.py
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

import  wx
import  wx.gizmos as gizmos

from PYME.IO.MetaDataHandler import NestedClassMDHandler

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
from bisect import bisect


class TextEditMixin:
    """
    A mixin class that enables any text in any column of a
    multi-column listctrl to be edited by clicking on the given row
    and column.  You close the text editor by hitting the ENTER key or
    clicking somewhere else on the listctrl. You switch to the next
    column by hiting TAB.

    To use the mixin you have to include it in the class definition
    and call the __init__ function::

        class TestListCtrl(wx.ListCtrl, TextEditMixin):
            def __init__(self, parent, ID, pos=wx.DefaultPosition,
                         size=wx.DefaultSize, style=0):
                wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
                TextEditMixin.__init__(self)


    Authors:     Steve Zatz, Pim Van Heuven (pim@think-wize.com)
    """

    editorBgColour = wx.Colour(255,255,175) # Yellow
    editorFgColour = wx.Colour(0,0,0)       # black

    def __init__(self):
        #editor = wx.TextCtrl(self, -1, pos=(-1,-1), size=(-1,-1),
        #                     style=wx.TE_PROCESS_ENTER|wx.TE_PROCESS_TAB \
        #                     |wx.TE_RICH2)

        self.make_editor()
        self.Bind(wx.EVT_TEXT_ENTER, self.CloseEditor)
        self.GetMainWindow().Bind(wx.EVT_RIGHT_DOWN, self.OnLeftDown)
        self.GetMainWindow().Bind(wx.EVT_LEFT_DCLICK, self.OnLeftDown)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnItemSelected)

        self.editableColumns = []

    def makeColumnEditable(self, col):
        self.editableColumns.append(col)


    def make_editor(self, col_style=wx.LIST_FORMAT_LEFT):

        style =wx.TE_PROCESS_ENTER|wx.TE_PROCESS_TAB|wx.TE_RICH2
        style |= {wx.LIST_FORMAT_LEFT: wx.TE_LEFT,
                  wx.LIST_FORMAT_RIGHT: wx.TE_RIGHT,
                  wx.LIST_FORMAT_CENTRE : wx.TE_CENTRE
                  }[col_style]

        editor = wx.TextCtrl(self, -1, style=style)
        editor.SetBackgroundColour(self.editorBgColour)
        editor.SetForegroundColour(self.editorFgColour)
        font = self.GetFont()
        editor.SetFont(font)

        self.curRow = 0
        self.curCol = 0

        editor.Hide()
        if hasattr(self, 'editor'):
            self.editor.Destroy()
        self.editor = editor

        self.col_style = col_style
        self.editor.Bind(wx.EVT_CHAR, self.OnChar)
        self.editor.Bind(wx.EVT_KILL_FOCUS, self.CloseEditor)


    def OnItemSelected(self, evt):
        self.curRow = evt.GetIndex()
        evt.Skip()


    def OnChar(self, event):
        """ Catch the TAB, Shift-TAB, cursor DOWN/UP key code
            so we can open the editor at the next column (if any)."""

        keycode = event.GetKeyCode()
        if keycode == wx.WXK_TAB and event.ShiftDown():
            self.CloseEditor()
            if self.curCol-1 >= 0:
                self.OpenEditor(self.curCol-1, self.curRow)

        elif keycode == wx.WXK_TAB:
            self.CloseEditor()
            if self.curCol+1 < self.GetColumnCount():
                self.OpenEditor(self.curCol+1, self.curRow)

        elif keycode == wx.WXK_ESCAPE:
            self.CloseEditor()

        elif keycode == wx.WXK_DOWN:
            self.CloseEditor()
            if self.curRow+1 < self.GetItemCount():
                self._SelectIndex(self.curRow+1)
                self.OpenEditor(self.curCol, self.curRow)

        elif keycode == wx.WXK_UP:
            self.CloseEditor()
            if self.curRow > 0:
                self._SelectIndex(self.curRow-1)
                self.OpenEditor(self.curCol, self.curRow)

        else:
            event.Skip()


    def OnLeftDown(self, evt=None):
        """ Examine the click and double
        click events to see if a row has been click on twice. If so,
        determine the current row and columnn and open the editor."""

        if self.editor.IsShown():
            self.CloseEditor()

        x,y = evt.GetPosition()
        #print x,y
        item, flags, col = self.HitTest((x,y))

        print((item, flags, col))

#        if row != self.curRow: # self.curRow keeps track of the current row
#            evt.Skip()
#            return

        # the following should really be done in the mixin's init but
        # the wx.ListCtrl demo creates the columns after creating the
        # ListCtrl (generally not a good idea) on the other hand,
        # doing this here handles adjustable column widths

#        self.col_locs = [0]
#        loc = 0
#        for n in range(self.GetColumnCount()):
#            loc = loc + self.GetColumnWidth(n)
#            self.col_locs.append(loc)
#
#
#        col = bisect(self.col_locs, x+self.GetScrollPos(wx.HORIZONTAL)) - 1
        #row = item
        if col in self.editableColumns:
            self.OpenEditor(col, item)


    def OpenEditor(self, col, row):
        """ Opens an editor at the current position. """

        # give the derived class a chance to Allow/Veto this edit.
        evt = wx.ListEvent(wx.wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, self.GetId())
        evt.m_itemIndex = row
        evt.m_col = col
        item = self.GetItem(row, col)
        evt.m_item.SetId(item.GetId())
        evt.m_item.SetColumn(item.GetColumn())
        evt.m_item.SetData(item.GetData())
        evt.m_item.SetText(item.GetText())
        ret = self.GetEventHandler().ProcessEvent(evt)
        if ret and not evt.IsAllowed():
            return   # user code doesn't allow the edit.

        if self.GetColumn(col).m_format != self.col_style:
            self.make_editor(self.GetColumn(col).m_format)

        x0 = self.col_locs[col]
        x1 = self.col_locs[col+1] - x0

        scrolloffset = self.GetScrollPos(wx.HORIZONTAL)

        # scroll forward
        if x0+x1-scrolloffset > self.GetSize()[0]:
            if wx.Platform == "__WXMSW__":
                # don't start scrolling unless we really need to
                offset = x0+x1-self.GetSize()[0]-scrolloffset
                # scroll a bit more than what is minimum required
                # so we don't have to scroll everytime the user presses TAB
                # which is very tireing to the eye
                addoffset = self.GetSize()[0]/4
                # but be careful at the end of the list
                if addoffset + scrolloffset < self.GetSize()[0]:
                    offset += addoffset

                self.ScrollList(offset, 0)
                scrolloffset = self.GetScrollPos(wx.HORIZONTAL)
            else:
                # Since we can not programmatically scroll the ListCtrl
                # close the editor so the user can scroll and open the editor
                # again
                self.editor.SetValue(self.GetItem(row, col).GetText())
                self.curRow = row
                self.curCol = col
                self.CloseEditor()
                return

        y0 = self.GetItemRect(row)[1]

        editor = self.editor
        editor.SetDimensions(x0-scrolloffset,y0, x1,-1)

        editor.SetValue(self.GetItem(row, col).GetText())
        editor.Show()
        editor.Raise()
        editor.SetSelection(-1,-1)
        editor.SetFocus()

        self.curRow = row
        self.curCol = col


    # FIXME: this function is usually called twice - second time because
    # it is binded to wx.EVT_KILL_FOCUS. Can it be avoided? (MW)
    def CloseEditor(self, evt=None):
        """ Close the editor and save the new value to the ListCtrl. """
        if not self.editor.IsShown():
            return
        text = self.editor.GetValue()
        self.editor.Hide()
        self.SetFocus()

        # post wxEVT_COMMAND_LIST_END_LABEL_EDIT
        # Event can be vetoed. It doesn't has SetEditCanceled(), what would
        # require passing extra argument to CloseEditor()
        evt = wx.ListEvent(wx.EVT_LIST_END_LABEL_EDIT.evtType[0], self.GetId())
        evt.m_itemIndex = self.curRow
        evt.m_col = self.curCol
        item = self.GetItem(self.curRow, self.curCol)
        evt.m_item.SetId(item.GetId())
        evt.m_item.SetColumn(item.GetColumn())
        evt.m_item.SetData(item.GetData())
        evt.m_item.SetText(text) #should be empty string if editor was canceled
        ret = self.GetEventHandler().ProcessEvent(evt)
        if not ret or evt.IsAllowed():
            if self.IsVirtual():
                # replace by whather you use to populate the virtual ListCtrl
                # data source
                self.SetVirtualData(self.curRow, self.curCol, text)
            else:
                self.SetStringItem(self.curRow, self.curCol, text)
        self.RefreshItem(self.curRow)

    def _SelectIndex(self, row):
        listlen = self.GetItemCount()
        if row < 0 and not listlen:
            return
        if row > (listlen-1):
            row = listlen -1

        self.SetItemState(self.curRow, ~wx.LIST_STATE_SELECTED,
                          wx.LIST_STATE_SELECTED)
        self.EnsureVisible(row)
        self.SetItemState(row, wx.LIST_STATE_SELECTED,
                          wx.LIST_STATE_SELECTED)



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


class EditableTreeList(gizmos.TreeListCtrl, TextEditMixin):
    def __init__(self, parent, id=-1, style=wx.TR_DEFAULT_STYLE):
        print(style)
        gizmos.TreeListCtrl.__init__(self,parent, id, style=style)
        TextEditMixin.__init__(self)

class MyTreeListCtrl(gizmos.TreeListCtrl):
    def DoGetBestSize(self):
        return wx.Size(400, 200)


class MetadataPanel(wx.Panel):
    def __init__(self, parent, mdh, editable=True, refreshable=True):
        self.mdh=mdh
        wx.Panel.__init__(self, parent, -1)
        #self.Bind(wx.EVT_SIZE, self.OnSize)

        sizer1 = wx.BoxSizer(wx.VERTICAL)

        self.tree = MyTreeListCtrl(self, -1, style =
                                        wx.TR_DEFAULT_STYLE
                                        #| wx.TR_HAS_BUTTONS
                                        #| wx.TR_TWIST_BUTTONS
                                        #| wx.TR_ROW_LINES
                                        | wx.TR_EDIT_LABELS
                                        #| wx.TR_COLUMN_LINES
                                        #| wx.TR_NO_LINES
                                        | wx.TR_FULL_ROW_HIGHLIGHT
                                   )

        
        # create some columns
        self.tree.AddColumn("Entry")
        self.tree.AddColumn("Value")
        
        self.tree.SetMainColumn(0) # the one with the tree in it...
        self.tree.SetColumnWidth(0, 300)
        self.tree.SetColumnWidth(1, 300)


        self.root = self.tree.AddRoot("Metadata")
        self.tree.SetItemText(self.root, "root", 0)

        self.paths = {}

        nmdh = NestedClassMDHandler(mdh)
        self.addEntries(nmdh, self.root)

        if editable:
            self.editableCols = [1]
        else:
            self.editableCols = []

        #entryNames = self.mdh.getEntryNames()
        
#        for k in nmdh.__dict__.keys():
#            #txt = "Item %d" % x
#            child = self.tree.AppendItem(self.root, k)
#            self.tree.SetItemText(child, txt + "(c1)", 1)
                                                

        if wx.__version__ > '4':
            self.tree.ExpandAll()#self.root)
        else:
            self.tree.ExpandAll(self.root)

        #self.tree.GetMainWindow().Bind(wx.EVT_LEFT_DOWN, self.OnRightDown)
        self.tree.GetMainWindow().Bind(wx.EVT_LEFT_UP, self.OnRightUp)
        self.tree.Bind(wx.EVT_TREE_END_LABEL_EDIT, self.OnEndEdit)
        self.tree.Bind(wx.EVT_TREE_BEGIN_LABEL_EDIT, self.OnBeginEdit)
        #self.tree.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.OnActivate)

        sizer1.Add(self.tree, 1, wx.EXPAND, 0)

        if refreshable:
            bRefresh = wx.Button(self, -1, 'Refresh')
            bRefresh.Bind(wx.EVT_BUTTON, self.rebuild)

            sizer1.Add(bRefresh, 0, wx.ALL|wx.ALIGN_RIGHT, 5)

        self.SetSizerAndFit(sizer1)

    def addEntries(self, mdh, node, entrypath=''):
        #en = []
        for k in sorted(mdh.__dict__.keys()):
            child = self.tree.AppendItem(node, k)
            self.tree.SetItemText(child, k, 0)
            if isinstance(mdh.__dict__[k], NestedClassMDHandler):
                self.addEntries(mdh.__dict__[k], child, '.'.join((entrypath, k)))
            else:
                self.tree.SetItemText(child, str(mdh.getEntry(k)), 1)
                self.paths[child] = '.'.join((entrypath, k))

    def rebuild(self, event=None):
        self.tree.DeleteRoot()

        self.root = self.tree.AddRoot("Metadata")
        self.tree.SetItemText(self.root, "root", 0)

        nmdh = NestedClassMDHandler(self.mdh)
        self.addEntries(nmdh, self.root)
        
        if wx.__version__ > '4':
            self.tree.ExpandAll()#self.root)
        else:
            self.tree.ExpandAll(self.root)


    #def OnActivate(self, evt):
        #self.log.write('OnActivate: %s' % self.tree.GetItemText(evt.GetItem()))


#    def OnRightDown(self, evt):
#        pos = evt.GetPosition()
#        item, flags, col = self.tree.HitTest(pos)
#        if item:
#            self.log.write('Flags: %s, Col:%s, Text: %s' %
#                           (flags, col, self.tree.GetItemText(item, col)))

    def OnRightDown(self, event):
        pt = event.GetPosition()
        item, flags, col = self.tree.HitTest(pt)
        if item:
#            self.log.WriteText("OnRightClick: %s, %s, %s\n" %
#                               (self.tree.GetItemText(item), type(item), item.__class__))
            self.tree.SelectItem(item)


    def OnRightUp(self, event):
        pt = event.GetPosition()
        item, flags, col = self.tree.HitTest(pt)
        #print item, flags, col
        if item and col in self.editableCols:
            #self.log.WriteText("OnRightUp: %s (manually starting label edit)\n"
            #                   % self.tree.GetItemText(item))
            self.tree.EditLabel(item, col)



    def OnBeginEdit(self, event):
        #self.log.WriteText("OnBeginEdit\n")
        # show how to prevent edit...
        item = event.GetItem()
        entryName = self.GetItemFullname(item)
        if not entryName in self.mdh.getEntryNames():
            event.Veto()


    def OnEndEdit(self, event):
#        self.log.WriteText("OnEndEdit: %s %s\n" %
#                           (event.IsEditCancelled(), event.GetLabel()) )
        # show how to reject edit, we'll not allow any digits
        if not event.IsEditCancelled():
            item = event.GetItem()
            newLabel = event.GetLabel()
            entryName = self.GetItemFullname(item)
            if entryName in self.mdh.getEntryNames():
                try:
                    ne = self.mdh.getEntry(entryName).__class__(newLabel)
                    #print ne
                    self.mdh.setEntry(entryName, ne)
                except:
                    event.Veto()
            #print event.GetLabel()

    def GetItemFullname(self, item):
        cp = item
        parents = []
        while not cp == self.root:
            parents.append(cp)
            cp = self.tree.GetItemParent(cp)

        names = [self.tree.GetItemText(p) for p in parents]
        return '.'.join(names[::-1])

    def OnSize(self, evt):
        self.tree.SetSize(self.GetSize())


