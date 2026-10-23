#!/usr/bin/python
##################
# editList.py
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
#import PYME.contrib.listctrlMixins  as  listmix
from wx.lib.mixins import listctrl as listmix

class EditListCtrl(wx.ListCtrl,
                   listmix.ListCtrlAutoWidthMixin,
                   listmix.TextEditMixin):

    def __init__(self, parent, ID, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.LC_REPORT|wx.LC_SINGLE_SEL|wx.SUNKEN_BORDER):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)

        listmix.ListCtrlAutoWidthMixin.__init__(self)
        listmix.TextEditMixin.__init__(self)


class DictCtrlDialog(wx.Dialog):
    def __init__(self, parent=None, base_dict=dict(), title='', 
                 size=wx.DefaultSize, column_names=('keys', 'values')):
        import sys
        wx.Dialog.__init__(self, parent, title=title)
        
        self.edit_list = EditListCtrl(self, -1, size=size)

        self.edit_list.InsertColumn(0, column_names[0])
        self.edit_list.InsertColumn(1, column_names[1])
        self.edit_list.makeColumnEditable(0)
        self.edit_list.makeColumnEditable(1)

        self.base_dict = base_dict

        for key, value in self.base_dict.items():
            ind = self.edit_list.InsertStringItem(sys.maxsize, key)
            self.edit_list.SetStringItem(ind, 1, value)

        v_sizer = wx.BoxSizer(wx.VERTICAL)
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer.Add(self.edit_list, 0, wx.ALL, 5)
        v_sizer.Add(h_sizer, 0, wx.ALL, 5)

        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        h_sizer.Add(btn, 0, wx.ALL, 5)

        self.b_add = wx.Button(self, -1, 'Add')
        self.b_add.Bind(wx.EVT_BUTTON, self.OnItemAdd)
        h_sizer.Add(self.b_add, 0, wx.ALL, 5)

        v_sizer.Add(h_sizer, 0, wx.ALL, 5)

        # for wxMSW
        self.edit_list.Bind(wx.EVT_COMMAND_RIGHT_CLICK, self.OnRightClick)
        self.edit_list.Bind(wx.EVT_COMMAND_LEFT_CLICK, self.OnRightClick)
        # for wxGTK
        self.edit_list.Bind(wx.EVT_RIGHT_UP, self.OnRightClick)
        self.edit_list.Bind(wx.EVT_LEFT_UP, self.OnRightClick)

        self.SetSizerAndFit(v_sizer)

        self.ID_FILT_ADD = wx.NewId()
        self.ID_FILT_DELETE = wx.NewId()
        self.ID_FILT_EDIT = wx.NewId()

        self.Bind(wx.EVT_MENU, self.OnItemAdd, id=self.ID_FILT_ADD)
        self.Bind(wx.EVT_MENU, self.OnItemDelete, id=self.ID_FILT_DELETE)
        self.Bind(wx.EVT_MENU, self.OnItemEdit, id=self.ID_FILT_EDIT)
    
    def OnRightClick(self, event):
        x = event.GetX()
        y = event.GetY()

        item, flags = self.edit_list.HitTest((x, y))

        menu = wx.Menu()
        menu.Append(self.ID_FILT_ADD, "Add")

        if item != wx.NOT_FOUND and flags & wx.LIST_HITTEST_ONITEM:
            self.current_list_item = item
            self.edit_list.Select(item)

            menu.Append(self.ID_FILT_DELETE, "Delete")
            menu.Append(self.ID_FILT_EDIT, "Edit")

        # Popup the menu.  If an item is selected then its handler
        # will be called before PopupMenu returns.
        self.PopupMenu(menu)
        menu.Destroy()

    def OnItemSelected(self, event):
        self.current_list_item = event.GetIndex()
        event.Skip()

    def OnItemDeselected(self, event):
        self.current_list_item = None
        event.Skip()

    def OnItemAdd(self, event):
        import sys

        dlg = DictItemEditDialog(self)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            val = str(dlg.tVal.GetValue())
            key = str(dlg.tKey.GetValue())

            self.base_dict[key] = val

            ind = self.edit_list.InsertStringItem(sys.maxsize, key)
            self.edit_list.SetStringItem(ind, 1, val)

        dlg.Destroy()

    def OnItemDelete(self, event):
        it = self.edit_list.GetItem(self.current_list_item)
        self.edit_list.DeleteItem(self.current_list_item)
        self.base_dict.pop(it.GetText())

    def OnItemEdit(self, event):
        key = str(self.edit_list.GetItem(self.current_list_item).GetText())
        
        val = self.base_dict[key]
        
        dlg = DictItemEditDialog(self, key=key, val=val)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            val = str(dlg.tVal.GetValue())
            self.base_dict[key] = val
            self.edit_list.SetStringItem(self.current_list_item, 1, val)

        dlg.Destroy()

class DictItemEditDialog(wx.Dialog):
    def __init__(self, parent, key='', val=''):
        wx.Dialog.__init__(self, parent, title='Edit ...')

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(self, -1, 'Key:'), 0, 
                   wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tKey = wx.TextCtrl(self, -1, key, size=(140, -1))
        hsizer.Add(self.tKey, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        vsizer.Add(hsizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Value:'), 0, 
                   wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tVal = wx.TextCtrl(self, -1, val, size=(140, -1))
        hsizer.Add(self.tVal, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        vsizer.Add(hsizer, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        btSizer = wx.StdDialogButtonSizer()
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btSizer.AddButton(btn)
        btn = wx.Button(self, wx.ID_CANCEL)
        btSizer.AddButton(btn)
        btSizer.Realize()

        vsizer.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(vsizer)
        vsizer.Fit(self)
