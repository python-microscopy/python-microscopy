#!/usr/bin/python

##################
# dsviewer_npy.py
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
from arrayViewPanel import ArraySettingsAndViewPanel

class DSViewFrame(wx.Frame):
    def __init__(self, parent=None, title='', dstack = None, log = None, filename = None, mdh = None, size = (400,300)):
        wx.Frame.__init__(self,parent, -1, title, size=size)

        self.ds = dstack
        self.log = log
        self.mdh = mdh

        self.saved = False
        self.vp = ArraySettingsAndViewPanel(self, self.ds)
        self.do = self.vp.do

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.vp, 1,wx.EXPAND,0)
        self.SetAutoLayout(1)
        self.SetSizer(sizer)
        sizer.Fit(self)

        # Menu Bar
        self.menubar = wx.MenuBar()
        self.SetMenuBar(self.menubar)
        tmp_menu = wx.Menu()
        F_EXPORT = wx.NewId()
        tmp_menu.Append(wx.ID_SAVEAS, "&Save As", "", wx.ITEM_NORMAL)
        tmp_menu.Append(F_EXPORT, "&Export Cropped", "", wx.ITEM_NORMAL)
        tmp_menu.Append(wx.ID_CLOSE, "Close", "", wx.ITEM_NORMAL)
        self.menubar.Append(tmp_menu, "File")

        mEdit = wx.Menu()
        EDIT_CLEAR_SEL = wx.NewId()
        mEdit.Append(EDIT_CLEAR_SEL, "Reset Selection", "", wx.ITEM_NORMAL)
        self.menubar.Append(mEdit, "Edit")

        # Menu Bar end
        wx.EVT_MENU(self, wx.ID_SAVEAS, self.OnSave)
        wx.EVT_MENU(self, F_EXPORT, self.OnExport)
        wx.EVT_MENU(self, wx.ID_CLOSE, self.menuClose)
        wx.EVT_MENU(self, EDIT_CLEAR_SEL, self.clearSel)
        wx.EVT_CLOSE(self, self.OnCloseWindow)
		
        self.statusbar = self.CreateStatusBar(1, wx.ST_SIZEGRIP)

        self.Layout()
        self.update()

    def update(self):
        #self.vp.update()
        self.statusbar.SetStatusText('Slice No: (%d/%d)  x: %d  y: %d' % (self.do.zp, self.do.ds.shape[2], self.do.xp, self.do.yp))

    def OnSave(self, event=None):
        import dataExporter

        if 'getEvents' in dir(self.ds):
            evts = self.ds.getEvents()
        else:
            evts = []

        fn = dataExporter.ExportData(self.vp.do.ds, self.mdh, evts)

        self.SetTitle(fn)

        self.saved = True

    def OnExport(self, event=None):
        import dataExporter

        if 'getEvents' in dir(self.ds):
            evts = self.ds.getEvents()
        else:
            evts = []

        dataExporter.CropExportData(self.vp.view, self.mdh, evts)

    def menuClose(self, event):
        self.Close()

    def OnCloseWindow(self, event):
        if (not self.saved):
            dialog = wx.MessageDialog(self, "Save data stack?", "pySMI", wx.YES_NO|wx.CANCEL)
            ans = dialog.ShowModal()
            if(ans == wx.ID_YES):
                self.OnSave()
                self.Destroy()
            elif (ans == wx.ID_NO):
                self.Destroy()
            else: #wxID_CANCEL:   
                if (not event.CanVeto()): 
                    self.Destroy()
                else:
                    event.Veto()
        else:
            self.Destroy()
			
    def clearSel(self, event):
        self.vp.ResetSelection()
        self.vp.Refresh()

def View3D(data, title='', mdh = None ):
    dvf = DSViewFrame(dstack = data, title=title, mdh=mdh, size=(500, 500))
    dvf.SetSize((500,500))
    dvf.Show()
    return dvf

