#!/usr/bin/python
##################
# dsviewer_npy_nb.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
import wx.lib.agw.aui as aui

import pylab
import modules

import PYME.misc.autoFoldPanel as afp
from PYME.DSView.arrayViewPanel import ArraySettingsAndViewPanel
from PYME.DSView.image import ImageStack

from PYME.Acquire.mytimer import mytimer
from PYME.Analysis import piecewiseMapping


class DSViewFrame(wx.Frame):
    def __init__(self, image,  parent=None, title='', mode='LM', size = (800,800)):
        wx.Frame.__init__(self,parent, -1, title,size=size, pos=(1100, 300))

        self.mode = mode
        self.paneHooks = []
        self.updateHooks = []
        self.statusHooks = []
        self.installedModules = []

        self.timer = mytimer()
        self.timer.Start(10000)

        self.image = image
        #self.image = ImageStack(data = dstack, mdh = mdh, filename = filename, queueURI = queueURI, events = None)
        if not self.image.filename == None:
            self.SetTitle(self.image.filename)

        self._mgr = aui.AuiManager(agwFlags = aui.AUI_MGR_DEFAULT | aui.AUI_MGR_AUTONB_NO_CAPTION)
        atabstyle = self._mgr.GetAutoNotebookStyle()
        self._mgr.SetAutoNotebookStyle((atabstyle ^ aui.AUI_NB_BOTTOM) | aui.AUI_NB_TOP)
        # tell AuiManager to manage this frame
        self._mgr.SetManagedWindow(self)

        self.vp = ArraySettingsAndViewPanel(self, self.image.data, wantUpdates=[self.update], mdh=self.image.mdh)
        self._mgr.AddPane(self.vp, aui.AuiPaneInfo().
                          Name("Data").Caption("Data").Centre().CloseButton(False).CaptionVisible(False))

        self.mainFrame = self
        
        # Menu Bar
        self.menubar = wx.MenuBar()
        self.SetMenuBar(self.menubar)
        tmp_menu = wx.Menu()
        tmp_menu.Append(wx.ID_SAVE, "&Save As", "", wx.ITEM_NORMAL)
        tmp_menu.Append(wx.ID_SAVEAS, "&Export Cropped", "", wx.ITEM_NORMAL)

        #a submenu for modules to hook and install saving functions into
        self.save_menu = wx.Menu()
        tmp_menu.AppendMenu(-1, 'Save &Results', self.save_menu)
        
        tmp_menu.Append(wx.ID_CLOSE, "Close", "", wx.ITEM_NORMAL)
        self.menubar.Append(tmp_menu, "File")

        #'extras' menu for modules to install stuff into
        self.mExtras = wx.Menu()
        self.menubar.Append(self.mExtras, "&Extras")

        # Menu Bar end
        wx.EVT_MENU(self, wx.ID_SAVE, self.OnSave)
        wx.EVT_MENU(self, wx.ID_SAVEAS, self.OnExport)
        wx.EVT_CLOSE(self, self.OnCloseWindow)

		
        self.statusbar = self.CreateStatusBar(1, wx.ST_SIZEGRIP)

        modules.loadMode(self.mode, self)
        self.CreateModuleMenu()

        self.CreateFoldPanel()
        self._mgr.Update()
        self.Layout()

        self.vp.Refresh()
        self.update()

    def AddPage(self, page=None, select=True,caption='Dummy'):
        self._mgr.Update()
        pn = self._mgr.GetPaneByName("Data")
        if pn.IsNotebookPage():
            print pn.notebook_id
            nbs = self._mgr.GetNotebooks()
            if len(nbs) > pn.notebook_id:
                currPage = nbs[pn.notebook_id].GetSelection()
            self._mgr.AddPane(page, aui.AuiPaneInfo().
                          Name(caption.replace(' ', '')).Caption(caption).CloseButton(False).NotebookPage(pn.notebook_id))
            if (not select) and len(nbs) > pn.notebook_id:
                nbs[pn.notebook_id].SetSelection(currPage)
        else:
            self._mgr.AddPane(page, aui.AuiPaneInfo().
                          Name(caption.replace(' ', '')).Caption(caption).CloseButton(False), target=pn)
            #nb = self._mgr.GetNotebooks()[0]
            #if not select:
            #    nb.SetSelection(0)

        self._mgr.Update()

    def CreateModuleMenu(self):
        self.modMenuIds = {}
        self.mModules = wx.Menu()
        for mn in modules.allmodules:
            id = wx.NewId()
            self.mModules.AppendCheckItem(id, mn)
            self.modMenuIds[id] = mn
            if mn in self.installedModules:
                self.mModules.Check(id, True)

            wx.EVT_MENU(self, id, self.OnToggleModule)
            
        self.menubar.Append(self.mModules, "&Modules")

    def OnToggleModule(self, event):
        id = event.GetId()
        mn = self.modMenuIds[id]
        if self.mModules.IsChecked(id):
            modules.loadModule(mn, self)

        if mn in self.installedModules:
            self.mModules.Check(id, True)

        self.CreateFoldPanel()
        self._mgr.Update()



    def CreateFoldPanel(self):
        pinfo = self._mgr.GetPaneByName('sidePanel')
        if pinfo.IsOk(): #we already have a sidepanel, clear
            self.sidePanel.Clear()
        else:
            self.sidePanel = afp.foldPanel(self, -1, wx.DefaultPosition,size = wx.Size(180, 1000))
            pinfo = aui.AuiPaneInfo().Name("sidePanel").Left().CloseButton(False).CaptionVisible(False)

            self._mgr.AddPane(self.sidePanel, pinfo)
            
        if len(self.paneHooks) > 0:
            pinfo.Show()

            for genFcn in self.paneHooks:
                genFcn(self.sidePanel)
        else:
            pinfo.Hide()
            

        self._mgr.Update()
        self.Refresh()


    def update(self):
        self.vp.update()
        statusText = 'Slice No: (%d/%d)    x: %d    y: %d' % (self.vp.do.zp, self.vp.do.ds.shape[2], self.vp.do.xp, self.vp.do.yp)
        #grab status from modules which supply it
        for sCallback in self.statusHooks:
            statusText += '\t' + sCallback() #'Frames Analysed: %d    Events detected: %d' % (self.vp.do.zp, self.vp.do.ds.shape[2], self.vp.do.xp, self.vp.do.yp, self.LMAnalyser.numAnalysed, self.LMAnalyser.numEvents)
        self.statusbar.SetStatusText(statusText)

        #update any modules which require it
        for uCallback in self.updateHooks:
            uCallback()
        

    def OnSave(self, event=None):
        self.image.Save()
        self.SetTitle(self.image.filename)

    def OnExport(self, event=None):
        self.image.Save(crop = True, view = self.vp.view)

    def OnCloseWindow(self, event):
        pylab.close('all')
        if (not self.image.saved):
            dialog = wx.MessageDialog(self, "Save data stack?", "PYME", wx.YES_NO|wx.CANCEL)
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

    def dsRefresh(self):
        #zp = self.vp.do.zp #save z -position
        self.vp.do.SetDataStack(self.image.dataSource)
        #self.vp.do.zp = zp #restore z position
        self.elv.SetEventSource(self.image.dataSource.getEvents())
        self.elv.SetRange([0, self.image.dataSource.getNumSlices()])
        
        if 'ProtocolFocus' in self.elv.evKeyNames:
            self.zm = piecewiseMapping.GeneratePMFromEventList(self.elv.eventSource, self.mdh.getEntry('Camera.CycleTime'), self.mdh.getEntry('StartTime'), self.mdh.getEntry('Protocol.PiezoStartPos'))
            self.elv.SetCharts([('Focus [um]', self.zm, 'ProtocolFocus'),])

        self.update()


class MyApp(wx.App):
    def OnInit(self):
        import sys
        from optparse import OptionParser

        op = OptionParser(usage = 'usage: %s [options] [filename]' % sys.argv[0])

        op.add_option('-m', '--mode', dest='mode', help="mode (or personality), as defined in PYME/DSView/modules/__init__.py")
        op.add_option('-q', '--queueURI', dest='queueURI', help="the Pyro URI of the task queue - to avoid having to use the nameserver lookup")

        options, args = op.parse_args()

        if len (args) > 0:
            im = ImageStack(filename=args[0], queueURI=options.queueURI)
        else:
            im = ImageStack(queueURI=options.queueURI)

#        #wx.InitAllImageHandlers()
#        if (len(args) == 2):
#            im = ImageStack(filename = args[1])
#            vframe = DSViewFrame(im, None, args[1], mode = im.mode)
#        elif (len(sys.argv) == 3):
#            im = ImageStack(filename = args[1], queueURI=args[2])
#            vframe = DSViewFrame(im, None, args[1], mode = im.mode)
#        else:
#            vframe = DSViewFrame(None, '')

        if options.mode == None:
            mode = im.mode
        else:
            mode = options.mode

        vframe = DSViewFrame(im, None, im.filename, mode = mode)

        self.SetTopWindow(vframe)
        vframe.Show(1)

        return 1

# end of class MyApp

def main():
    app = MyApp(0)
    app.MainLoop()


if __name__ == "__main__":
    main()


def View3D(data, title='', mdh = None, mode='lite', parent=None):
    im = ImageStack(data = data, mdh = mdh)
    dvf = DSViewFrame(im, title=title, mode=mode, size=(500, 500), parent=parent)
    dvf.SetSize((500,500))
    dvf.Show()
    return dvf

def ViewIm3D(image, title='', mode='lite'):
    dvf = DSViewFrame(image, title=title, mode=mode, size=(500, 500))
    dvf.SetSize((500,500))
    dvf.Show()
    return dvf
