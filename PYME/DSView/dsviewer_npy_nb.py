#!/usr/bin/python
##################
# dsviewer_npy_nb.py
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
import wx.lib.agw.aui as aui
#import PYME.misc.aui as aui
import matplotlib
matplotlib.use('WxAgg')

import pylab
pylab.ion()
import modules

try:
   import PYMEnf.DSView.modules
except ImportError:
    pass

import PYME.misc.autoFoldPanel as afp
#from PYME.DSView.arrayViewPanel import ArraySettingsAndViewPanel
#from PYME.DSView.arrayViewPanel import ArrayViewPanel
from PYME.DSView.displayOptions import DisplayOpts
from PYME.DSView.DisplayOptionsPanel import OptionsPanel
#from PYME.DSView.OverlaysPanel import OverlayPanel
from PYME.DSView.image import ImageStack

from PYME.Acquire.mytimer import mytimer
from PYME.Analysis import piecewiseMapping

import weakref
openViewers = weakref.WeakValueDictionary()

class dt(wx.FileDropTarget):
    def OnDropFiles(self, x, y, filenames):
        print(filenames)
        
        for filename in filenames:
            im = ImageStack(filename=filename)
            ViewIm3D(im)
            
#drop = dt()
        

class DSViewFrame(wx.Frame):
    def __init__(self, image,  parent=None, title='', mode='LM', 
                 size = (800,700), glCanvas=None):
        wx.Frame.__init__(self,parent, -1, title,size=size, pos=(1100, 300))
        
        self.SetAutoLayout(True)

        self.mode = mode
        self.glCanvas = glCanvas
        self.paneHooks = []
        self.updateHooks = []
        self.statusHooks = []
        self.installedModules = []
        
        self.dataChangeHooks = []

        self.updating = False

        if glCanvas:
            self.glCanvas.wantViewChangeNotification.add(self)

        self.pane0 = None

        self.timer = mytimer()
        self.timer.Start(10000)

        self.image = image
        #self.image = ImageStack(data = dstack, mdh = mdh, filename = filename, queueURI = queueURI, events = None)
        if not self.image.filename == None and title == '':
            self.SetTitle(self.image.filename)

        self._mgr = aui.AuiManager(agwFlags = aui.AUI_MGR_DEFAULT | aui.AUI_MGR_AUTONB_NO_CAPTION)
        atabstyle = self._mgr.GetAutoNotebookStyle()
        self._mgr.SetAutoNotebookStyle((atabstyle ^ aui.AUI_NB_BOTTOM) | aui.AUI_NB_TOP)
        # tell AuiManager to manage this frame
        self._mgr.SetManagedWindow(self)

        self.do = DisplayOpts(self.image.data)
        if self.image.data.shape[1] == 1:
            self.do.slice = self.do.SLICE_XZ
        self.do.Optimise()

        if self.image.mdh and 'ChannelNames' in self.image.mdh.getEntryNames():
            self.do.names = self.image.mdh.getEntry('ChannelNames')

        #self.vp = ArraySettingsAndViewPanel(self, self.image.data, wantUpdates=[self.update], mdh=self.image.mdh)
        #self.view = ArrayViewPanel(self, do=self.do)
        #self.AddPage(self.view, True, 'Data')
        #self._mgr.AddPane(self.vp, aui.AuiPaneInfo().
        #                  Name("Data").Caption("Data").Centre().CloseButton(False).CaptionVisible(False))

        

        self.mainFrame = weakref.ref(self)
        #self.do = self.vp.do
        
        self._menus = {}
        # Menu Bar
        self.menubar = wx.MenuBar()
        self.SetMenuBar(self.menubar)
        tmp_menu = wx.Menu()
        tmp_menu.Append(wx.ID_OPEN, '&Open', "", wx.ITEM_NORMAL)
        tmp_menu.Append(wx.ID_SAVE, "&Save As", "", wx.ITEM_NORMAL)
        tmp_menu.Append(wx.ID_SAVEAS, "&Export Cropped", "", wx.ITEM_NORMAL)

        #a submenu for modules to hook and install saving functions into
        self.save_menu = wx.Menu()
        self._menus['Save'] = self.save_menu
        tmp_menu.AppendMenu(-1, 'Save &Results', self.save_menu)
        
        tmp_menu.AppendSeparator()
        tmp_menu.Append(wx.ID_CLOSE, "Close", "", wx.ITEM_NORMAL)
        self.menubar.Append(tmp_menu, "File")

        self.view_menu = wx.Menu()
        self.menubar.Append(self.view_menu, "&View")
        self._menus['View'] = self.view_menu

        #'extras' menu for modules to install stuff into
        self.mProcessing = wx.Menu()
        self.menubar.Append(self.mProcessing, "&Processing")
        self._menus['Processing'] = self.mProcessing

        # Menu Bar end
        wx.EVT_MENU(self, wx.ID_OPEN, self.OnOpen)
        wx.EVT_MENU(self, wx.ID_SAVE, self.OnSave)
        wx.EVT_MENU(self, wx.ID_SAVEAS, self.OnExport)
        wx.EVT_CLOSE(self, self.OnCloseWindow)
        wx.EVT_SIZE(self, self.OnSize)

		
        self.statusbar = self.CreateStatusBar(1, wx.ST_SIZEGRIP)

        self.panesToMinimise = []

        modules.loadMode(self.mode, self)
        self.CreateModuleMenu()

        

        self.optionspanel = OptionsPanel(self, self.do, thresholdControls=True)
        self.optionspanel.SetSize(self.optionspanel.GetBestSize())
        pinfo = aui.AuiPaneInfo().Name("optionsPanel").Right().Caption('Display Settings').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        self._mgr.AddPane(self.optionspanel, pinfo)

        self.panesToMinimise.append(pinfo)

        self._mgr.AddPane(self.optionspanel.CreateToolBar(self), aui.AuiPaneInfo().Name("ViewTools").Caption("View Tools").CloseButton(False).
                      ToolbarPane().Right().GripperTop())

        if self.do.ds.shape[2] > 1:
            from PYME.DSView.modules import playback
            self.playbackpanel = playback.PlayPanel(self, self)
            self.playbackpanel.SetSize(self.playbackpanel.GetBestSize())

            pinfo1 = aui.AuiPaneInfo().Name("playbackPanel").Bottom().Caption('Playback').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
            self._mgr.AddPane(self.playbackpanel, pinfo1)
            self.do.WantChangeNotification.append(self.playbackpanel.update)

        #self.mWindows =  wx.Menu()
        #self.menubar.append(self.mWindows, '&Composite With')
        self.do.WantChangeNotification.append(self.update)

        self.CreateFoldPanel()
        self._mgr.Update()

        for pn in self.panesToMinimise:
            self._mgr.MinimizePane(pn)
        #self._mgr.MinimizePane(pinfo2)
        self.Layout()

        if 'view' in dir(self):
            sc = pylab.floor(pylab.log2(1.0*self.view.Size[0]/self.do.ds.shape[0]))
            #print self.view.Size[0], self.do.ds.shape[0], sc
            self.do.SetScale(sc)
            self.view.Refresh()
        self.update()
        
        self.drop = dt()        
        self.SetDropTarget(self.drop)
        
        
        
        openViewers[self.image.filename] = self
        
    def OnSize(self, event):
        #self.Layout()
        self._mgr.Update()
        #self.Refresh()
        #self.Update()

    def AddPage(self, page=None, select=True,caption='Dummy'):
        if self.pane0 == None:
            name = caption.replace(' ', '')
            self._mgr.AddPane(page, aui.AuiPaneInfo().
                          Name(name).Caption(caption).Centre().CloseButton(False).CaptionVisible(False))
            self.pane0 = name
        else:
            self._mgr.Update()
            pn = self._mgr.GetPaneByName(self.pane0)
            if pn.IsNotebookPage():
                print((pn.notebook_id))
                nbs = self._mgr.GetNotebooks()
                if len(nbs) > pn.notebook_id:
                    currPage = nbs[pn.notebook_id].GetSelection()
                self._mgr.AddPane(page, aui.AuiPaneInfo().
                              Name(caption.replace(' ', '')).Caption(caption).CloseButton(False).NotebookPage(pn.notebook_id))
                if (not select) and len(nbs) > pn.notebook_id:
                    self._mgr.Update()
                    nbs[pn.notebook_id].SetSelection(currPage)
            else:
                self._mgr.AddPane(page, aui.AuiPaneInfo().
                              Name(caption.replace(' ', '')).Caption(caption).CloseButton(False), target=pn)
                
                
                if not select:
                    self._mgr.Update()
                    nb = self._mgr.GetNotebooks()[0]
                    nb.SetSelection(0)

               
        wx.CallAfter(self._mgr.Update)
        #self.Layout() 
        #self.OnSize(None)
        #self.OnSize(None)
        

    def CreateModuleMenu(self):
        self.modMenuIds = {}
        self.mModules = wx.Menu()
        for mn in modules.allmodules():
            id = wx.NewId()
            self.mModules.AppendCheckItem(id, mn)
            self.modMenuIds[id] = mn
            if mn in self.installedModules:
                self.mModules.Check(id, True)

            wx.EVT_MENU(self, id, self.OnToggleModule)
            
        self.menubar.Append(self.mModules, "&Modules")
        
    def AddMenuItem(self, menuName, itemName='', itemCallback = None, itemType='normal', helpText = ''):   
        mItem = None
        if not menuName in self._menus.keys():
            menu = wx.Menu()
            self.menubar.Insert(self.menubar.GetMenuCount()-1, menu, menuName)
            self._menus[menuName] = menu
        else:
            menu = self._menus[menuName]
        
        if itemType == 'normal':        
            mItem = menu.Append(wx.ID_ANY, itemName, helpText, wx.ITEM_NORMAL)
            self.Bind(wx.EVT_MENU, itemCallback, mItem)
        elif itemType == 'separator':
            menu.AppendSeparator()
            
        return mItem

    def OnToggleModule(self, event):
        id = event.GetId()
        mn = self.modMenuIds[id]
        if self.mModules.IsChecked(id):
            modules.loadModule(mn, self)

        if mn in self.installedModules:
            self.mModules.Check(id, True)

        self.CreateFoldPanel()
        self._mgr.Update()

    def GetSelectedPage(self):
        nbs = self._mgr.GetNotebooks()
        currPage = nbs[0].GetCurrentPage()

        return currPage

    



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
        self.Update()


    def update(self):
        if not self.updating:
            self.updating = True
            #if 'view' in dir(self):
            #    self.view.Refresh()
            statusText = 'Slice No: (%d/%d)    x: %d    y: %d' % (self.do.zp, self.do.ds.shape[2], self.do.xp, self.do.yp)
            #grab status from modules which supply it
            for sCallback in self.statusHooks:
                statusText += '\t' + sCallback() #'Frames Analysed: %d    Events detected: %d' % (self.vp.do.zp, self.vp.do.ds.shape[2], self.vp.do.xp, self.vp.do.yp, self.LMAnalyser.numAnalysed, self.LMAnalyser.numEvents)
            self.statusbar.SetStatusText(statusText)

            #if 'playbackpanel' in dir(self):
            #    self.playbackpanel.update()

            #update any modules which require it
            for uCallback in self.updateHooks:
                #print uCallback
                uCallback(self)

            self.updating = False
            
    #def Redraw(self):
    #    self.v
            
    def DataChanged(self):
        for uCallback in self.dataChangeHooks:
            uCallback(self)

    def OnOpen(self, event=None):
        ViewIm3D(ImageStack())
        

    def OnSave(self, event=None):
        self.image.Save()
        self.SetTitle(self.image.filename)

    def OnExport(self, event=None):
        self.image.Save(crop = True, view = self.view)

    def OnCrop(self):
        pass
        #View3D(self.image.data[])

    def OnCloseWindow(self, event):
        pylab.close('all')
        if (not self.image.saved):
            dialog = wx.MessageDialog(self, "Save data stack?", "PYME", wx.YES_NO|wx.CANCEL)
            ans = dialog.ShowModal()
            if(ans == wx.ID_YES):
                self.OnSave()
                self._cleanup()
            elif (ans == wx.ID_NO):
                self._cleanup()
            else: #wxID_CANCEL:   
                if (not event.CanVeto()):
                    self._cleanup()
                else:
                    event.Veto()
        else:
            self._cleanup()

    def _cleanup(self):
        self.timer.Stop()
        #for some reason AUI doesn't clean itself up properly and stops the
        #window from being garbage collected - fix this here
        self._mgr.UnInit()
        self._mgr._frame = None
        #if self.glCanvas:
        #    self.glCanvas.wantViewChangeNotification.remove(self)
        self.Destroy()

    def dsRefresh(self):
        #zp = self.vp.do.zp #save z -position
        self.do.SetDataStack(self.image.dataSource)
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
        op.add_option('-t', '--test', dest='test', help="Show a test image", action="store_true", default=False)

        options, args = op.parse_args()
        
        if options.test:
            import pylab
            im = ImageStack(pylab.randn(100,100))
        elif len (args) > 0:
            im = ImageStack(filename=args[0], queueURI=options.queueURI)
        else:
            im = ImageStack(queueURI=options.queueURI)

        if options.mode == None:
            mode = im.mode
        else:
            mode = options.mode

        vframe = DSViewFrame(im, None, im.filename, mode = mode)

        self.SetTopWindow(vframe)
        vframe.Show(1)
        
        if len(args) > 1:
            for fn in args[1:]:
                im = ImageStack(filename=fn)
                ViewIm3D(im)

        return 1

# end of class MyApp

def main():
    app = MyApp(0)
    app.MainLoop()


if __name__ == "__main__":
    main()


def View3D(data, titleStub='Untitled Image', mdh = None, mode='lite', 
           parent=None, glCanvas=None):
    im = ImageStack(data = data, mdh = mdh, titleStub=titleStub)
    dvf = DSViewFrame(im, mode=mode, size=(500, 500), 
                      parent=parent, glCanvas=glCanvas)
    dvf.SetSize((500,500))
    dvf.Show()
    return dvf

def ViewIm3D(image, title='', mode='lite', parent=None, glCanvas=None):
    dvf = DSViewFrame(image, title=title, mode=mode, size=(500, 500), 
                      parent=parent, glCanvas=glCanvas)
    dvf.SetSize((500,500))
    dvf.Show()
    return dvf
