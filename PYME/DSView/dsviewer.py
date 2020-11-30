#!/usr/bin/python
##################
# dsviewer.py
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
import os
os.environ['ETS_TOOLKIT'] = 'wx'

import wx
import wx.lib.agw.aui as aui
#import PYME.misc.aui as aui
#import sys
import matplotlib
matplotlib.use('WxAgg')

# import pylab
# pylab.ion()
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from . import modules

from PYME.DSView import splashScreen

try:
   import PYMEnf.DSView.modules
except ImportError:
    pass

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR) #clobber unhelpful matplotlib debug messages
logging.getLogger('matplotlib.backends.backend_wx').setLevel(logging.ERROR)

#import PYME.ui.autoFoldPanel as afp

from PYME.DSView.displayOptions import DisplayOpts
from PYME.DSView.DisplayOptionsPanel import OptionsPanel
#from PYME.DSView.OverlaysPanel import OverlayPanel
from PYME.IO.image import ImageStack

from PYME.ui.mytimer import mytimer
from PYME.Analysis import piecewiseMapping

from PYME.ui.AUIFrame import AUIFrame

import weakref
openViewers = weakref.WeakValueDictionary()

class dt(wx.FileDropTarget):
    def OnDropFiles(self, x, y, filenames):
        print(filenames)
        
        for filename in filenames:
            im = ImageStack(filename=filename)
            ViewIm3D(im)
            
#drop = dt()
                   

class DSViewFrame(AUIFrame):
    def __init__(self, image,  parent=None, title='', mode='LM', 
                 size = (800,700), glCanvas=None):
        AUIFrame.__init__(self,parent, -1, title,size=size, pos=wx.DefaultPosition)

        self.mode = mode
        self.glCanvas = glCanvas
        
        self.updateHooks = []
        self.statusHooks = []
        self.installedModules = []
        
        # will store weakrefs to things that modules previously injected into our namespace
        #self._module_injections = weakref.WeakValueDictionary()
        
        self.dataChangeHooks = []

        self.updating = False

        if glCanvas:
            self.glCanvas.wantViewChangeNotification.add(self)

        

        self.timer = mytimer()
        self.timer.Start(10000)

        self.image = image
        #self.image = ImageStack(data = dstack, mdh = mdh, filename = filename, queueURI = queueURI, events = None)
        if not self.image.filename is None and title == '':
            self.SetTitle(self.image.filename)

        

        self.do = DisplayOpts(self.image.data)
        if self.image.data.shape[1] == 1:
            self.do.slice = self.do.SLICE_XZ
        self.do.Optimise()

        if self.image.mdh and 'ChannelNames' in self.image.mdh.getEntryNames():
            chan_names = self.image.mdh.getEntry('ChannelNames')
            if len(chan_names) == self.image.data.shape[3]:
                self.do.names = chan_names

        #self.vp = ArraySettingsAndViewPanel(self, self.image.data, wantUpdates=[self.update], mdh=self.image.mdh)
        #self.view = ArrayViewPanel(self, do=self.do)
        #self.AddPage(self.view, True, 'Data')
        #self._mgr.AddPane(self.vp, aui.AuiPaneInfo().
        #                  Name("Data").Caption("Data").Centre().CloseButton(False).CaptionVisible(False))

        

        self.mainFrame = weakref.ref(self)
        #self.do = self.vp.do
        
        
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
        self.Bind(wx.EVT_MENU, self.OnOpen, id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, self.OnSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.OnExport, id=wx.ID_SAVEAS)
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
        

		
        self.statusbar = self.CreateStatusBar(1, wx.STB_SIZEGRIP)

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
       

        for pn in self.panesToMinimise:
            self._mgr.MinimizePane(pn)
        #self._mgr.MinimizePane(pinfo2)
        self.Layout()

        if 'view' in dir(self):
            sc = np.floor(np.log2(1.0*self.view.Size[0]/self.do.ds.shape[0]))
            #print self.view.Size[0], self.do.ds.shape[0], sc
            self.do.SetScale(sc)
            self.view.Refresh()
        self.update()
        
        self.drop = dt()        
        self.SetDropTarget(self.drop)
        
        self.AddMenuItem('Save', 'To Cluster', self.OnSaveToCluster)
        
        
        
        openViewers[self.image.filename] = self
        
   

    def CreateModuleMenu(self):
        #self.modMenuIds = {}
        self.moduleNameByID = {}
        self.moduleMenuIDByName = {}
        self.mModules = wx.Menu()
        for mn in modules.allmodules():
            id = wx.NewId()
            self.mModules.AppendCheckItem(id, mn)
            self.moduleNameByID[id] = mn
            self.moduleMenuIDByName[mn] = id
            if mn in self.installedModules:
                self.mModules.Check(id, True)
                self.mModules.Enable(id, False)

            self.Bind(wx.EVT_MENU, self.OnToggleModule, id=id)

        self.menubar.Append(self.mModules, "&Modules")
        self._menus["&Modules"] = self.mModules
        

    def OnToggleModule(self, event):
        id = event.GetId()
        mn = self.moduleNameByID[id]
        #if self.mModules.IsChecked(id):
        
        self.LoadModule(mn)
         
        
    def LoadModule(self, moduleName):
        """Load a module with the given name and update GUI
        """
        
        modules.loadModule(moduleName, self)

        if moduleName in self.installedModules:
            id = self.moduleMenuIDByName[moduleName]
            self.mModules.Check(id, True)
            self.mModules.Enable(id, False)

        self.CreateFoldPanel()
        self._mgr.Update()
        

    def GetSelectedPage(self):
        nbs = self._mgr.GetNotebooks()
        currPage = nbs[0].GetCurrentPage()

        return currPage

    
    def create_overlay_panel(self):
        from PYME.DSView.OverlaysPanel import OverlayPanel
        if not 'overlaypanel' in dir(self):
            self.overlaypanel = OverlayPanel(self, self.view, self.image.mdh)
            self.overlaypanel.SetSize(self.overlaypanel.GetBestSize())
            pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(
                False).MinimizeButton(True).MinimizeMode(
                aui.AUI_MINIMIZE_CAPT_SMART | aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
            self._mgr.AddPane(self.overlaypanel, pinfo2)
        
            self.panesToMinimise.append(pinfo2)
    


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
        
    def OnSaveToCluster(self, event=None):
        from PYME.IO import clusterExport
        
        seriesName = clusterExport.suggest_cluster_filename(self.image)

        ted = wx.TextEntryDialog(None, 'Cluster filename:', 'Save file to cluster', seriesName)

        if ted.ShowModal() == wx.ID_OK:
            #pd = wx.ProgressDialog()
            clusterExport.ExportImageToCluster(self.image, ted.GetValue())

        ted.Destroy()

    def OnExport(self, event=None):
        bx = min(self.do.selection_begin_x, self.do.selection_end_x)
        ex = max(self.do.selection_begin_x, self.do.selection_end_x)
        by = min(self.do.selection_begin_y, self.do.selection_end_y)
        ey = max(self.do.selection_begin_y, self.do.selection_end_y)
        
        roi = [[bx, ex + 1],[by, ey + 1], [0, self.image.data.shape[2]]]
        
        self.image.Save(crop = True, roi=roi)

    def OnCrop(self):
        pass
        #View3D(self.image.data[])

    def OnCloseWindow(self, event):
        plt.close('all')
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
        del(self.image)
        
        AUIFrame._cleanup(self)

    def dsRefresh(self):
        #zp = self.vp.do.zp #save z -position
        self.do.SetDataStack(self.image.dataSource)
        #self.vp.do.zp = zp #restore z position
        self.elv.SetEventSource(self.image.dataSource.getEvents())
        self.elv.SetRange([0, self.image.dataSource.getNumSlices()])
        
        if b'ProtocolFocus' in self.elv.evKeyNames:
            self.zm = piecewiseMapping.GeneratePMFromEventList(self.elv.eventSource, self.mdh.getEntry('Camera.CycleTime'), self.mdh.getEntry('StartTime'), self.mdh.getEntry('Protocol.PiezoStartPos'))
            self.elv.SetCharts([('Focus [um]', self.zm, b'ProtocolFocus'),])

        self.update()

def OSXActivateKludge():
    """On OSX our main window doesn't show until you click on it's icon. Try
    to kludge around this ..."""
    import subprocess
    import os 
    subprocess.Popen(['osascript', '-e', """\
        tell application "System Events"
          set procName to name of first process whose unix id is %s
        end tell
        tell application procName to activate
    """ % os.getpid()])

class MyApp(wx.App):
    def __init__(self, argv, *args, **kwargs):
        self.argv = argv
        wx.App.__init__(self, *args, **kwargs)
        
    def OnInit(self):
        
        
        #self.sscreen = wx.Frame(None, size=(100,100))
        #self.SetTopWindow(self.sscreen)
        #self.sscreen.Show(1)
        
        self.splash = splashScreen.SplashScreen(None, None)
        self.splash.Show(1)
        
        wx.CallAfter(self.LoadData)
        
        return True
        
    def LoadData(self):
        import sys
        from optparse import OptionParser

        op = OptionParser(usage = 'usage: %s [options] [filename]' % sys.argv[0])

        op.add_option('-m', '--mode', dest='mode', help="mode (or personality), as defined in PYME/DSView/modules/__init__.py")
        op.add_option('-q', '--queueURI', dest='queueURI', help="the Pyro URI of the task queue - to avoid having to use the nameserver lookup")
        op.add_option('-t', '--test', dest='test', help="Show a test image", action="store_true", default=False)
        op.add_option( '--test3d', dest='test3d', help="Show a 3d test image", action="store_true", default=False)
        op.add_option('-d', '--metadata', dest='metadata', help="Load image with specified metadata file", default=None)
        op.add_option('-g', '--start-analysis', dest='start_analysis', action="store_true", help="Automatically start the analysis (where appropriate)", default=False)
        op.add_option('-r', '--recipe', dest='recipe_filename', help='Recipe to load', default=None)

        options, args = op.parse_args(self.argv)
        
        try:
            #md = None
            #if not options.metadata == '':
            #    md = options.metadata
            print('Loading data')
            if options.test:
                # import pylab
                im = ImageStack(np.random.randn(100,100))
            elif options.test3d:
                # import numpy as np
                from scipy import ndimage
                im = ImageStack(ndimage.gaussian_filter(np.random.rand(100,100,100, 2), [20, 20, 20, 0]))
            elif len (args) > 0:
                im = ImageStack(filename=args[0], queueURI=options.queueURI, mdh=options.metadata)
            else:
                im = ImageStack(queueURI=options.queueURI)
    
            if options.mode is None:
                mode = im.mode
            else:
                mode = options.mode
                print('Mode: %s' % mode)
    
            vframe = DSViewFrame(im, None, im.filename, mode = mode)
            
            #this is a bit of a hack - requires explicit knowledge of the LMAnalysis module here
            if options.start_analysis and 'LMAnalyser' in dir(vframe):
                logging.info('Automatically starting analysis')
                wx.CallLater(5,vframe.LMAnalyser.OnGo)

            #likewise for the recipe module - requires explicit knowledge of the recipe module here
            if (not options.recipe_filename is None) and 'recipes' in dir(vframe):
                logging.info('Loading recipe ... ')
                wx.CallLater(5, vframe.recipes.LoadRecipe, options.recipe_filename)
    
            self.SetTopWindow(vframe)
            vframe.Show(1)
            vframe.CenterOnScreen()
            #vframe.Raise()
            #vframe.TopLevel()
            #vframe.Show(1)
            #vframe.RequestUserAttention()
            
            if len(args) > 1:
                for fn in args[1:]:
                    im = ImageStack(filename=fn)
                    ViewIm3D(im)
        finally:
            self.splash.Destroy()
        
        if sys.platform == 'darwin':
            OSXActivateKludge()
        return 1
    


# end of class MyApp
import sys
def main(argv=sys.argv[1:]):
    app = MyApp(argv)
    print('Starting main loop')
    app.MainLoop()
    print('Finished main loop')


if __name__ == "__main__":
    main(sys.argv[1:])


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
