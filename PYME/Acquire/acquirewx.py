#!/usr/bin/python

##################
# smimainframe.py
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
"""
This contains the bulk of the GUI code for the main window of PYMEAcquire.
"""
import logging
import os
import time

#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
import wx
import wx.lib.agw.aui as aui
import wx.py.shell

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import PYME.DSView.displaySettingsPanel as disppanel
from PYME.DSView import arrayViewPanel
from PYME.DSView import dsviewer as dsviewer

from PYME.ui import mytimer

from PYME.Acquire.ui import positionUI
from PYME.Acquire.ui import intsliders
from PYME.Acquire.ui import seqdialog
from PYME.Acquire.ui import selectCameraPanel
from PYME.Acquire.ui import splashScreen
from PYME.Acquire.ui import HDFSpoolFrame

from PYME.Acquire import microscope
from PYME.Acquire import protocol

from PYME.Acquire import acquire_server

from PYME.IO import MetaDataHandler
#from PYME.IO.FileUtils import nameUtils
import six


from PYME.ui.AUIFrame import AUIFrame

def create(parent, options = None):
    return PYMEMainFrame(parent, options)

class PYMEMainFrame(acquire_server.AcquireHTTPServer, AUIFrame):
    def _create_menu(self):     
        #self._menus = {}
        #self.menubar = wx.MenuBar()
        
        ##############
        #Create the menus
        
        #File menu
        self.AddMenuItem('File', '&Open Stack', self.OnFileOpenStack)
        self.AddMenuItem('File', itemType='separator')        
        self.AddMenuItem('File', 'Exit', self.OnFileExit, id = wx.ID_EXIT)
        
        #Camera menu
        self.AddMenuItem('Camera', 'Toggle ROI\tF8', self.OnMCamRoi)
        self.miCamBin = self.AddMenuItem('Camera', 'Turn binning on', self.OnMCamBin, itemType='check')
        self.AddMenuItem('Camera', 'Set pixel size', self.OnMCamSetPixelSize)
        
        #Acquire menu
        self.AddMenuItem('Acquire', 'Snapshot\tF5', self.OnMAquireOnePic)
              
        #self._init_utils()
        #self.SetMenuBar(self.menubar)
        #######################
        
        self.SetClientSize(wx.Size(1020, 800))
        

        self.statusBar1 = wx.StatusBar(id=wx.ID_ANY,
              name='statusBar1', parent=self, style=wx.STB_SIZEGRIP)
        self.SetStatusBar(self.statusBar1)



    def __init__(self, parent, options = None):
        AUIFrame.__init__(self, id=wx.ID_ANY, name='smiMainFrame',
              parent=parent, pos=wx.Point(20, 20), size=wx.Size(600, 800),
              style=wx.DEFAULT_FRAME_STYLE, title= getattr(options, 'window_title', 'PYME Acquire'))
        
        acquire_server.AcquireHTTPServer.__init__(self, options, port=options.port, evt_loop=mytimer)
        
        self._create_menu()
    
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)        
        

        # create list of UI panels to be added after init
        # Will be populated buy init script
        self.toolPanels = []
        self.camPanels = []
        self.aqPanels = []
        self.anPanels = []

        self.splash = splashScreen.SplashScreen(self, self.scope)
        self.splash.Show()
        

        self.sh = wx.py.shell.Shell(id=-1,
              parent=self, size=wx.Size(-1, -1), style=0, locals=self.__dict__,
              introText='PYMEAcquire - note that help, license, etc. below is for Python, not PYME\n\n')
        self.AddPage(self.sh, caption='Shell')

        self.CreateToolPanel(getattr(options, 'gui_mode', 'default'))

        self.SetSize((1030, 895))
        
        self.time1 = mytimer.mytimer()   
        self.time1.Start(50)
        
        wx.CallAfter(self.run)
        self.time1.WantNotification.append(self._check_init_done)

        self.time1.WantNotification.append(self.splash.Tick) 

    def run(self):
        import threading
        self._poll_thread = threading.Thread(target=self.main_loop)
        self._poll_thread.start()

        if self.options.server:
            # only start the server if requested
            self._server_thread = threading.Thread(target=self.serve_forever)
            self._server_thread.start()

            if self.options.ipy:
                # make this a separate config option as port is hard coded so can't run more than one
                # process with this option. Also probably not desirable if you just want progamatic
                # remote control (through REST API).
                from PYME.Acquire.webui import ipy
                ns = dict(scope=self.scope, server=self)
                print('namespace:', ns)
                ipy.launch_ipy_server_thread(user_ns=ns)
        

    def _check_init_done(self):
        if self.scope.initialized == True and self._check_init_done in self.time1.WantNotification:
            logger.debug('Backround initialization done')
            self.time1.WantNotification.remove(self._check_init_done)
            
            self.doPostInit()
            
            self.time1.Stop()
            self.time1.Start(500)
    
    def _refreshDataStack(self):
        try:
            if not (self.view.do.ds.data is self.scope.frameWrangler.currentFrame):
                self.view.SetDataStack(self.scope.frameWrangler.currentFrame)
        except ArithmeticError:
            pass
        
    
    def _add_live_view(self):
        """Gets called once during post-init to start pulling data from the
        camera
        
        """

        if self.scope.cam.GetPicHeight() > 1:
            try:
                self.view.SetDataStack(self.scope.frameWrangler.currentFrame)
            except AttributeError:
                self._view = arrayViewPanel.ArrayViewPanel(self, self.scope.frameWrangler.currentFrame, initial_overlays=[])
                self.view.crosshairs = False
                self.view.showScaleBar = False
                self.view.do.leftButtonAction = self.view.do.ACTION_SELECTION
                self.view.do.showSelection = True
                self.view.CenteringHandlers.append(self.scope.PanCamera)

                self.vsp = disppanel.dispSettingsPanel2(self, self.view, self.scope)


                self.time1.WantNotification.append(self.vsp.RefrData)
                self.time1.WantNotification.append(self._refreshDataStack)

                self.AddPage(page=self.view, select=True,caption='Preview')

                self.AddCamTool(self.vsp, 'Display')

            self.scope.frameWrangler.onFrameGroup.connect(self.view.Redraw)

        else:
            #1d data - use graph instead
            from PYME.ui import fastGraph
            if 'sp' in dir(self):
                    pass
            else:
                self.sp = fastGraph.SpecGraphPanel(self, self)

                self.AddPage(page=self.sp, select=True,caption='Preview')

            #self.scope.frameWrangler.WantFrameGroupNotification.append(self.sp.refr)
            self.scope.frameWrangler.onFrameGroup.connect(self.sp.refr)
            
        #self.scope.PACallbacks.append(self._refreshDataStack)


    def doPostInit(self):
        logger.debug('Starting post-init')
        
        for cm in self.postInit:
            logger.debug('Loading GUI component for %s' %cm.name)
            try:
                cm.run(self, self.scope)
            except Exception as e:
                logger.exception('Error whilst initializing %s GUI' % cm.name)

        logger.debug('Run all custom GUI init tasks')
            
        

        if self.scope.cam.CamReady():# and ('chaninfo' in self.scope.__dict__)):
            #self._start_polling_camera()
            self._add_live_view()
            

            self.int_sl = intsliders.IntegrationSliders(self, self.scope)
            self.AddCamTool(self.int_sl, 'Integration Time')

            if len(self.scope.cameras) > 1:
                self.pCamChoose = selectCameraPanel.CameraChooserPanel(self, self.scope)
                self.AddCamTool(self.pCamChoose, 'Camera Selection')


            

        
        
        self.time1.WantNotification.append(self.StatusBarUpdate)

        for t in self.camPanels:
            #print(t)
            self.AddCamTool(*t)
        
        if len(self.scope.positioning.keys()) > 0.5:
            self.pos_sl = positionUI.PositionPanel(self.scope, self, self.scope.joystick)
            self.time1.WantNotification.append(self.pos_sl.update)

            self.AddTool(self.pos_sl, 'Positioning')

            self.seq_d = seqdialog.seqPanel(self, self.scope)
            self.AddAqTool(self.seq_d, 'Z-Stack', pinned=False)
            #self.seq_d.Show()
        
        for t in self.toolPanels:
            #print(t)
            self.AddTool(*t)
            
        if self.scope.cam.CamReady():
            self.pan_spool = HDFSpoolFrame.PanSpool(self, self.scope)
            self.AddAqTool(self.pan_spool, 'Time/Blinking series', pinned=False, folded=False)
            
        for t in self.aqPanels:
            self.AddAqTool(*t)
            
        
        
        for t in self.anPanels:
            self.AddTool(*t, panel=self.aqPanel)

        #self.splash.Destroy()

        #self.time1.WantNotification.append(self.scope.actions.Tick)
        
        self.initDone = True
        self._mgr.Update()

        if 'pCamChoose' in dir(self):
            self.pCamChoose.OnCCamera(None)
            
        logger.debug('Finished post-init')
        
        self.Show()

        #fudge to get layout right
#        panes = self.notebook1.GetAuiManager().AllPanes
#
#        self.paneNames = []
#
#        for p in panes:
#            self.paneNames.append(p.name)

        #self.LoadPerspective()

    def _getPerspectiveFilename(self):
        #print __file__
        fname = os.path.join(os.path.split(__file__)[0], 'GUILayout.txt')
        return fname

    def SavePerspective(self):
        #mgr = self.notebook1.GetAuiManager()
        persp = self._mgr.SavePerspective()

        for i, p in enumerate(self.paneNames):
            persp = persp.replace(p, 'pane_%d' % i)
            
        f = open(self._getPerspectiveFilename(), 'w')
        f.write(persp)
        f.close()

    def LoadPerspective(self):
        pfname = self._getPerspectiveFilename()
        if os.path.exists(pfname):
            f = open(pfname)
            persp = f.read()
            f.close()

            for i, p in enumerate(self.paneNames):
                persp = persp.replace('pane_%d' % i, p)

            self._mgr.LoadPerspective(persp)
            self._mgr.Update()


    def CreateToolPanel(self, mode='default'):
        self.camPanel = afp.foldPanel(self, -1, wx.DefaultPosition,
                                     wx.Size(240,1000))

        cpinfo = aui.AuiPaneInfo().Name("camControls").Caption("Hardware").Layer(1).Right().CloseButton(False)
        #cpinfo.dock_proportion  = int(cpinfo.dock_proportion*1.6)
        
        self._mgr.AddPane(self.camPanel, cpinfo)
        
        self.toolPanel = self.camPanel
        
        # self.toolPanel = afp.foldPanel(self, -1, wx.DefaultPosition,
        #                              wx.Size(240,1000))
        #
        # if mode == 'compact':
        #     self._mgr.AddPane(self.toolPanel, aui.AuiPaneInfo().
        #                       Name("hardwareControls").Caption("Hardware").CloseButton(False).BestSize(240, 250), target=cpinfo)
        # else:
        #     self._mgr.AddPane(self.toolPanel, aui.AuiPaneInfo().
        #                   Name("hardwareControls").Caption("Hardware").Layer(2).Position(1).Right().CloseButton(False).BestSize(240, 250))


        self.aqPanel = afp.foldPanel(self, -1, wx.DefaultPosition,
                                     wx.Size(240,1000), single_active_pane=True)

        if mode == 'compact':
            self._mgr.AddPane(self.toolPanel, aui.AuiPaneInfo().
                              Name("aqControls").Caption("Acquisition").CloseButton(False), target=cpinfo)
        else:
            aqinfo = aui.AuiPaneInfo().Name("aqControls").Caption("Acquisition Tasks").Layer(2).Position(0).Right().CloseButton(False)
            self._mgr.AddPane(self.aqPanel, aqinfo)
            aqinfo.dock_proportion  = int(aqinfo.dock_proportion*1.3)

        
            # self.anPanel = afp.foldPanel(self, -1, wx.DefaultPosition,
            #                         wx.Size(240, 1000))
            #
            # self._mgr.AddPane(self.anPanel, aui.AuiPaneInfo().
            #               Name("anControls").Caption("Analysis").CloseButton(False), target=aqinfo)



    def AddTool(self, pane, title, pinned=True, folded=True, panel=None):
        """Adds a pane to the tools section of the GUI
        
        Parameters
        ----------
        pane : an instance of a wx.Window derived class
            The pane to add. Optionally, a list of panes
        title : string
            The caption for the panel.
        """
        
        if panel is None:
            panel = self.toolPanel
        
        if isinstance(pane, afp.foldingPane):
            pane.SetCaption(title).Pin(pinned).Fold(folded)
            pane.Reparent(panel)
            panel.AddPane(pane)
        else:
            # a normal wx.Panel / wx.Window
            #print(panel, title, pinned, folded)
            item = afp.foldingPane(panel, -1, caption=title, pinned=pinned, folded=folded)
            pane.Reparent(item)
            item.AddNewElement(pane, priority=1)
            panel.AddPane(item)
#        item = self.toolPanel.AddFoldPanel(title, collapsed=False, foldIcons=self.Images)
#        panel.Reparent(item)
#        self.toolPanel.AddFoldPanelWindow(item, panel, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        #wx.LayoutAlgorithm().LayoutWindow(self, self._leftWindow1)

    def AddCamTool(self, pane, title, pinned=True, folded=True):
        """Adds a pane to the Camera section of the GUI
        
        Parameters
        ----------
        pane : an instance of a wx.Window derived class
            The pane to add
        title : string
            The caption for the panel.
        """
        
        self.AddTool(pane, title, pinned=pinned, panel=self.camPanel, folded=folded)
        

    def AddAqTool(self, pane, title, pinned=True, folded=True):
        """Adds a pane to the Acquisition section of the GUI
        
        Parameters
        ----------
        pane : an instance of a wx.Window derived class
            The pane to add
        title : string
            The caption for the panel.
        """
        self.AddTool(pane, title, pinned=pinned, panel=self.aqPanel, folded=folded)
        

    def OnFileOpenStack(self, event):
        #self.dv = dsviewer.DSViewFrame(self)
        #self.dv.Show()
        im = dsviewer.ImageStack(haveGUI=True)
        dvf = dsviewer.DSViewFrame(im, parent=self, size=(500, 500))
        dvf.SetSize((500,500))
        dvf.Show()
        event.Skip()

    def OnFileExit(self, event):
        self.Close()
        #event.Skip()
        
    def StatusBarUpdate(self):
        self.statusBar1.SetStatusText(self.scope.genStatus())

    def OnMAquireStack(self, event):
        self.seq_d.Show()
        self.seq_d.Raise()
        #event.Skip()

    def OnMAquireOnePic(self, event):
        import numpy as np
        self.scope.frameWrangler.stop()
        ds2 = np.atleast_3d(self.scope.frameWrangler.currentFrame.reshape(self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight()).copy())


        #metadata handling
        mdh = MetaDataHandler.NestedClassMDHandler()
        mdh.setEntry('StartTime', time.time())
        mdh.setEntry('AcquisitionType', 'SingleImage')

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(mdh)

        im = dsviewer.ImageStack(data = ds2, mdh = mdh, titleStub='Unsaved Image')
        if not im.mode == 'graph':
            im.mode = 'lite'

        #print im.mode
        dvf = dsviewer.DSViewFrame(im, mode= im.mode, size=(500, 500))
        dvf.SetSize((500,500))
        dvf.Show()

        self.snapNum += 1

        self.scope.frameWrangler.Prepare(True)
        self.scope.frameWrangler.start()

        #event.Skip()

   

    def OnMCamBin(self, event):
        logging.debug('Stopping frame wrangler to set ROI')
        self.scope.frameWrangler.stop()
        
        binx, biny = self.scope.state['Camera.Binning']
        if binx > 1:
            #turn binning off
            self.scope.state['Camera.Binning'] = (1,1)
        else:
            self.scope.state['Camera.Binning'] = (2, 2)

        logging.debug('Preparing frame wrangler for new ROI size')

        self.scope.frameWrangler.Prepare()
        self.view.SetDataStack(self.scope.frameWrangler.currentFrame)

        logging.debug('Restarting frame wrangler with new ROI')
        self.scope.frameWrangler.start()
        self.view.Refresh()
        self.view.GetParent().Refresh()

    def OnMCamSetPixelSize(self, event):
        from PYME.Acquire.ui import voxelSizeDialog

        dlg = voxelSizeDialog.VoxelSizeDialog(self, self.scope)
        dlg.ShowModal()


    def OnMCamRoi(self, event):
        logging.debug('Stopping frame wrangler to set ROI')
        self.scope.frameWrangler.stop()
        
        if 'validROIS' in dir(self.scope.cam) and self.scope.cam.ROIsAreFixed():
            #special case for cameras with restricted ROIs - eg Neo
            #print('setting ROI')
            logging.debug('setting ROI')
            dlg = wx.SingleChoiceDialog(self, 'Please select the ROI size', 'Camera ROI', ['%dx%d at (%d, %d)' % roi for roi in self.scope.cam.validROIS])
            dlg.ShowModal()
            logging.debug('Dlg Shown')
            self.scope.cam.SetROIIndex(dlg.GetSelection())
            dlg.Destroy()
            
            x1 = 0
            y1 = 0
            x2 = self.scope.cam.GetPicWidth()
            y2 = self.scope.cam.GetPicHeight()
            logging.debug('ROI Set')
        
        else:
            if (self.roi_on):
                x1, y1, x2, y2 = self.scope.state['Camera.ROI']

                self.scope.state['Camera.ROI'] = (0,0, self.scope.cam.GetCCDWidth(), self.scope.cam.GetCCDHeight())
                self.roi_on = False
            else:
                x1, y1, x2, y2 = self.view.do.GetSliceSelection()
    
                #if we're splitting colours/focal planes across the ccd, then only allow symetric ROIs
                if 'splitting' in dir(self.scope.cam):
                    if self.scope.cam.splitting.lower() == 'left_right':
                        x1 = min(x1, self.scope.cam.GetCCDWidth() - x2)
                        x2 = max(x2, self.scope.cam.GetCCDWidth() - x1)

                        if not self.scope.cam.splitterFlip:
                            x1 = 0
                            x2 = self.scope.cam.GetCCDWidth()

                    if self.scope.cam.splitting.lower() == 'up_down':
                        y1 = min(y1, self.scope.cam.GetCCDHeight() - y2)
                        y2 = max(y2, self.scope.cam.GetCCDHeight() - y1)
    
                        if not self.scope.cam.splitterFlip:
                            y1 = 0
                            y2 = self.scope.cam.GetCCDHeight()
                        
                #self.scope.cam.SetROI(x1,y1,x2,y2)
                self.scope.state['Camera.ROI'] = (x1,y1,x2,y2)

                self.roi_on = True
    
                x1 = 0
                y1 = 0
                x2 = self.scope.cam.GetPicWidth()
                y2 = self.scope.cam.GetPicHeight()

            
        logging.debug('Preparing frame wrangler for new ROI size')
        #self.scope.cam.SetCOC()
        #self.scope.cam.GetStatus()
        self.scope.frameWrangler.Prepare()
        self.view.SetDataStack(self.scope.frameWrangler.currentFrame)
        
        self.view.do.SetSelection((x1,y1,0), (x2,y2,0))

        logging.debug('Restarting frame wrangler with new ROI')
        self.scope.frameWrangler.start()
        self.view.Refresh()
        self.view.GetParent().Refresh()
        #event.Skip()

    def SetCentredRoi(self, event=None, halfwidth=5):
        self.scope.frameWrangler.stop()

        w = self.scope.cam.GetCCDWidth()
        h = self.scope.cam.GetCCDHeight()

        x1 = w/2 - halfwidth
        y1 = h/2 - halfwidth
        x2 = w/2 + halfwidth
        y2 = h/2 + halfwidth

        #self.scope.cam.SetROI(x1,y1,x2,y2)
        self.scope.state['Camera.ROI'] = (x1,y1,x2,y2)
        #self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMROI, 'Clear ROI\tF8')
        self.roi_on = True

        x1 = 0
        y1 = 0
        x2 = self.scope.cam.GetPicWidth()
        y2 = self.scope.cam.GetPicHeight()


        #self.scope.cam.SetCOC()
        #self.scope.cam.GetStatus()
        self.scope.frameWrangler.Prepare()
        self.view.SetDataStack(self.scope.frameWrangler.currentFrame)
        
        self.view.do.SetSelection((x1,y1,0), (x2,y2,0))

        self.scope.frameWrangler.start()
        self.view.Refresh()
        self.view.GetParent().Refresh()
        #event.Skip()

    def OnMDisplayClearSel(self, event):
        self.view.ResetSelection()
        #event.Skip()

    @property
    def view(self):
        '''Return the current view panel. This is the panel that is used to display the camera data.
        
        deprecates .vp

        '''
        try:
            return self._view
        except AttributeError:
            raise AttributeError('View panel not yet created. This usually happens if .view is accessed too early in the initialisation process.')

    @property
    def vp(self):
        ''' Backwards compatibility for access to the view panel
        
        Generates a deprecation warning - use .view instead
        '''
        import warnings
        warnings.warn('The .vp attribute is deprecated. Use .view instead', DeprecationWarning)
        return self.view

    def OnCloseWindow(self, event):   
        self.scope.frameWrangler.stop()
        self.time1.Stop()
        if 'cameras' in dir(self.scope):
            for c in self.scope.cameras.values():
                c.Shutdown()
        else:
            self.scope.cam.Shutdown()
            
        for f in self.scope.CleanupFunctions:
            f()
            
        logging.info('All cleanup functions called')
        
        time.sleep(1)
        
        import threading
        msg = 'Remaining Threads:\n'
        for t in threading.enumerate():
            if six.PY3:
                cd = None
                if hasattr(t._target, '__code__'):
                    cd = t._target.__code__
                elif hasattr(t._target, '__func__'):
                    cd = t._target.__func__.__code__
                elif hasattr(t, '__code__'):
                    cd = t.__code__
                else:
                    # Thread sub-class
                    try:
                        cd = t.__class__.run.__code__
                    except AttributeError:
                        pass
                msg += '%s, %s, daemon=%s, %s\n' % (t.name, t._target, t.daemon, cd)
            else:
                msg += '%s, %s, daemon=%s\n' % (t, t._Thread__target, t.daemon)
            
        logging.info(msg)

        self.Destroy()
        wx.Exit()
        
