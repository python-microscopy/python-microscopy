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
''' 
This contains the bulk of the GUI code for the main window of PYMEAcquire.
'''
import wx
import wx.py.shell
import wx.lib.agw.aui as aui

import PYME.misc.autoFoldPanel as afp
import os
import time

import logging
logging.basicConfig(level=logging.DEBUG)

import PYME.DSView.displaySettingsPanel as disppanel
from PYME.DSView import arrayViewPanel
from PYME.DSView import dsviewer as dsviewer

from PYME.Acquire import mytimer

from PYME.Acquire.ui import positionUI
from PYME.Acquire.ui import intsliders
from PYME.Acquire.ui import seqdialog
from PYME.Acquire.ui import selectCameraPanel
from PYME.Acquire.ui import splashScreen

from PYME.Acquire import microscope
from PYME.Acquire import protocol
from PYME.Acquire import HDFSpoolFrame

from PYME.io import MetaDataHandler
from PYME.io.FileUtils import nameUtils


from PYME.ui.AUIFrame import AUIFrame

def create(parent, options = None):
    return PYMEMainFrame(parent, options)

class PYMEMainFrame(AUIFrame):
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
              name='statusBar1', parent=self, style=wx.ST_SIZEGRIP)
        self.SetStatusBar(self.statusBar1)



    def __init__(self, parent, options = None):
        AUIFrame.__init__(self, id=wx.ID_ANY, name='smiMainFrame',
              parent=parent, pos=wx.Point(20, 20), size=wx.Size(600, 800),
              style=wx.DEFAULT_FRAME_STYLE, title='PYME Acquire')    
        
        self._create_menu()
        self.options = options

        self.snapNum = 0

        wx.EVT_CLOSE(self, self.OnCloseWindow)        
        
        self.MainFrame = self #reference to this window for use in scripts etc...
        protocol.MainFrame = self

        self.toolPanels = []
        self.camPanels = []
        self.postInit = []

        self.initDone = False

        self.scope = microscope.microscope()

        self.splash = splashScreen.SplashScreen(self, self.scope)
        self.splash.Show()
        

        self.sh = wx.py.shell.Shell(id=-1,
              parent=self, size=wx.Size(-1, -1), style=0, locals=self.__dict__,
              introText='Python SMI bindings - note that help, license etc below is for Python, not PySMI\n\n')
        self.AddPage(self.sh, caption='Console')

        self.CreateToolPanel()

        self.SetSize((1030, 895))

        self.roi_on = False
        self.bin_on = False
        
        self.time1 = mytimer.mytimer()
        
        self.time1.Start(500)

        self.time1.WantNotification.append(self.runInitScript)
        self.time1.WantNotification.append(self.checkInitDone)
        self.time1.WantNotification.append(self.splash.Tick)


    def runInitScript(self):
        #import os
        self.time1.WantNotification.remove(self.runInitScript)
        #self.sh.shell.runfile('init.py')
        #fstub = os.path.join(os.path.split(__file__)[0], 'Scripts')
        initFile = 'init.py'
        if not self.options == None and not self.options.initFile == None:
            initFile = self.options.initFile
            
        #initFile = os.path.join(fstub, initFile)
        self.sh.run('from PYME.Acquire import ExecTools')
        self.sh.run('ExecTools.setDefaultNamespace(locals(), globals())')
        self.sh.run('from PYME.Acquire.ExecTools import InitBG, joinBGInit, InitGUI, HWNotPresent')
        #self.sh.run('''def InitGUI(code):\n\tpostInit.append(code)\n\n\n''')
        self.sh.run('ExecTools.execFileBG("%s", locals(), globals())' % initFile)

        

    def checkInitDone(self):
        if self.scope.initDone == True and self.checkInitDone in self.time1.WantNotification:
            self.time1.WantNotification.remove(self.checkInitDone)
            #self.time1.WantNotification.remove(self.splash.Tick)
            self.doPostInit()
    
    def _refreshDataStack(self):
        if 'vp' in dir(self):
            if not (self.vp.do.ds.data is self.scope.frameWrangler.currentFrame):
                self.vp.SetDataStack(self.scope.frameWrangler.currentFrame)
        
    def livepreview(self):
        self.scope.startAquisistion()

        if self.scope.cam.GetPicHeight() > 1:
            if 'vp' in dir(self):
                    self.vp.SetDataStack(self.scope.frameWrangler.currentFrame)
            else:
                self.vp = arrayViewPanel.ArrayViewPanel(self, self.scope.frameWrangler.currentFrame)
                self.vp.crosshairs = False
                self.vp.showScaleBar = False
                self.vp.do.leftButtonAction = self.vp.do.ACTION_SELECTION
                self.vp.do.showSelection = True
                self.vp.CenteringHandlers.append(self.scope.centreView)

                self.vsp = disppanel.dispSettingsPanel2(self, self.vp)


                self.time1.WantNotification.append(self.vsp.RefrData)
                self.time1.WantNotification.append(self._refreshDataStack)

                self.AddPage(page=self.vp, select=True,caption='Preview')

                self.AddCamTool(self.vsp, 'Display')

            #self.scope.frameWrangler.WantFrameGroupNotification.append(self.vp.Redraw)
            self.scope.frameWrangler.onFrameGroup.connect(self.vp.Redraw)

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
            
        self.scope.PACallbacks.append(self._refreshDataStack)


    def doPostInit(self):
        logging.debug('Starting post-init')
        for cm in self.postInit:
            #print cm
            for cl in cm.split('\n'):
                self.sh.run(cl)

        #if len(self.scope.piezos) > 0.5:
        #    self.piezo_sl = psliders.PiezoSliders(self.scope.piezos, self, self.scope.joystick)
        #    self.time1.WantNotification.append(self.piezo_sl.update)
            
        if len(self.scope.positioning.keys()) > 0.5:
            self.pos_sl = positionUI.PositionSliders(self.scope, self, self.scope.joystick)
            self.time1.WantNotification.append(self.pos_sl.update)

            self.AddTool(self.pos_sl, 'Positioning')

            self.seq_d = seqdialog.seqPanel(self, self.scope)
            self.AddAqTool(self.seq_d, 'Z-Stack')
            #self.seq_d.Show()

        if (self.scope.cam.CamReady() and ('chaninfo' in self.scope.__dict__)):
            self.livepreview()
            

            self.int_sl = intsliders.IntegrationSliders(self.scope.chaninfo,self, self.scope)
            self.AddCamTool(self.int_sl, 'Integration Time')

            if len(self.scope.cameras) > 1:
                self.pCamChoose = selectCameraPanel.CameraChooserPanel(self, self.scope)
                self.AddCamTool(self.pCamChoose, 'Camera Selection')


            self.pan_spool = HDFSpoolFrame.PanSpool(self, self.scope, nameUtils.genHDFDataFilepath())
            self.AddAqTool(self.pan_spool, 'Spooling')

        
        self.time1.WantNotification.append(self.StatusBarUpdate)

        for t in self.toolPanels:
            #print(t)
            self.AddTool(*t)

        for t in self.camPanels:
            #print(t)
            self.AddCamTool(*t)

        #self.splash.Destroy()

        
        
        self.initDone = True
        self._mgr.Update()

        if 'pCamChoose' in dir(self):
            self.pCamChoose.OnCCamera(None)
            
        logging.debug('Finished post-init')

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


    def CreateToolPanel(self):
        self.camPanel = afp.foldPanel(self, -1, wx.DefaultPosition,
                                     wx.Size(240,1000))

        cpinfo = aui.AuiPaneInfo().Name("camControls").Caption("Camera").Layer(1).Right().CloseButton(False)
        #cpinfo.dock_proportion  = int(cpinfo.dock_proportion*1.6)
        
        self._mgr.AddPane(self.camPanel, cpinfo)

        self.aqPanel = afp.foldPanel(self, -1, wx.DefaultPosition,
                                     wx.Size(240,1000))

        aqinfo = aui.AuiPaneInfo().Name("aqControls").Caption("Acquisition").Layer(2).Position(0).Right().CloseButton(False)
        
        self._mgr.AddPane(self.aqPanel, aqinfo)

        self.toolPanel = afp.foldPanel(self, -1, wx.DefaultPosition,
                                     wx.Size(240,1000))
#        self.toolPanel = fpb.FoldPanelBar(self, -1, wx.DefaultPosition,
#                                     wx.Size(240,200))#, fpb.FPB_DEFAULT_STYLE,0)

        self._mgr.AddPane(self.toolPanel, aui.AuiPaneInfo().
                          Name("hardwareControls").Caption("Hardware").Layer(2).Position(1).Right().CloseButton(False).BestSize(240, 250))

        aqinfo.dock_proportion  = int(aqinfo.dock_proportion*1.3)



    def AddTool(self, panel, title, pinned=True):
        '''Adds a pane to the tools section of the GUI
        
        Parameters
        ----------
        panel : an instance of a wx.Window derived class
            The pane to add
        title : string
            The caption for the panel.
        '''
        item = afp.foldingPane(self.toolPanel, -1, caption=title, pinned = pinned)
        panel.Reparent(item)
        item.AddNewElement(panel)
        self.toolPanel.AddPane(item)
#        item = self.toolPanel.AddFoldPanel(title, collapsed=False, foldIcons=self.Images)
#        panel.Reparent(item)
#        self.toolPanel.AddFoldPanelWindow(item, panel, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        #wx.LayoutAlgorithm().LayoutWindow(self, self._leftWindow1)

    def AddCamTool(self, panel, title, pinned=True):
        '''Adds a pane to the Camera section of the GUI
        
        Parameters
        ----------
        panel : an instance of a wx.Window derived class
            The pane to add
        title : string
            The caption for the panel.
        '''
        #item = self.camPanel.AddFoldPanel(title, collapsed=False, foldIcons=self.Images)
        item = afp.foldingPane(self.camPanel, -1, caption=title, pinned = pinned)
        panel.Reparent(item)
        item.AddNewElement(panel)
        self.camPanel.AddPane(item)
        #self.camPanel.AddFoldPanelWindow(item, panel, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

    def AddAqTool(self, panel, title, pinned=True):
        '''Adds a pane to the Acquisition section of the GUI
        
        Parameters
        ----------
        panel : an instance of a wx.Window derived class
            The pane to add
        title : string
            The caption for the panel.
        '''
        item = afp.foldingPane(self.aqPanel, -1, caption=title, pinned = pinned)
        panel.Reparent(item)
        item.AddNewElement(panel)
        self.aqPanel.AddPane(item)
        #item = self.aqPanel.AddFoldPanel(title, collapsed=False, foldIcons=self.Images)
        #panel.Reparent(item)
        #self.aqPanel.AddFoldPanelWindow(item, panel, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        #wx.LayoutAlgorithm().LayoutWindow(self, self._leftWindow1)

    def OnFileOpenStack(self, event):
        #self.dv = dsviewer.DSViewFrame(self)
        #self.dv.Show()
        im = dsviewer.ImageStack()
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
        self.scope.frameWrangler.stop()
        if (self.bin_on):
            self.scope.cam.SetHorizBin(1)
            self.scope.cam.SetVertBin(1)
            #self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMBIN, 'Turn Binning On')
            self.bin_on = False
        else:
            self.scope.cam.SetHorizBin(16)
            self.scope.cam.SetVertBin(16)
            #self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMBIN, 'Turn Binning Off')
            self.bin_on = True
            
        self.scope.cam.SetCOC()
        self.scope.cam.GetStatus()
        self.scope.frameWrangler.Prepare()
        self.vp.SetDataStack(self.scope.frameWrangler.currentFrame)
        self.pa.start()
        #event.Skip()

    def OnMCamSetPixelSize(self, event):
        from PYME.Acquire.ui import voxelSizeDialog

        dlg = voxelSizeDialog.VoxelSizeDialog(self, self.scope)
        dlg.ShowModal()


    def OnMCamRoi(self, event):
        self.scope.frameWrangler.stop()
        
        #print (self.scope.vp.selection_begin_x, self.scope.vp.selection_begin_y, self.scope.vp.selection_end_x, self.scope.vp.selection_end_y)

        if 'validROIS' in dir(self.scope.cam):
            #special case for cameras with restricted ROIs - eg Neo
            print('setting ROI')
            dlg = wx.SingleChoiceDialog(self, 'Please select the ROI size', 'Camera ROI', ['%dx%d at (%d, %d)' % roi for roi in self.scope.cam.validROIS])
            dlg.ShowModal()
            print('Dlg Shown')
            self.scope.cam.SetROIIndex(dlg.GetSelection())
            dlg.Destroy()
            
            x1 = 0
            y1 = 0
            x2 = self.scope.cam.GetPicWidth()
            y2 = self.scope.cam.GetPicHeight()
            print ('ROI Set')
        
        else:
            if (self.roi_on):
                x1 = self.scope.cam.GetROIX1()
                y1 = self.scope.cam.GetROIY1()
                x2 = self.scope.cam.GetROIX2()
                y2 = self.scope.cam.GetROIY2()
    
                self.scope.cam.SetROI(0,0, self.scope.cam.GetCCDWidth(), self.scope.cam.GetCCDHeight())
                #self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMROI, 'Set ROI\tF8')
                self.roi_on = False
            else:
    
                x1, y1, x2, y2 = self.vp.do.GetSliceSelection()
    
                #if we're splitting colours/focal planes across the ccd, then only allow symetric ROIs
                if 'splitting' in dir(self.scope.cam):
                    if self.scope.cam.splitting.lower() == 'left_right':
                        x1 = min(x1, self.scope.cam.GetCCDWidth() - x2)
                        x2 = max(x2, self.scope.cam.GetCCDWidth() - x1)
                    if self.scope.cam.splitting.lower() == 'up_down':
                        y1 = min(y1, self.scope.cam.GetCCDHeight() - y2)
                        y2 = max(y2, self.scope.cam.GetCCDHeight() - y1)
    
                        if not self.scope.cam.splitterFlip:
                            y1 = 0
                            y2 = self.scope.cam.GetCCDHeight()
                        
                self.scope.cam.SetROI(x1,y1,x2,y2)
                #self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMROI, 'Clear ROI\tF8')
                self.roi_on = True
    
                x1 = 0
                y1 = 0
                x2 = self.scope.cam.GetPicWidth()
                y2 = self.scope.cam.GetPicHeight()

            
        print('about to set COC')
        self.scope.cam.SetCOC()
        self.scope.cam.GetStatus()
        self.scope.frameWrangler.Prepare()
        self.vp.SetDataStack(self.scope.frameWrangler.currentFrame)
        
        self.vp.do.SetSelection((x1,y1,0), (x2,y2,0))

        self.scope.frameWrangler.start()
        self.vp.Refresh()
        self.vp.GetParent().Refresh()
        #event.Skip()

    def SetCentredRoi(self, event=None, halfwidth=5):
        self.scope.frameWrangler.stop()

        #print (self.scope.vp.selection_begin_x, self.scope.vp.selection_begin_y, self.scope.vp.selection_end_x, self.scope.vp.selection_end_y)

        w = self.scope.cam.GetCCDWidth()
        h = self.scope.cam.GetCCDHeight()

        x1 = w/2 - halfwidth
        y1 = h/2 - halfwidth
        x2 = w/2 + halfwidth
        y2 = h/2 + halfwidth

        self.scope.cam.SetROI(x1,y1,x2,y2)
        #self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMROI, 'Clear ROI\tF8')
        self.roi_on = True

        x1 = 0
        y1 = 0
        x2 = self.scope.cam.GetPicWidth()
        y2 = self.scope.cam.GetPicHeight()


        self.scope.cam.SetCOC()
        self.scope.cam.GetStatus()
        self.scope.frameWrangler.Prepare()
        self.vp.SetDataStack(self.scope.frameWrangler.currentFrame)
        
        self.vp.do.SetSelection((x1,y1,0), (x2,y2,0))

        self.scope.frameWrangler.start()
        self.vp.Refresh()
        self.vp.GetParent().Refresh()
        #event.Skip()

    def OnMDisplayClearSel(self, event):
        self.vp.ResetSelection()
        #event.Skip()


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
            
        print 'All cleanup functions called'
        
        time.sleep(1)
        
        import threading
        print 'Remaining Threads:'
        for t in threading.enumerate():
            print t, t._Thread__target

        self.Destroy()
        wx.Exit()
        
