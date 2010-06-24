#!/usr/bin/python

##################
# smimainframe.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#Boa:Frame:smiMainFrame

#from wxPython.wx import *
import wx
#from wxPython.stc import *
#import wx.stc
import wx.py.shell
import wx.aui

import wx.lib.foldpanelbar as fpb
from PYME.misc.fbpIcons import *

import sys
import os
sys.path.append('.')

from PYME.misc.auiFloatBook import AuiNotebookWithFloatingPages

#import example
import PYME.cSMI as example

import mytimer
import psliders
import intsliders
import seqdialog
import timeseqdialog
import stepDialog
import funcs
import PYME.DSView.dsviewer as dsviewer
import chanfr
import HDFSpoolFrame
from PYME.FileUtils import nameUtils

import splashScreen
import time

import PYME.Acquire.protocol as protocol

def create(parent, options = None):
    return smiMainFrame(parent, options)

[wxID_SMIMAINFRAME, wxID_SMIMAINFRAMENOTEBOOK1, wxID_SMIMAINFRAMEPANEL1, 
 wxID_SMIMAINFRAMESTATUSBAR1, wxID_SMIMAINFRAMETEXTCTRL1, 
] = map(lambda _init_ctrls: wx.NewId(), range(5))

[wxID_SMIMAINFRAMEMENU1FILE_EXIT, wxID_SMIMAINFRAMEMENU1FILE_OPEN, 
 wxID_SMIMAINFRAMEMENU1FILE_OPEN_SCRIPT, 
] = map(lambda _init_coll_menu1_Items: wx.NewId(), range(3))

[wxID_SMIMAINFRAMEMFILEFILE_EXIT, wxID_SMIMAINFRAMEMFILEFILE_OPEN, 
] = map(lambda _init_coll_mFile_Items: wx.NewId(), range(2))

[wxID_SMIMAINFRAMEWINDOWSITEMS0] = map(lambda _init_coll_Windows_Items: wx.NewId(), range(1))

[wxID_SMIMAINFRAMEMAQUIREONE_PIC, wxID_SMIMAINFRAMEMAQUIRESTACK, 
 wxID_SMIMAINFRAMEMAQUIRETD_SERIES, wxID_SMIMAINFRAMEMAQUIRETWO_D_TIME, 
] = map(lambda _init_coll_mAquire_Items: wx.NewId(), range(4))

[wxID_SMIMAINFRAMEMCONTROLSCAM, wxID_SMIMAINFRAMEMCONTROLSINT_TIME, 
 wxID_SMIMAINFRAMEMCONTROLSPIEZO, wxID_SMIMAINFRAMEMCONTROLSPIEZO_INIT, 
 wxID_SMIMAINFRAMEMCONTROLSSTEP, 
] = map(lambda _init_coll_mControls_Items: wx.NewId(), range(5))

[wxID_SMIMAINFRAMEMCAMBIN, wxID_SMIMAINFRAMEMCAMCHANS, 
 wxID_SMIMAINFRAMEMCAMROI, 
] = map(lambda _init_coll_mCam_Items: wx.NewId(), range(3))

[wxID_SMIMAINFRAMEMDISPLAYCLEAR_SEL] = map(lambda _init_coll_mDisplay_Items: wx.NewId(), range(1))

class smiMainFrame(wx.Frame):
    def _init_coll_mCam_Items(self, parent):
        # generated method, don't edit

        parent.Append(wxID_SMIMAINFRAMEMCAMROI,
              'Set ROI\tF8', kind=wx.ITEM_NORMAL)
        parent.Append(wxID_SMIMAINFRAMEMCAMBIN,
              'Turn Binning on', kind=wx.ITEM_CHECK)
        parent.Append(wxID_SMIMAINFRAMEMCAMCHANS,
              'Channels', kind=wx.ITEM_NORMAL)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMCAMBIN, self.OnMCamBin)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMCAMCHANS, self.OnMCamChans)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMCAMROI, self.OnMCamRoi)

    def _init_coll_menuBar1_Menus(self, parent):
        # generated method, don't edit

        parent.Append(menu=self.mFile, title='&File')
        parent.Append(menu=self.mControls, title='&Controls')
        parent.Append(menu=self.mAquire, title='&Aquire')

    def _init_coll_mDisplay_Items(self, parent):
        # generated method, don't edit

        parent.Append(wxID_SMIMAINFRAMEMDISPLAYCLEAR_SEL,
              'Clear Selection', kind=wx.ITEM_NORMAL)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMDISPLAYCLEAR_SEL,
              self.OnMDisplayClearSel)

    def _init_coll_mFile_Items(self, parent):
        # generated method, don't edit

        parent.Append(wxID_SMIMAINFRAMEMFILEFILE_OPEN,
              '&Open Stack', kind=wx.ITEM_NORMAL)
        parent.AppendSeparator()
        parent.Append(wxID_SMIMAINFRAMEMFILEFILE_EXIT,
              'E&xit', kind=wx.ITEM_NORMAL)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMFILEFILE_OPEN, self.OnFileOpenStack)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMFILEFILE_EXIT, self.OnFileExit)

    def _init_coll_mControls_Items(self, parent):
        # generated method, don't edit

        parent.Append(wxID_SMIMAINFRAMEMCONTROLSINT_TIME,
              'Integration Time', kind=wx.ITEM_NORMAL)
        parent.Append(wxID_SMIMAINFRAMEMCONTROLSPIEZO,
              'Piezos', kind=wx.ITEM_NORMAL)
        parent.Append(wxID_SMIMAINFRAMEMCONTROLSSTEP,
              'Stepper Motors', kind=wx.ITEM_NORMAL)
        #parent.AppendMenu( id=wxID_SMIMAINFRAMEMCONTROLSCAM,
        #      string='Camera', subMenu=self.mCam)
        parent.AppendSubMenu(self.mCam, 'Camera')
        parent.AppendSeparator()
        parent.Append(wxID_SMIMAINFRAMEMCONTROLSPIEZO_INIT,
              'Piezo Init', kind=wx.ITEM_NORMAL)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMCONTROLSINT_TIME, self.OnMIntTime)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMCONTROLSPIEZO, self.OnMPiezo)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMCONTROLSSTEP, self.OnMStep)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMCONTROLSPIEZO_INIT,
              self.OnMControlsPiezoInit)

    def _init_coll_mAquire_Items(self, parent):
        # generated method, don't edit

        parent.Append(wxID_SMIMAINFRAMEMAQUIREONE_PIC,
              '1 Pic\tF5', kind=wx.ITEM_NORMAL)
        parent.Append(wxID_SMIMAINFRAMEMAQUIRESTACK,
              '3D Stack', kind=wx.ITEM_NORMAL)
        parent.AppendSeparator()
        parent.Append(wxID_SMIMAINFRAMEMAQUIRETWO_D_TIME,
              'Time Series [2D]', kind=wx.ITEM_NORMAL)
        parent.Append(wxID_SMIMAINFRAMEMAQUIRETD_SERIES,
              'Time Series [3D]', kind=wx.ITEM_NORMAL)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMAQUIRESTACK, self.OnMAquireStack)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMAQUIREONE_PIC, self.OnMAquireOnePic)
        wx.EVT_MENU(self, wxID_SMIMAINFRAMEMAQUIRETWO_D_TIME,self.OnMAquireTwo_d_timeMenu)

    def _init_utils(self):
        # generated method, don't edit
        self.menuBar1 = wx.MenuBar()

        self.mFile = wx.Menu(title='')

        self.mControls = wx.Menu(title='')

        self.mAquire = wx.Menu(title='')

        self.mCam = wx.Menu(title='')

        self.mDisplay = wx.Menu(title='')

        self._init_coll_menuBar1_Menus(self.menuBar1)
        self._init_coll_mFile_Items(self.mFile)
        self._init_coll_mControls_Items(self.mControls)
        self._init_coll_mAquire_Items(self.mAquire)
        self._init_coll_mCam_Items(self.mCam)
        self._init_coll_mDisplay_Items(self.mDisplay)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_SMIMAINFRAME, name='smiMainFrame',
              parent=prnt, pos=wx.Point(20, 20), size=wx.Size(600, 800),
              style=wx.DEFAULT_FRAME_STYLE, title='PYME Acquire')
        self._init_utils()
        self.SetClientSize(wx.Size(400, 800))
        self.SetMenuBar(self.menuBar1)

        self.statusBar1 = wx.StatusBar(id=wxID_SMIMAINFRAMESTATUSBAR1,
              name='statusBar1', parent=self, style=wx.ST_SIZEGRIP)
        self.SetStatusBar(self.statusBar1)

        #self.notebook1 = wx.Notebook(id=wxID_SMIMAINFRAMENOTEBOOK1,
        #      name='notebook1', parent=self, pos=wx.Point(0, 0), size=wx.Size(618,
        #      450), style=0)
        #self.notebook1 = wx.aui.AuiNotebook(id=wxID_SMIMAINFRAMENOTEBOOK1, parent=self, pos=wx.Point(0, 0), size=wx.Size(618,
        #      450), style=wx.aui.AUI_NB_TAB_SPLIT|wx.aui.AUI_NB_TAB_SPLIT|wx.aui.AUI_NB_TAB_SPLIT)

        self.notebook1 = AuiNotebookWithFloatingPages(id=wxID_SMIMAINFRAMENOTEBOOK1, parent=self, pos=wx.Point(0, 0), size=wx.Size(400,
              -1), style=wx.aui.AUI_NB_TAB_SPLIT|wx.aui.AUI_NB_TAB_SPLIT|wx.aui.AUI_NB_TAB_SPLIT)

#        self.panel1 = wx.Panel(id=wxID_SMIMAINFRAMEPANEL1, name='panel1',
#              parent=self.notebook1, pos=wx.Point(0, 0), size=wx.Size(-1, -1),
#              style=wx.TAB_TRAVERSAL)
#
#        self.textCtrl1 = wx.TextCtrl(id=wxID_SMIMAINFRAMETEXTCTRL1,
#              name='textCtrl1', parent=self.panel1, pos=wx.Point(121, 56),
#              size=wx.Size(368, 312), style=0,
#              value='PySMI    Version 0.5   KIP Heidelberg')
#        self.textCtrl1.Enable(False)
#        self.textCtrl1.Center(wx.BOTH)

    def __init__(self, parent, options = None):
        self.options = options
        self._init_ctrls(parent)

        wx.EVT_CLOSE(self, self.OnCloseWindow)        
        
        self.MainFrame = self #reference to this window for use in scripts etc...
        self.MainMenu = self.menuBar1
        protocol.MainFrame = self

        self.toolPanels = []
        self.camPanels = []
        self.postInit = []

        self.initDone = False

        self.scope = funcs.microscope()

        self.splash = splashScreen.SplashScreen(self, self.scope)
        self.splash.Show()
        

        self.sh = wx.py.shell.Shell(id=-1,
              parent=self.notebook1, size=wx.Size(-1, -1), style=0, locals=self.__dict__,
              introText='Python SMI bindings - note that help, license etc below is for Python, not PySMI\n\n')
        
        #self.notebook1.AddPage(imageId=-1, page=self.sh, select=True, text='Console')
        #self.notebook1.AddPage(imageId=-1, page=self.panel1, select=False, text='About')
        
        self.notebook1.AddPage(page=self.sh, select=True, caption='Console')
#        self.notebook1.AddPage( page=self.panel1, select=False, caption='About')

        self.SetSize((400, 800))
        self.CreateToolPanel()
        #self.SetSize((1000, 800))
        
        
        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.notebook1, 1, wx.EXPAND)
        self.SetSizer(self.Sizer)
        self.SetSize((1000, 800))
        
        self.roi_on = False
        self.bin_on = False
        
        self.time1 = mytimer.mytimer()
        
        self.time1.Start(500)

        self.time1.WantNotification.append(self.runInitScript)
        self.time1.WantNotification.append(self.checkInitDone)
        self.time1.WantNotification.append(self.splash.Tick)

    def runInitScript(self):
        self.time1.WantNotification.remove(self.runInitScript)
        #self.sh.shell.runfile('init.py')
        initFile = 'init.py'
        if not self.options == None and not self.options.initFile == None:
            initFile = self.options.initFile
        self.sh.run('import ExecTools')
        self.sh.run('ExecTools.setDefaultNamespace(locals(), globals())')
        self.sh.run('from ExecTools import InitBG, joinBGInit, InitGUI, HWNotPresent')
        #self.sh.run('''def InitGUI(code):\n\tpostInit.append(code)\n\n\n''')
        self.sh.run('ExecTools.execFileBG("%s", locals(), globals())' % initFile)

        

    def checkInitDone(self):
        if self.scope.initDone == True and self.checkInitDone in self.time1.WantNotification:
            self.time1.WantNotification.remove(self.checkInitDone)
            #self.time1.WantNotification.remove(self.splash.Tick)
            self.doPostInit()

    def doPostInit(self):
        for cm in self.postInit:
            for cl in cm.split('\n'):
                self.sh.run(cl)

        if len(self.scope.piezos) > 0.5:
            self.piezo_sl = psliders.PiezoSliders(self.scope.piezos, self)
            self.time1.WantNotification.append(self.piezo_sl.update)

            self.AddTool(self.piezo_sl, 'Piezo Control')

            #self.notebook1.AddPage( page=self.piezo_sl, select=False, caption='Piezo Control')
            #self.notebook1.Split(self.notebook1.GetPageCount() -1, wx.DOWN)
            #self.piezo_sl.Show()

            #self.time1.WantNotification.append(self.piezo_sl.update)

            self.seq_d = seqdialog.seqPanel(self, self.scope)
            self.AddAqTool(self.seq_d, 'Z-Stack')
            #self.seq_d.Show()

        if (self.scope.cam.CamReady() and ('chaninfo' in self.scope.__dict__)):
            self.scope.livepreview(self, Notebook = self.notebook1)
            

            self.int_sl = intsliders.IntegrationSliders(self.scope.chaninfo,self, self.scope)
            self.AddCamTool(self.int_sl, 'Integration Time')
            #self.notebook1.AddPage( page=self.int_sl, select=False, caption='Integration Time')
            #self.notebook1.Split(self.notebook1.GetPageCount() -1, wx.DOWN)
            #self.int_sl.Show()

            self.tseq_d = timeseqdialog.seqDialog(self, self.scope)

            self.pan_spool = HDFSpoolFrame.PanSpool(self, self.scope, nameUtils.genHDFDataFilepath())
            self.AddAqTool(self.pan_spool, 'Spooling')

        
            
        if 'step' in self.scope.__dict__:
            self.step_d = stepDialog.stepPanel(self, self.scope)
            #self.step_d.Show()
            self.AddTool(self.step_d, 'Stepper Motors')
        
        self.time1.WantNotification.append(self.StatusBarUpdate)

        for t in self.toolPanels:
            print t
            self.AddTool(*t)

        for t in self.camPanels:
            print t
            self.AddCamTool(*t)

        #self.splash.Destroy()
        
        self.initDone = True

        #fudge to get layout right
        panes = self.notebook1.GetAuiManager().AllPanes

        self.paneNames = []

        for p in panes:
            self.paneNames.append(p.name)

        #self.LoadPerspective()

    def _getPerspectiveFilename(self):
        #print __file__
        fname = os.path.join(os.path.split(__file__)[0], 'GUILayout.txt')
        return fname

    def SavePerspective(self):
        mgr = self.notebook1.GetAuiManager()
        persp = mgr.SavePerspective()

        for i, p in enumerate(self.paneNames):
            persp = persp.replace(p, 'pane_%d' % i)
            
        f = open(self._getPerspectiveFilename(), 'w')
        f.write(persp)
        f.close()

    def LoadPerspective(self):
        pfname = self._getPerspectiveFilename()
        if os.path.exists(pfname):
            f = open(pfname)
            pesp = f.read()
            f.close()

            for i, p in enumerate(self.paneNames):
                persp = persp.replace('pane_%d' % i, p)

            mgr = self.notebook1.GetAuiManager()
            mgr.LoadPerspective(persp)
            mgr.Update()


    def CreateToolPanel(self):

        # delete earlier panel
        #self._leftWindow1.DestroyChildren()

        # recreate the foldpanelbar
        self.Images = wx.ImageList(16,16)
        self.Images.Add(GetExpandedIconBitmap())
        self.Images.Add(GetCollapsedIconBitmap())

#        self._leftWindow1 = wx.SashLayoutWindow(self, 101, wx.DefaultPosition,
#                                                wx.Size(300, 1000), wx.NO_BORDER |
#                                                wx.SW_3D | wx.CLIP_CHILDREN)
#
#        self._leftWindow1.SetDefaultSize(wx.Size(220, 1000))
#        self._leftWindow1.SetOrientation(wx.LAYOUT_VERTICAL)
#        self._leftWindow1.SetAlignment(wx.LAYOUT_LEFT)
#        self._leftWindow1.SetSashVisible(wx.SASH_RIGHT, True)
#        self._leftWindow1.SetExtraBorderSize(10)

        self.camPanel = fpb.FoldPanelBar(self.notebook1, -1, wx.DefaultPosition,
                                     wx.Size(200,1000), fpb.FPB_DEFAULT_STYLE,0)

        self.notebook1.AddPage(page=self.camPanel, select=False, caption='Camera')
        self.notebook1.Split(self.notebook1.GetPageCount() -1, wx.RIGHT)



        self.toolPanel = fpb.FoldPanelBar(self.notebook1, -1, wx.DefaultPosition,
                                     wx.Size(200,1000), fpb.FPB_DEFAULT_STYLE,0)
        
        self.notebook1.AddPage(page=self.toolPanel, select=False, caption='Hardware')
        self.notebook1.Split(self.notebook1.GetPageCount() -1, wx.RIGHT)
        #self.notebook1.Split(self.notebook1.GetPageCount() -2, wx.RIGHT)

#        self._rightWindow1 = wx.SashLayoutWindow(self, 101, wx.DefaultPosition,
#                                                wx.Size(300, 1000), wx.NO_BORDER |
#                                                wx.SW_3D | wx.CLIP_CHILDREN)
#
#        self._rightWindow1.SetDefaultSize(wx.Size(300, 1000))
#        self._rightWindow1.SetOrientation(wx.LAYOUT_VERTICAL)
#        self._rightWindow1.SetAlignment(wx.LAYOUT_LEFT)
#        self._rightWindow1.SetSashVisible(wx.SASH_RIGHT, True)
#        self._rightWindow1.SetExtraBorderSize(10)


        self.aqPanel = fpb.FoldPanelBar(self.notebook1, -1, wx.DefaultPosition,
                                     wx.Size(300,1000), fpb.FPB_DEFAULT_STYLE,0)



        self.notebook1.AddPage(page=self.aqPanel, select=False, caption='Acquisition')
        #self.notebook1.Split(self.notebook1.GetPageCount() -2, wx.RIGHT)
        #self.notebook1.Split(self.notebook1.GetPageCount() -3, wx.RIGHT)
        #self.notebook1.Split(self.notebook1.GetPageCount() -2, wx.RIGHT)
        #self.notebook1.Split(self.notebook1.GetPageCount() -2, wx.RIGHT)

        self.notebook1.SetSelection(0)

    def AddTool(self, panel, title):
        item = self.toolPanel.AddFoldPanel(title, collapsed=False, foldIcons=self.Images)
        panel.Reparent(item)
        self.toolPanel.AddFoldPanelWindow(item, panel, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        #wx.LayoutAlgorithm().LayoutWindow(self, self._leftWindow1)

    def AddCamTool(self, panel, title):
        item = self.camPanel.AddFoldPanel(title, collapsed=False, foldIcons=self.Images)
        panel.Reparent(item)
        self.camPanel.AddFoldPanelWindow(item, panel, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

    def AddAqTool(self, panel, title):
        item = self.aqPanel.AddFoldPanel(title, collapsed=False, foldIcons=self.Images)
        panel.Reparent(item)
        self.aqPanel.AddFoldPanelWindow(item, panel, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        #wx.LayoutAlgorithm().LayoutWindow(self, self._leftWindow1)

    def OnFileOpenStack(self, event):
        self.dv = dsviewer.DSViewFrame(self)
        self.dv.Show()
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
        self.scope.pa.stop()
        ds2 = example.CDataStack(self.scope.pa.ds)

        df2 = dsviewer.DSViewFrame(self, '--new pic--', ds2)
        df2.Show()
        self.scope.pa.Prepare(True)
        self.scope.pa.start()

        #event.Skip()

    def OnMIntTime(self, event):
        self.int_sl.Show()
        self.int_sl.Raise()
        #event.Skip()

    def OnMPiezo(self, event):
        self.piezo_sl.Show()
        self.piezo_sl.Raise()
        #event.Skip()

    def OnMStep(self, event):
        self.step_d.Show()
        self.step_d.Raise()
        #event.Skip()

    def OnMCamBin(self, event):
        self.scope.pa.stop()
        if (self.bin_on):
            self.scope.cam.SetHorizBin(0)
            self.scope.cam.SetVertBin(0)
            self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMBIN, 'Turn Binning On')
            self.bin_on = False
        else:
            self.scope.cam.SetHorizBin(1)
            self.scope.cam.SetVertBin(1)
            self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMBIN, 'Turn Binning Off')
            self.bin_on = True
            
        self.scope.cam.SetCOC()
        self.scope.cam.GetStatus()
        self.scope.pa.Prepare()
        self.scope.vp.SetDataStack(self.scope.pa.ds)
        self.scope.pa.start()
        #event.Skip()

    def OnMCamChans(self, event):
        #return
        self.scope.pa.stop()
        
        chand = chanfr.ChanFrame(self, self.scope.chaninfo)
        chand.ShowModal()
        
        self.int_sl.Destroy()
        self.int_sl = intsliders.IntegrationSliders(self.scope.chaninfo,self)
        self.int_sl.Show()
            
        self.scope.pa.Prepare()
        self.scope.vp.SetDataStack(self.scope.pa.ds)
        self.scope.pa.start()
        #event.Skip()

    def OnMCamRoi(self, event):
        self.scope.pa.stop()
        
        #print (self.scope.vp.selection_begin_x, self.scope.vp.selection_begin_y, self.scope.vp.selection_end_x, self.scope.vp.selection_end_y)
        
        if (self.roi_on):
            x1 = self.scope.cam.GetROIX1()
            y1 = self.scope.cam.GetROIY1()
            x2 = self.scope.cam.GetROIX2()
            y2 = self.scope.cam.GetROIY2()

            self.scope.cam.SetROI(0,0, self.scope.cam.GetCCDWidth(), self.scope.cam.GetCCDHeight())
            self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMROI, 'Set ROI\tF8')
            self.roi_on = False
        else:
            x1 = self.scope.vp.selection_begin_x
            y1 = self.scope.vp.selection_begin_y
            x2 = self.scope.vp.selection_end_x
            y2 = self.scope.vp.selection_end_y

            #if we're splitting colours/focal planes across the ccd, then only allow symetric ROIs
            if 'splitting' in dir(self.scope):
                if self.scope.splitting.lower() == 'left_right':
                    x1 = min(x1, self.scope.cam.GetCCDWidth() - x2)
                    x2 = max(x2, self.scope.cam.GetCCDWidth() - x1)
                if self.scope.splitting.lower() == 'up_down':
                    y1 = min(y1, self.scope.cam.GetCCDHeight() - y2)
                    y2 = max(y2, self.scope.cam.GetCCDHeight() - y1)
                    
            self.scope.cam.SetROI(x1,y1,x2,y2)
            self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMROI, 'Clear ROI\tF8')
            self.roi_on = True

            x1 = 0
            y1 = 0
            x2 = self.scope.cam.GetPicWidth()
            y2 = self.scope.cam.GetPicHeight()

            
        self.scope.cam.SetCOC()
        self.scope.cam.GetStatus()
        self.scope.pa.Prepare()
        self.scope.vp.SetDataStack(self.scope.pa.ds)
        
        self.scope.vp.selection_begin_x = x1
        self.scope.vp.selection_begin_y = y1
        self.scope.vp.selection_end_x = x2
        self.scope.vp.selection_end_y = y2

        self.scope.pa.start()
        self.scope.vp.Refresh()
        self.scope.vp.GetParent().Refresh()
        #event.Skip()

    def SetCentredRoi(self, event=None, halfwidth=5):
        self.scope.pa.stop()

        #print (self.scope.vp.selection_begin_x, self.scope.vp.selection_begin_y, self.scope.vp.selection_end_x, self.scope.vp.selection_end_y)

        w = self.scope.cam.GetCCDWidth()
        h = self.scope.cam.GetCCDHeight()

        x1 = w/2 - halfwidth
        y1 = h/2 - halfwidth
        x2 = w/2 + halfwidth
        y2 = h/2 + halfwidth

        self.scope.cam.SetROI(x1,y1,x2,y2)
        self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMROI, 'Clear ROI\tF8')
        self.roi_on = True

        x1 = 0
        y1 = 0
        x2 = self.scope.cam.GetPicWidth()
        y2 = self.scope.cam.GetPicHeight()


        self.scope.cam.SetCOC()
        self.scope.cam.GetStatus()
        self.scope.pa.Prepare()
        self.scope.vp.SetDataStack(self.scope.pa.ds)

        self.scope.vp.selection_begin_x = x1
        self.scope.vp.selection_begin_y = y1
        self.scope.vp.selection_end_x = x2
        self.scope.vp.selection_end_y = y2

        self.scope.pa.start()
        self.scope.vp.Refresh()
        self.scope.vp.GetParent().Refresh()
        #event.Skip()

    def OnMDisplayClearSel(self, event):
        self.vp.ResetSelection()
        #event.Skip()

    def OnMControlsPiezoInit(self, event):
        self.scope.pz.Init(1)
        self.piezo_sl.update()
        #event.Skip()

    def OnMAquireTwo_d_timeMenu(self, event):
        self.tseq_d.Show()
        self.tseq_d.Raise()
        #event.Skip()

    def OnCloseWindow(self, event):   
        self.scope.pa.stop()
        self.time1.Stop()
        self.scope.cam.Shutdown()
        #self.int_sl.Destroy()
        #self.piezo_sl.Destroy()
        #self.seq_d.Destroy()
        #self.Close()
        self.Destroy()
        wx.Exit()
        
