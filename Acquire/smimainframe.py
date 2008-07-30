#Boa:Frame:smiMainFrame

#from wxPython.wx import *
import wx
#from wxPython.stc import *
import wx.stc
import wx.py.shell
import wx.aui

import sys
sys.path.append('.')

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
              'Set ROI', kind=wx.ITEM_NORMAL)
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
        parent.Append(menu=self.mControls, title='Controls')
        parent.Append(menu=self.mAquire, title='Aquire')

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
              '1 Pic', kind=wx.ITEM_NORMAL)
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
              parent=prnt, pos=wx.Point(55, 392), size=wx.Size(626, 516),
              style=wx.DEFAULT_FRAME_STYLE, title='PySMI')
        self._init_utils()
        self.SetClientSize(wx.Size(618, 489))
        self.SetMenuBar(self.menuBar1)

        self.statusBar1 = wx.StatusBar(id=wxID_SMIMAINFRAMESTATUSBAR1,
              name='statusBar1', parent=self, style=wx.ST_SIZEGRIP)
        self.SetStatusBar(self.statusBar1)

        #self.notebook1 = wx.Notebook(id=wxID_SMIMAINFRAMENOTEBOOK1,
        #      name='notebook1', parent=self, pos=wx.Point(0, 0), size=wx.Size(618,
        #      450), style=0)
        self.notebook1 = wx.aui.AuiNotebook(id=wxID_SMIMAINFRAMENOTEBOOK1, parent=self, pos=wx.Point(0, 0), size=wx.Size(618,
              450), style=wx.aui.AUI_NB_TAB_SPLIT|wx.aui.AUI_NB_TAB_SPLIT|wx.aui.AUI_NB_TAB_SPLIT)

        self.panel1 = wx.Panel(id=wxID_SMIMAINFRAMEPANEL1, name='panel1',
              parent=self.notebook1, pos=wx.Point(0, 0), size=wx.Size(610, 425),
              style=wx.TAB_TRAVERSAL)

        self.textCtrl1 = wx.TextCtrl(id=wxID_SMIMAINFRAMETEXTCTRL1,
              name='textCtrl1', parent=self.panel1, pos=wx.Point(121, 56),
              size=wx.Size(368, 312), style=0,
              value='PySMI    Version 0.5   KIP Heidelberg')
        self.textCtrl1.Enable(False)
        self.textCtrl1.Center(wx.BOTH)

    def __init__(self, parent, options = None):
        self.options = options
        self._init_ctrls(parent)

        wx.EVT_CLOSE(self, self.OnCloseWindow)        
        
        self.sh = wx.py.shell.Shell(id=-1,
              parent=self.notebook1, pos=wx.Point(0, 0), size=wx.Size(618, 451), style=0, locals=self.__dict__, 
              introText='Python SMI bindings - note that help, license etc below is for Python, not PySMI\n\n')
        
        #self.notebook1.AddPage(imageId=-1, page=self.sh, select=True, text='Console')
        #self.notebook1.AddPage(imageId=-1, page=self.panel1, select=False, text='About')
        
        self.notebook1.AddPage(page=self.sh, select=True, caption='Console')
        self.notebook1.AddPage( page=self.panel1, select=False, caption='About')
        
        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.notebook1, 1, wx.EXPAND)
        self.SetSizer(self.Sizer)
        
        self.roi_on = False
        self.bin_on = False
        
        self.time1 = mytimer.mytimer()
        self.scope = funcs.microscope()
        self.time1.Start(500)
        #self.sh.shell.runfile('init.py')
        initFile = 'init.py'
        if not self.options == None and not self.options.initFile == None:
            initFile = self.options.initFile
        self.sh.run('import ExecTools')
        self.sh.run('ExecTools.execFile("%s", locals(), globals())' % initFile)
                
        if (self.scope.cam.CamReady() and ('chaninfo' in self.scope.__dict__)):
            self.scope.livepreview(Notebook = self.notebook1)
            
            self.int_sl = intsliders.IntegrationSliders(self.scope.chaninfo,self)
            self.int_sl.Show()
            self.tseq_d = timeseqdialog.seqDialog(self, self.scope)

        if len(self.scope.piezos) > 0.5:
            self.piezo_sl = psliders.PiezoSliders(self.scope.piezos, self)
            self.piezo_sl.Show()

            #self.time1.WantNotification.append(self.piezo_sl.update)

            self.seq_d = seqdialog.seqDialog(self, self.scope)
            self.seq_d.Show()
            
        if 'step' in self.scope.__dict__:
            self.step_d = stepDialog.stepDialog(self, self.scope)
            self.step_d.Show()
        
        self.time1.WantNotification.append(self.StatusBarUpdate)

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
            self.scope.cam.SetROI(0,0, self.scope.cam.GetCCDWidth(), self.scope.cam.GetCCDHeight())
            self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMROI, 'Set ROI')
            self.roi_on = False
        else:
            self.scope.cam.SetROI(self.scope.vp.selection_begin_x, self.scope.vp.selection_begin_y, self.scope.vp.selection_end_x, self.scope.vp.selection_end_y)
            self.mCam.SetLabel(wxID_SMIMAINFRAMEMCAMROI, 'Clear ROI')
            self.roi_on = True
            
        self.scope.cam.SetCOC()
        self.scope.cam.GetStatus()
        self.scope.pa.Prepare()
        self.scope.vp.SetDataStack(self.scope.pa.ds)
        self.scope.pa.start()
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
        self.int_sl.Destroy()
        self.piezo_sl.Destroy()
        self.seq_d.Destroy()
        #self.Close()
        self.Destroy()
        wx.Exit()
        
