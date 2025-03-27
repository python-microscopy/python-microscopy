#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import wx

import numpy as np
from PYME.contrib.wxPlotPanel import PlotPanel
from PYME.IO import MetaDataHandler
from PYME.DSView import dsviewer as dsviewer
import PYME.IO.image as im

import os

import logging
logger = logging.getLogger(__name__)

#TODO: these don't belong here!
def YesNo(parent, question, caption = 'Yes or no?'):
    dlg = wx.MessageDialog(parent, question, caption, wx.YES_NO | wx.ICON_QUESTION)
    result = dlg.ShowModal() == wx.ID_YES
    dlg.Destroy()
    return result
def Info(parent, message, caption = 'Insert program title'):
    dlg = wx.MessageDialog(parent, message, caption, wx.OK | wx.ICON_INFORMATION)
    dlg.ShowModal()
    dlg.Destroy()
def Warn(parent, message, caption = 'Warning!'):
    dlg = wx.MessageDialog(parent, message, caption, wx.OK | wx.ICON_WARNING)
    dlg.ShowModal()
    dlg.Destroy()

class TrackerPlotPanel(PlotPanel):
    def __init__(self, parent, driftTracker, *args, **kwargs):
        self.dt = driftTracker
        PlotPanel.__init__(self, parent, *args, **kwargs)    

    # add 5th suplot
    # replace 4th plot with offset and
    # new 5th subplot for z-pos (how calculated, z-nominal + dz?, remove offset)
    def draw(self):
        if self.IsShownOnScreen():
            if not hasattr( self, 'subplotxy' ):
                    self.subplotxy = self.figure.add_subplot( 411 )
                    self.subplotz = self.figure.add_subplot( 412 )
                    self.subploto = self.figure.add_subplot( 413 )
                    self.subplotc = self.figure.add_subplot(414)
    
            #try:
            t, dx, dy, dz, corr, corrmax, poffset, pos  = np.array(self.dt.get_history(1000)).T

            self.subplotxy.cla()
            self.subplotxy.plot(t, dx, 'r')
            self.subplotxy.plot(t, dy, 'g')
            self.subplotxy.set_ylabel('dx (r) dy (g) [pixels]')
            self.subplotxy.set_xlim(t.min(), t.max())
                        
            self.subplotz.cla()
            self.subplotz.plot(t, 1000*dz, 'b')
            self.subplotz.set_ylabel('dz [nm]')
            self.subplotz.set_xlim(t.min(), t.max())
            
            self.subploto.cla()
            self.subploto.plot(t, 1e3*poffset, 'm')
            self.subploto.set_ylabel('offs [nm]')
            self.subploto.set_xlim(t.min(), t.max())

            self.subplotc.cla()
            self.subplotc.plot(t, corr/corrmax, 'r')
            self.subplotc.set_ylabel('C/C_m')
            self.subplotc.set_xlim(t.min(), t.max())

    
            #except:
            #    pass
    
            #self.subplot.set_xlim(0, 512)
            #self.subplot.set_ylim(0, 256)
    
            self.canvas.draw()


resx = []
resy = []
resz = []

class CalculateZfactorDialog(wx.Dialog):
    def __init__(self):
        self.Zfactorfilename = ''
        wx.Dialog.__init__(self, None, -1, 'Calculate Z-factor')
        sizer1 = wx.BoxSizer(wx.VERTICAL)

        pan = wx.Panel(self, -1)
        vsizermain = wx.BoxSizer(wx.VERTICAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2.Add(wx.StaticText(pan, -1, '  Record a Z-stack with 101 slices & 50 nm step. Use the Z-stack to calculate the Z-factor'), 0, wx.ALL, 2)
        vsizer.Add(hsizer2)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.bSelect = wx.Button(pan, -1, 'Select')
        self.bSelect.Bind(wx.EVT_BUTTON, self.OnSelect)
        hsizer2.Add(self.bSelect, 0, wx.ALL, 2)
        self.bPlot = wx.Button(pan, -1, 'Plot')
        self.bPlot.Bind(wx.EVT_BUTTON, self.OnPlot)
        hsizer2.Add(self.bPlot, 0, wx.ALL, 2)
        vsizer.Add(hsizer2)

        self.textZstackFilename = wx.StaticText(pan, -1, 'Z-stack file:    no file selected')
        vsizer.Add(self.textZstackFilename, 0, wx.ALL, 2)

        vsizermain.Add(vsizer, 0, 0, 0)

        self.plotPan = ZFactorPlotPanel(pan, size=(1200,600))
        vsizermain.Add(self.plotPan, 1, wx.EXPAND, 0)

        pan.SetSizerAndFit(vsizermain)
        sizer1.Add(pan, 1,wx.EXPAND, 0)
        self.SetSizerAndFit(sizer1)

    def OnSelect(self, event):
        dlg = wx.FileDialog(self, message="Open a Z-stack Image...", defaultDir=os.getcwd(), 
                            defaultFile="", style=wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            self.Zfactorfilename = dlg.GetPath()

        dlg.Destroy()
        self.textZstackFilename.SetLabel('Z-stack file:   '+self.Zfactorfilename)

    
    def OnPlot(self, event):
        import PYMEcs.Analysis.offlineTracker as otrack

        ds = im.ImageStack(filename=self.Zfactorfilename)
        dataset = ds.data[:,:,:].squeeze()
        refim0 = dataset[:,:,10:91:4]
        calImages0, calFTs0, dz0, dzn0, mask0, X0, Y0 = otrack.genRef(refim0,normalised=False)

        del resx[:]
        del resy[:]
        del resz[:] # empty all these three lists every time before a new plot

        for i in range(dataset.shape[2]):
            image = dataset[:,:,i]
            driftx, drifty, driftz, cm, d = otrack.compare(calImages0, calFTs0, dz0, dzn0, 10, image, mask0, X0, Y0, deltaZ=0.2)
            resx.append(driftx)
            resy.append(drifty)
            resz.append(driftz)

        self.plotPan.draw()
        self.plotPan.Refresh()


class ZFactorPlotPanel(PlotPanel):

    def draw(self):
        dznm = 1e3*np.array(resz)
        dxnm = 110*np.array(resx)
        dynm = 110*np.array(resy)
        t = np.arange(dznm.shape[0])

        dzexp = dznm[50-4:50+5]
        dztheo = np.arange(-200,201,50)
        x = np.arange(-150,151,50)
        y = dznm[50-3:50+4]
        m, b = np.polyfit(x,y,1)
        Zfactor = 1.0/m

        if not hasattr( self, 'subplot' ):
                self.subplot1 = self.figure.add_subplot( 121 )
                self.subplot2 = self.figure.add_subplot( 122 )

        self.subplot1.cla()

        self.subplot1.scatter(t,-dznm,s=5)
        self.subplot1.plot(-dznm, label='z')
        self.subplot1.plot(-dxnm, label='x')
        self.subplot1.plot(-dynm, label='y')

        self.subplot1.grid()
        self.subplot1.legend()

        self.subplot2.cla()

        self.subplot2.plot(dztheo,dzexp,'-o')
        self.subplot2.plot(dztheo,1.0/Zfactor*dztheo,'--')
        font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
        self.subplot2.text(-50, 100, 'Z-factor = %3.1f' % Zfactor, fontdict=font)
        self.subplot2.grid()

        self.canvas.draw()



from PYME.DSView import overlays
import weakref
class DriftROIOverlay(overlays.Overlay):
    def __init__(self, driftTracker):
        self.dt = driftTracker
    
    def __call__(self, view, dc):
        if self.dt.sub_roi is not None:
            dc.SetPen(wx.Pen(colour=wx.CYAN, width=1))
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            x0, x1, y0, y1 = self.dt.sub_roi
            x0c, y0c = view.pixel_to_screen_coordinates(x0, y0)
            x1c, y1c = view.pixel_to_screen_coordinates(x1, y1)
            sX, sY = x1c-x0c, y1c-y0c
            dc.DrawRectangle(int(x0c), int(y0c), int(sX), int(sY))
            dc.SetPen(wx.NullPen)
        else:
            dc.SetBackground(wx.TRANSPARENT_BRUSH)
            dc.Clear()

class DriftTrackingControl(wx.Panel):
    def __init__(self, main_frame, driftTracker, winid=-1, showPlots=True):
        ''' This class provides a GUI for controlling the drift tracking system. 
        
        It should be initialised with a reference to the PYMEAcquire main frame, which will stand in as a parent while other GUI items are
        created. Note that the actual parent will be reassigned once the GUI tool panel is created using a Reparent() call.
        '''
        # begin wxGlade: MyFrame1.__init__
        #kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Panel.__init__(self, main_frame, winid)
        self.dt = driftTracker
        self.plotInterval = 10
        self.showPlots = showPlots

        # keep a reference to the main frame. Do this as a weakref to avoid circular references.
        # we need this to be able to access the view to get the current selection and to add overlays.
        self._main_frame = weakref.proxy(main_frame)
        self._view_overlay = None # dummy reference to the overlay so we only create it once

        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.cbTrack = wx.CheckBox(self, -1, 'Track')
        hsizer.Add(self.cbTrack, 0, wx.ALL, 2) 
        self.cbTrack.Bind(wx.EVT_CHECKBOX, self.OnCBTrack)
        self.cbLock = wx.CheckBox(self, -1, 'Lock')
        self.cbLock.Bind(wx.EVT_CHECKBOX, self.OnCBLock)
        hsizer.Add(self.cbLock, 0, wx.ALL, 2)
        #self.bSaveHist = wx.Button(self, -1, 'Save Hist')
        #hsizer.Add(self.bSaveHist, 0, wx.ALL, 2)
        #self.bSaveHist.Bind(wx.EVT_BUTTON, self.OnBSaveHist)
        self.cbLockActive = wx.CheckBox(self, -1, 'Lock Active')
        self.cbLockActive.Enable(False)
        hsizer.Add(self.cbLockActive, 0, wx.ALL, 2)        
        sizer_1.Add(hsizer, 0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.bSetPostion = wx.Button(self, -1, 'Set focus to current')
        hsizer.Add(self.bSetPostion, 0, wx.ALL, 2) 
        self.bSetPostion.Bind(wx.EVT_BUTTON, self.OnBSetPostion)
        #self.bSaveCalib = wx.Button(self, -1, 'Save Cal')
        #hsizer.Add(self.bSaveCalib, 0, wx.ALL, 2)
        #self.bSaveCalib.Bind(wx.EVT_BUTTON, self.OnBSaveCalib)
        sizer_1.Add(hsizer, 0, wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.tbSubROI = wx.ToggleButton(self, -1, 'Restrict to sub-ROI')
        hsizer.Add(self.tbSubROI, 0, wx.ALL, 2)
        self.tbSubROI.Bind(wx.EVT_TOGGLEBUTTON, self.OnTBToggleSubROI)
        #self.bSaveCalib = wx.Button(self, -1, 'Save Cal')
        #hsizer.Add(self.bSaveCalib, 0, wx.ALL, 2)
        #self.bSaveCalib.Bind(wx.EVT_BUTTON, self.OnBSaveCalib)
        sizer_1.Add(hsizer, 0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, "Calibration:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.gCalib = wx.Gauge(self, -1, 11)
        hsizer.Add(self.gCalib, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2) 
        sizer_1.Add(hsizer, 0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, "Tolerance [nm]:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tTolerance = wx.TextCtrl(self, -1, '%3.0f'% (1e3*self.dt.get_focus_tolerance()), size=[30,-1])
        hsizer.Add(self.tTolerance, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bSetTolerance = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        hsizer.Add(self.bSetTolerance, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2) 
        self.bSetTolerance.Bind(wx.EVT_BUTTON, self.OnBSetTolerance)
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, "Z increment [nm]:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tdeltaZ = wx.TextCtrl(self, -1, '%3.0f'% (1e3*self.dt.get_delta_Z()), size=[30,-1])
        hsizer.Add(self.tdeltaZ, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bSetdeltaZ = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        hsizer.Add(self.bSetdeltaZ, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2) 
        self.bSetdeltaZ.Bind(wx.EVT_BUTTON, self.OnBSetdeltaZ)
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, "Stack halfsize:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tHalfsize = wx.TextCtrl(self, -1, '%3.0f'% (self.dt.get_stack_halfsize()), size=[30,-1])
        hsizer.Add(self.tHalfsize, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bSetHalfsize = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        hsizer.Add(self.bSetHalfsize, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2) 
        self.bSetHalfsize.Bind(wx.EVT_BUTTON, self.OnBSetHalfsize)
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)

        # hsizer = wx.BoxSizer(wx.HORIZONTAL)
        # hsizer.Add(wx.StaticText(self, -1, "Z-factor:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        # self.tZfactor = wx.TextCtrl(self, -1, '%3.1f'% self.dt.Zfactor, size=[30,-1])
        # hsizer.Add(self.tZfactor, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        # self.bSetZfactor = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        # hsizer.Add(self.bSetZfactor, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        # self.bSetZfactor.Bind(wx.EVT_BUTTON, self.OnBSetZfactor)
        # self.bCalcZfactor = wx.Button(self, -1, 'Calculate Z-factor', style=wx.BU_EXACTFIT)
        # hsizer.Add(self.bCalcZfactor, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        # self.bCalcZfactor.Bind(wx.EVT_BUTTON, self.OnBCalculateZfactor)
        # sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        # hsizer = wx.BoxSizer(wx.HORIZONTAL)
        # hsizer.Add(wx.StaticText(self, -1, "feedback delay [frames]:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        # self.tMinDelay = wx.TextCtrl(self, -1, '%d' % (self.dt.minDelay), size=[30,-1])
        # hsizer.Add(self.tMinDelay, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        # self.bSetMinDelay = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        # hsizer.Add(self.bSetMinDelay, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        # self.bSetMinDelay.Bind(wx.EVT_BUTTON, self.OnBSetMinDelay)
        # sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        #
        # hsizer = wx.BoxSizer(wx.HORIZONTAL)
        # hsizer.Add(wx.StaticText(self, -1, "Plot Interval [frames]:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        # self.tPlotInterval = wx.TextCtrl(self, -1, '%d' % (self.plotInterval), size=[30,-1])
        # hsizer.Add(self.tPlotInterval, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        # self.bSetPlotInterval = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        # hsizer.Add(self.bSetPlotInterval, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        # self.bSetPlotInterval.Bind(wx.EVT_BUTTON, self.OnBSetPlotInterval)
        # sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        #
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.stError = wx.StaticText(self, -1, 'Error:\n\n', size=[200,-1])
        cfont = self.stError.GetFont()
        font = wx.Font(cfont.GetPointSize(), wx.TELETYPE, wx.NORMAL, wx.NORMAL)
        self.stError.SetFont(font)
        hsizer.Add(self.stError, 0, wx.ALL, 2)        
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        if self.showPlots:
            self.trackPlot = TrackerPlotPanel(self, self.dt, size=[300, 500])
            
            #hsizer.Add(self.stError, 0, wx.ALL, 2)        
            sizer_1.Add(self.trackPlot,0, wx.EXPAND, 0)
        
        self.SetAutoLayout(1)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        sizer_1.SetSizeHints(self)
        self.Layout()
        # end wxGlade

    def OnCBTrack(self, event):
        #print self.cbTrack.GetValue()
        if self.cbTrack.GetValue():
            self.dt.register()
        else:
            self.dt.deregister()
            
    def OnBSetPostion(self, event):
        self.dt.reCalibrate()

    def OnTBToggleSubROI(self, event):
        self.toggle_subroi(self.tbSubROI.GetValue())
    
    def toggle_subroi(self, new_state=True):
        ''' Turn sub-ROI tracking on or off, using the current selection in the live image display'''
        if new_state:
            x0, x1, y0, y1, _, _ = self._main_frame.view.do.sorted_selection
            self.dt.set_subroi((x0, x1, y0, y1))
        else:
            self.dt.set_subroi(None)

        if self._view_overlay is None:
            self._view_overlay = self._main_frame.view.add_overlay(DriftROIOverlay(self.dt), 'Drift tracking Sub-ROI')
        
    def OnBSaveCalib(self, event):
        if not hasattr(self.dt, 'calibState') or (self.dt.calibState < self.dt.NCalibStates):
            Warn(self,"not calibrated")
        else:
            self.showCalImages()

    def showCalImages(self):
        import numpy as np
        import time
        ds2 = self.dt.refImages

        #metadata handling
        mdh = MetaDataHandler.NestedClassMDHandler()
        mdh.setEntry('StartTime', time.time())
        mdh.setEntry('AcquisitionType', 'Stack')

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(mdh)
        mdh.setEntry('CalibrationPositions',self.dt.calPositions)

        im = dsviewer.ImageStack(data = ds2, mdh = mdh, titleStub='Unsaved Image')
        if not im.mode == 'graph':
            im.mode = 'lite'

        #print im.mode
        dvf = dsviewer.DSViewFrame(im, mode= im.mode, size=(500, 500))
        dvf.SetSize((500,500))
        dvf.Show()
        
    def OnBSaveHist(self, event):
        if not hasattr(self.dt, 'history') or (len(self.dt.history) <= 0):
            Warn(self,"no history")
        else:
            dlg = wx.FileDialog(self, message="Save file as...",  
                            defaultFile='history.txt', wildcard='txt File (*.txt)|*.txt', style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

            if dlg.ShowModal() == wx.ID_OK:
                historyfn = dlg.GetPath()
                np.savetxt(historyfn, self.dt.history)
                Info(self,"history saved")


    def OnBSetTolerance(self, event):
        self.dt.set_focus_tolerance(float(self.tTolerance.GetValue())/1e3)

    def OnBSetdeltaZ(self, event):
        self.dt.set_delta_Z(float(self.tdeltaZ.GetValue())/1e3)

    def OnBSetHalfsize(self, event):
        self.dt.set_stack_halfsize(int(self.tHalfsize.GetValue()))

    def OnBSetZfactor(self, event):
        self.dt.Zfactor = float(self.tZfactor.GetValue())

    def OnBCalculateZfactor(self, event):
        dlg = CalculateZfactorDialog()
        ret = dlg.ShowModal()
        dlg.Destroy()

    def OnBSetMinDelay(self, event):
        self.dt.minDelay = int(self.tMinDelay.GetValue())

    def OnBSetPlotInterval(self, event):
        self.plotInterval = int(self.tPlotInterval.GetValue())
    
    def OnCBLock(self, event):
        self.dt.set_focus_lock(self.cbLock.GetValue())

    def refresh(self):
        try:
            calibState, NStates = self.dt.get_calibration_state()
            self.gCalib.SetRange(NStates + 1)
            self.gCalib.SetValue(calibState)

            try:
                t, dx, dy, dz, corr, corrmax,poffset,pos = self.dt.get_history(1)[-1]
                self.stError.SetLabel(("Error: x = %s nm y = %s nm\n" +
                                      "z = %s nm noffs = %s nm c/cm = %4.2f") %
                                      ("{:>+3.2f}".format(dx), "{:>+3.2f}".format(dy),
                                       "{:>+6.1f}".format(1e3*dz), "{:>+6.1f}".format(1e3*poffset),
                                       corr/corrmax))

            except IndexError:
                pass

            self.cbLock.SetValue(self.dt.get_focus_lock())
            self.cbTrack.SetValue(self.dt.is_tracking())
            self.cbLockActive.SetValue(self.dt.lockActive)
            
            if (len(self.dt.get_history(0)) > 0) and (len(self.dt.get_history(0)) % self.plotInterval == 0) and self.showPlots:
                self.trackPlot.draw()
        except AttributeError:
            logger.exception('error in refresh')
            pass
        except IndexError:
            logger.exception('error in refresh')
            pass
