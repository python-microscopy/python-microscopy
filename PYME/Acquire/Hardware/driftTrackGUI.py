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
            self.subplotxy.plot(t, 88.0*dx, 'r')
            self.subplotxy.plot(t, 88*dy, 'g')
            self.subplotxy.set_ylabel('dx (r) dy (g) [nm]')
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


# add controls for lastAdjustment
class DriftTrackingControl(wx.Panel):
    def __init__(self, parent, driftTracker, winid=-1, showPlots=True):
        # begin wxGlade: MyFrame1.__init__
        #kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Panel.__init__(self, parent, winid)
        self.dt = driftTracker
        self.plotInterval = 10
        self.showPlots = showPlots

        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.cbTrack = wx.CheckBox(self, -1, 'Track')
        hsizer.Add(self.cbTrack, 0, wx.ALL, 2) 
        self.cbTrack.Bind(wx.EVT_CHECKBOX, self.OnCBTrack)
        self.cbLock = wx.CheckBox(self, -1, 'Lock')
        self.cbLock.Bind(wx.EVT_CHECKBOX, self.OnCBLock)
        hsizer.Add(self.cbLock, 0, wx.ALL, 2)
        self.bSaveHist = wx.Button(self, -1, 'Save Hist')
        hsizer.Add(self.bSaveHist, 0, wx.ALL, 2) 
        self.bSaveHist.Bind(wx.EVT_BUTTON, self.OnBSaveHist)        
        self.cbLockActive = wx.CheckBox(self, -1, 'Lock Active')
        self.cbLockActive.Enable(False)
        hsizer.Add(self.cbLockActive, 0, wx.ALL, 2)        
        sizer_1.Add(hsizer, 0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.bSetPostion = wx.Button(self, -1, 'Set focus to current')
        hsizer.Add(self.bSetPostion, 0, wx.ALL, 2) 
        self.bSetPostion.Bind(wx.EVT_BUTTON, self.OnBSetPostion)
        self.bSaveCalib = wx.Button(self, -1, 'Save Cal')
        hsizer.Add(self.bSaveCalib, 0, wx.ALL, 2) 
        self.bSaveCalib.Bind(wx.EVT_BUTTON, self.OnBSaveCalib)
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
        hsizer.Add(wx.StaticText(self, -1, "Z-factor:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tZfactor = wx.TextCtrl(self, -1, '%3.1f'% self.dt.Zfactor, size=[30,-1])
        hsizer.Add(self.tZfactor, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bSetZfactor = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        hsizer.Add(self.bSetZfactor, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2) 
        self.bSetZfactor.Bind(wx.EVT_BUTTON, self.OnBSetZfactor)
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, "feedback delay [frames]:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tMinDelay = wx.TextCtrl(self, -1, '%d' % (self.dt.minDelay), size=[30,-1])
        hsizer.Add(self.tMinDelay, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bSetMinDelay = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        hsizer.Add(self.bSetMinDelay, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2) 
        self.bSetMinDelay.Bind(wx.EVT_BUTTON, self.OnBSetMinDelay)
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, "Plot Interval [frames]:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tPlotInterval = wx.TextCtrl(self, -1, '%d' % (self.plotInterval), size=[30,-1])
        hsizer.Add(self.tPlotInterval, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bSetPlotInterval = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        hsizer.Add(self.bSetPlotInterval, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2) 
        self.bSetPlotInterval.Bind(wx.EVT_BUTTON, self.OnBSetPlotInterval)
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        
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

    def OnBSetZfactor(self, event):
        self.dt.Zfactor = float(self.tZfactor.GetValue())
        
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
            t, dx, dy, dz, corr, corrmax,poffset,pos = self.dt.get_history(1)[-1]
            self.stError.SetLabel(("Error: x = %s nm y = %s nm\n" +
                                  "z = %s nm noffs = %s nm c/cm = %4.2f") %
                                  ("{:>+6.1f}".format(88.0*dx), "{:>+6.1f}".format(88.0*dy),
                                   "{:>+6.1f}".format(1e3*dz), "{:>+6.1f}".format(1e3*poffset),
                                   corr/corrmax))
            self.cbLockActive.SetValue(self.dt.lockActive)
            if (len(self.dt.get_history(0)) % self.plotInterval == 0) and self.showPlots:
                self.trackPlot.draw()
        except AttributeError:
            print "AttrErr"
            pass
        except IndexError:
            print "IndexErr"
            pass
