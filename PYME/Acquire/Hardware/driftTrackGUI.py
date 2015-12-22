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
from PYME.misc.wxPlotPanel import PlotPanel

class TrackerPlotPanel(PlotPanel):
    def __init__(self, parent, driftTracker, *args, **kwargs):
        self.dt = driftTracker
        PlotPanel.__init__(self, parent, *args, **kwargs)

        

    # add 5th suplot
    # replace 4th plot with offset and
    # new 5th subplot for z-pos (how calculated, z-nominal + dz?, remove offset)
    def draw(self):
        if self.IsShownOnScreen():
            if not hasattr( self, 'subplotx' ):
                    self.subplotx = self.figure.add_subplot( 411 )
                    self.subploty = self.figure.add_subplot( 412 )
                    self.subplotz = self.figure.add_subplot( 413 )
                    self.subplotc = self.figure.add_subplot( 414 )
    
            #try:
            t, dx, dy, dz, corr, poffset, dzcorr, nomPos, posInd, calPos, posDelta  = np.array(self.dt.history[-1000:]).T

            self.subplotx.cla()
            self.subplotx.plot(t, 88.0*dx, 'r')
            self.subplotx.set_ylabel('Delta x [nm]')
            self.subplotx.set_xlim(t.min(), t.max())
            
            self.subploty.cla()
            self.subploty.plot(t, 88*dy, 'g')
            self.subploty.set_ylabel('Delta y [nm]')
            self.subploty.set_xlim(t.min(), t.max())
            
            self.subplotz.cla()
            self.subplotz.plot(t, 1000*dz, 'b')
            self.subplotz.set_ylabel('Delta z [nm]')
            self.subplotz.set_xlim(t.min(), t.max())
            
            self.subplotc.cla()
            self.subplotc.plot(t, poffset, 'm')
            self.subplotc.set_ylabel('Offset')
            self.subplotc.set_xlim(t.min(), t.max())

    
            #except:
            #    pass
    
            #self.subplot.set_xlim(0, 512)
            #self.subplot.set_ylim(0, 256)
    
            self.canvas.draw()


# add controls for lastAdjustment
class DriftTrackingControl(wx.Panel):
    def __init__(self, parent, driftTracker, winid=-1):
        # begin wxGlade: MyFrame1.__init__
        #kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Panel.__init__(self, parent, winid)
        self.dt = driftTracker
        self.plotInterval = 10

        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.cbTrack = wx.CheckBox(self, -1, 'Track')
        hsizer.Add(self.cbTrack, 0, wx.ALL, 2) 
        self.cbTrack.Bind(wx.EVT_CHECKBOX, self.OnCBTrack)
        self.cbLock = wx.CheckBox(self, -1, 'Lock')
        self.cbLock.Bind(wx.EVT_CHECKBOX, self.OnCBLock)
        hsizer.Add(self.cbLock, 0, wx.ALL, 2)        
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.bSetPostion = wx.Button(self, -1, 'Set focus to current')
        hsizer.Add(self.bSetPostion, 0, wx.ALL, 2) 
        self.bSetPostion.Bind(wx.EVT_BUTTON, self.OnBSetPostion)
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, "Calibration:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.gCalib = wx.Gauge(self, -1, self.dt.NCalibStates + 1)
        hsizer.Add(self.gCalib, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2) 
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, "Tolerance [nm]:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tTolerance = wx.TextCtrl(self, -1, '%3.0f'% (1e3*self.dt.focusTolerance), size=[30,-1])
        hsizer.Add(self.tTolerance, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bSetTolerance = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        hsizer.Add(self.bSetTolerance, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2) 
        self.bSetTolerance.Bind(wx.EVT_BUTTON, self.OnBSetTolerance)
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
        self.stError = wx.StaticText(self, -1, 'Error:\n\n')
        hsizer.Add(self.stError, 0, wx.ALL, 2)        
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        self.trackPlot = TrackerPlotPanel(self, self.dt, size=[300, 400])
        
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
        
    def OnBSetTolerance(self, event):
        self.dt.focusTolerance = float(self.tTolerance.GetValue())/1e3
        
    def OnBSetMinDelay(self, event):
        self.dt.minDelay = int(self.tMinDelay.GetValue())

    def OnBSetPlotInterval(self, event):
        self.plotInterval = int(self.tPlotInterval.GetValue())
    
    def OnCBLock(self, event):
        self.dt.lockFocus = self.cbLock.GetValue()

    def refresh(self):
        try:
            self.gCalib.SetRange(self.dt.NCalibStates + 1)
            self.gCalib.SetValue(self.dt.calibState)
            t, dx, dy, dz, corr, poffset, dzcorr, nomPos, posInd, calPos, posDelta = self.dt.history[-1]
            self.stError.SetLabel('Error: x = %3.2f px\ny = %3.2f px\nz = %3.2f nm\noffset = %3.2f' % (dx, dy, dz*1000, poffset))
            if len(self.dt.history) % self.plotInterval == 0:
                self.trackPlot.draw()
            #self.trackPlot.draw()
        except AttributeError:
            pass
        except IndexError:
            pass
