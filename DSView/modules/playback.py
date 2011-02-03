#!/usr/bin/python
##################
# playback.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import wx
from PYME.Acquire.mytimer import mytimer
import PYME.misc.autoFoldPanel as afp

class player:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.vp = dsviewer.vp

        #timer for playback
        self.tPlay = mytimer()
        self.tPlay.WantNotification.append(self.OnFrame)
        
    def GenPlayPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Playback", pinned = True)

        pan = wx.Panel(item, -1)

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Pos:'), 0,wx.ALIGN_CENTER_VERTICAL|wx.LEFT,0)

        self.slPlayPos = wx.Slider(pan, -1, 0, 0, 100, style=wx.SL_HORIZONTAL)
        self.slPlayPos.Bind(wx.EVT_SCROLL_CHANGED, self.OnPlayPosChanged)
        hsizer.Add(self.slPlayPos, 1,wx.ALIGN_CENTER_VERTICAL)

        vsizer.Add(hsizer, 0,wx.ALL|wx.EXPAND, 0)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        import os

        dirname = os.path.dirname(__file__)

        self.bSeekStart = wx.BitmapButton(pan, -1, wx.Bitmap(os.path.join(dirname, '../icons/media-skip-backward.png')))
        hsizer.Add(self.bSeekStart, 0,wx.ALIGN_CENTER_VERTICAL,0)
        self.bSeekStart.Bind(wx.EVT_BUTTON, self.OnSeekStart)

        self.bmPlay = wx.Bitmap(os.path.join(dirname,'../icons/media-playback-start.png'))
        self.bmPause = wx.Bitmap(os.path.join(dirname,'../icons/media-playback-pause.png'))
        self.bPlay = wx.BitmapButton(pan, -1, self.bmPlay)
        self.bPlay.Bind(wx.EVT_BUTTON, self.OnPlay)
        hsizer.Add(self.bPlay, 0,wx.ALIGN_CENTER_VERTICAL,0)

#        self.bSeekEnd = wx.BitmapButton(pan, -1, wx.Bitmap('icons/media-skip-forward.png'))
#        hsizer.Add(self.bSeekEnd, 0,wx.ALIGN_CENTER_VERTICAL,0)

        hsizer.Add(wx.StaticText(pan, -1, 'FPS:'), 0,wx.ALIGN_CENTER_VERTICAL|wx.LEFT,4)

        self.slPlaySpeed = wx.Slider(pan, -1, 5, 1, 50, style=wx.SL_HORIZONTAL)
        self.slPlaySpeed.Bind(wx.EVT_SCROLL_CHANGED, self.OnPlaySpeedChanged)
        hsizer.Add(self.slPlaySpeed, 1,wx.ALIGN_CENTER_VERTICAL)

        vsizer.Add(hsizer, 0,wx.TOP|wx.BOTTOM|wx.EXPAND, 4)
        pan.SetSizer(vsizer)
        vsizer.Fit(pan)

        item.AddNewElement(pan)
        _pnl.AddPane(item)

    def OnPlay(self, event):
        if not self.tPlay.IsRunning():
            self.tPlay.Start(1000./self.slPlaySpeed.GetValue())
            self.bPlay.SetBitmapLabel(self.bmPause)
        else:
            self.tPlay.Stop()
            self.bPlay.SetBitmapLabel(self.bmPlay)

    def OnFrame(self):
        self.vp.do.zp +=1
        if self.vp.do.zp >= self.vp.do.ds.shape[2]:
            self.vp.do.zp = 0

        self.dsviewer.update()

    def OnSeekStart(self, event):
        self.vp.do.zp = 0
        self.dsviewer.update()

    def OnPlaySpeedChanged(self, event):
        if self.tPlay.IsRunning():
            self.tPlay.Stop()
            self.tPlay.Start(1000./self.slPlaySpeed.GetValue())

    def OnPlayPosChanged(self, event):
        self.vp.do.zp = int((self.vp.do.ds.shape[2]-1)*self.slPlayPos.GetValue()/100.)
        self.dsviewer.update()

    def update(self):
        self.slPlayPos.SetValue((100*self.vp.do.zp)/max(1,self.vp.do.ds.shape[2]-1))

        if not self.tPlay.IsRunning():
            self.vp.optionspanel.RefreshHists()