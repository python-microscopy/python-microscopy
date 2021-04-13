#!/usr/bin/python
##################
# playback.py
#
# Copyright David Baddeley, 2011
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

#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
import wx
from PYME import resources
from PYME.ui.mytimer import mytimer


class PlayPanel(wx.Panel):
    def __init__(self, parent, dsviewer):
        wx.Panel.__init__(self,parent, -1)
        dirname = os.path.dirname(__file__)

        self.do = dsviewer.do
        self.dsviewer = dsviewer

        self.bmStartSeek = wx.Bitmap(resources.getIconPath('media-skip-backward.png'))
        self.bmPlay = wx.Bitmap(resources.getIconPath('media-playback-start.png'))
        self.bmPause = wx.Bitmap(resources.getIconPath('media-playback-pause.png'))

        self.mode = 'HORIZ'
        self.moving= False

        #timer for playback
        self.tPlay = mytimer()
        self.tPlay.WantNotification.append(self.OnFrame)

        self.do.WantChangeNotification.append(self.update)

        self.genContents(self.mode)

        self.Bind(wx.EVT_SIZE, self.OnSize)

    def genContents(self, mode='VERT'):
        self.DestroyChildren() #clean out existing gui

        self.slPlayPos = wx.Slider(self, -1, 0, 0, 100, style=wx.SL_HORIZONTAL)
        self.slPlayPos.Bind(wx.EVT_SCROLL_THUMBRELEASE, self.OnPlayPosChanged)
        self.slPlayPos.Bind(wx.EVT_SCROLL_LINEUP, self.OnPlayPosChanged)
        self.slPlayPos.Bind(wx.EVT_SCROLL_LINEDOWN, self.OnPlayPosChanged)

        self.bSeekStart = wx.BitmapButton(self, -1, self.bmStartSeek)
        self.bSeekStart.Bind(wx.EVT_BUTTON, self.OnSeekStart)

        self.bPlay = wx.BitmapButton(self, -1, self.bmPlay)
        self.bPlay.Bind(wx.EVT_BUTTON, self.OnPlay)

        self.bGoto = wx.Button(self, -1, 'GOTO', style=wx.BU_EXACTFIT)
        self.bGoto.Bind(wx.EVT_BUTTON, self.OnGoto)

        self.slPlaySpeed = wx.Slider(self, -1, 5, 1, 50, style=wx.SL_HORIZONTAL)
        self.slPlaySpeed.Bind(wx.EVT_SCROLL_THUMBRELEASE, self.OnPlaySpeedChanged)

        if self.mode == 'VERT':
            vsizer = wx.BoxSizer(wx.VERTICAL)

            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, 'Pos:'), 0,wx.ALIGN_CENTER_VERTICAL|wx.LEFT,0)
            hsizer.Add(self.slPlayPos, 1,wx.ALIGN_CENTER_VERTICAL)

            vsizer.Add(hsizer, 0,wx.ALL|wx.EXPAND, 0)
            
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(self.bSeekStart, 0,wx.ALIGN_CENTER_VERTICAL,0)
            hsizer.Add(self.bPlay, 0,wx.ALIGN_CENTER_VERTICAL,0)
            hsizer.Add(self.bGoto, 0,wx.ALIGN_CENTER_VERTICAL,0)
            hsizer.Add(wx.StaticText(self, -1, 'FPS:'), 0,wx.ALIGN_CENTER_VERTICAL|wx.LEFT,4)
            hsizer.Add(self.slPlaySpeed, 1,wx.ALIGN_CENTER_VERTICAL)

            vsizer.Add(hsizer, 0,wx.TOP|wx.BOTTOM|wx.EXPAND, 4)
            self.SetSizerAndFit(vsizer)
        else: #all on one line
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(self.bSeekStart, 0,wx.ALIGN_CENTER_VERTICAL)
            hsizer.Add(self.bPlay, 0,wx.ALIGN_CENTER_VERTICAL|wx.RIGHT,4)
            hsizer.Add(self.bGoto, 0,wx.ALIGN_CENTER_VERTICAL|wx.RIGHT,4)
            hsizer.Add(wx.StaticText(self, -1, 'Pos:'), 0,wx.ALIGN_CENTER_VERTICAL|wx.LEFT,0)
            hsizer.Add(self.slPlayPos, 3,wx.ALIGN_CENTER_VERTICAL)

            hsizer.Add(wx.StaticText(self, -1, 'FPS:'), 0,wx.ALIGN_CENTER_VERTICAL|wx.LEFT,4)
            hsizer.Add(self.slPlaySpeed, 1,wx.ALIGN_CENTER_VERTICAL)

            self.SetSizerAndFit(hsizer)

    def OnSize(self, event):
        if self.mode == 'VERT' and event.GetSize()[0] > 300:
            self.mode = 'HORIZ'
            self.genContents(self.mode)
        elif self.mode == 'HORIZ' and event.GetSize()[0] < 300:
            self.mode = 'VERT'
            self.genContents(self.mode)

        event.Skip()

    def OnPlay(self, event):
        if not self.tPlay.IsRunning():
            self.tPlay.Start(1000./self.slPlaySpeed.GetValue())
            self.bPlay.SetBitmapLabel(self.bmPause)
        else:
            self.tPlay.Stop()
            self.bPlay.SetBitmapLabel(self.bmPlay)

    def OnFrame(self):
        if self.do.zp >= (self.do.ds.shape[2]-1):
            self.do.zp = 0
        else:
            self.do.zp +=1

    def OnSeekStart(self, event):
        self.do.zp = 0
        #self.vp.update()

    def OnPlaySpeedChanged(self, event):
        if self.tPlay.IsRunning():
            self.tPlay.Stop()
            self.tPlay.Start(1000./self.slPlaySpeed.GetValue())

    def OnPlayPosChanged(self, event):
        self.moving = True #hide refreshes that we generate ourselves
        self.do.zp = int((self.do.ds.shape[2]-1)*self.slPlayPos.GetValue()/100.)
        self.moving = False
        #self.vp.update()

    def OnGoto(self, event):
        dialog = wx.TextEntryDialog(self, 'Goto frame:','GOTO','0')
        ans = dialog.ShowModal()
        if ans == wx.ID_OK:
            val = dialog.GetValue()
            try:
                frame = int(val)
            except(ValueError):
                # Not an integer
                raise ValueError('Please enter a valid frame number.')
            
            frame = max(0, frame)
            frame = min(frame, self.do.ds.shape[2]-1)
            self.do.zp = frame

    def update(self):
        #print 'foo'
        if not self.moving:
            self.slPlayPos.SetValue((100*self.do.zp)/max(1,self.do.ds.shape[2]-1))

            if not self.tPlay.IsRunning():
                self.dsviewer.optionspanel.RefreshHists()



class player:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        #self.vp = dsviewer.vp

        dsviewer.paneHooks.append(self.GenPlayPanel)
        #dsviewer.updateHooks.append(self.update)
        
    def GenPlayPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Playback", pinned = True)

        self.playPan = PlayPanel(item, self.dsviewer)

        

        item.AddNewElement(self.playPan)
        _pnl.AddPane(item)

    #def update(self):
    #    self.playPan.update()

    

def Plug(dsviewer):
    dsviewer.player = player(dsviewer)
