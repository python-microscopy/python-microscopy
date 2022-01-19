#!/usr/bin/python

###############
# tempMonitor.py
#
# Copyright David Baddeley, 2012
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
################
import wx
from PYME.Acquire.Hardware.DigiData.DigiDataClient import getDDClient



class MyFrame(wx.MiniFrame):
    def __init__(self, *args, **kwds):
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        
        self.dd = kwds['dd']

        vsizer = wx.BoxSizer(wx.VERTICAL)
        panel = wx.Panel(self, -1)

        v2sizer = wx.BoxSizer(wx.VERTICAL)

        self.stTemp = wx.StaticText(panel, -1, '25.00 C')
        self.stTemp.SetFont(wx.Font(30, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        v2sizer.Add(self.stTemp)
        panel.SetSizerAndFit(v2sizer)

        vsizer.Add(panel)

        self.SetSizerAndFit(vsizer)

        self.onTimer()

        self.timer = wx.Timer(self, -1)
        self.Bind(wx.EVT_TIMER, self.onTimer)
        self.timer.Start(2000)



    def onTimer(self, ev = None):
        temp = self.dd.GetAIValue(1)*1000./2.**15 - 273.15

        self.stTemp.SetLabel('%3.2f C' % temp)


if __name__ == "__main__":
    dd = getDDClient()
    
    app = wx.PySimpleApp(0)
    wx.InitAllImageHandlers()
    fTaskMon = MyFrame(None, -1, "Temperature", dd = dd)
    app.SetTopWindow(fTaskMon)
    fTaskMon.Show()
    app.MainLoop()

