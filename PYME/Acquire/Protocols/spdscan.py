#!/usr/bin/python

##################
# prebleach671.py
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

#import all the stuff to make this work
from PYME.Acquire.protocol import *
import numpy
import wx

#from PYME.Acquire.pointScanner import PointScanner
#from PYME.contrib.wxPlotPanel import PlotPanel
##from PYME.Analysis import ofind
#
#ps = PointScanner(scope.piezos[1], scope.piezos[2], scope, pixels = [10,10], pixelsize=numpy.array([0.03, .015]), dwelltime=3, avg=False, evtLog = True)
#
#class SFGenPlotPanel(PlotPanel):
#    def draw(self):
#        if not hasattr( self, 'subplot' ):
#                self.subplot = self.figure.add_subplot( 111 )
#
#        #ofd = ofind.ObjectIdentifier(scope.frameWrangler.currentFrame.astype('f').squeeze().T)
#        #ofd.FindObjects(70, 0, splitter=True)
#
#        #print len(ofd)
#        ox = scope.frameWrangler.currentFrame.shape[0]*numpy.array([0,1,1,0,0])*.07
#        oy = scope.frameWrangler.currentFrame.shape[1]*numpy.array([0,0,1,1,0])*.07
#
#        if scope.splitting =='up_down':
#            oy *= .5
#
#        X = (((ps.xp - ps.currPos[0])*1e3)[:, None]*numpy.ones(ps.yp.shape)[None, :]).ravel()
#        Y = (((ps.yp - ps.currPos[1])*1e3)[None, :]*numpy.ones(ps.xp.shape)[:, None]).ravel()
#
#        self.subplot.cla()
#
#        for i in xrange(X.size):
#            self.subplot.plot(ox + X[i], oy + Y[i])#, c=i)
#
#        #self.subplot.set_xlim(0, 512)
#        #self.subplot.set_ylim(0, 256)
#
#        self.canvas.draw()
#
#class ShiftfieldPreviewDialog(wx.Dialog):
#    def __init__(self):
#        wx.Dialog.__init__(self, None, -1, 'Shiftfield Settings')
#
#        sizer1 = wx.BoxSizer(wx.VERTICAL)
#
#        pan = wx.Panel(self, -1)
#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#        vsizer = wx.BoxSizer(wx.VERTICAL)
#
#        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
#        hsizer2.Add(wx.StaticText(pan, -1, 'Step Size x[mm]:'), 0, wx.ALL, 2)
#        self.tPixelSizeX = wx.TextCtrl(pan, -1, value='%3.4f' % ps.pixelsize[0])
#        hsizer2.Add(self.tPixelSizeX, 0, wx.ALL, 2)
#        vsizer.Add(hsizer2)
#
#        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
#        hsizer2.Add(wx.StaticText(pan, -1, 'Step Size y[mm]:'), 0, wx.ALL, 2)
#        self.tPixelSizeY = wx.TextCtrl(pan, -1, value='%3.4f' % ps.pixelsize[1])
#        hsizer2.Add(self.tPixelSizeY, 0, wx.ALL, 2)
#        vsizer.Add(hsizer2)
#
#        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
#        hsizer2.Add(wx.StaticText(pan, -1, '# x steps:'), 0, wx.ALL, 2)
#        self.tXPixels = wx.TextCtrl(pan, -1, value='%d' % ps.pixels[0])
#        hsizer2.Add(self.tXPixels, 0, wx.ALL, 2)
#        vsizer.Add(hsizer2)
#
#        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
#        hsizer2.Add(wx.StaticText(pan, -1, '# y steps:'), 0, wx.ALL, 2)
#        self.tYPixels = wx.TextCtrl(pan, -1, value='%d' % ps.pixels[1])
#        hsizer2.Add(self.tYPixels, 0, wx.ALL, 2)
#        vsizer.Add(hsizer2)
#
#        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
#        self.bTest = wx.Button(pan, -1, 'Test')
#        self.bTest.Bind(wx.EVT_BUTTON, self.OnTest)
#        hsizer2.Add(self.bTest, 0, wx.ALL, 2)
#        self.bGo = wx.Button(pan, -1, 'Go')
#        self.bGo.Bind(wx.EVT_BUTTON, self.OnGo)
#        hsizer2.Add(self.bGo, 0, wx.ALL, 2)
#        vsizer.Add(hsizer2)
#
#        hsizer.Add(vsizer, 0, 0, 0)
#
#        self.plotPan = SFGenPlotPanel(pan, size=(400,400))
#        hsizer.Add(self.plotPan, 1, wx.EXPAND, 0)
#
#        pan.SetSizerAndFit(hsizer)
#        sizer1.Add(pan, 1,wx.EXPAND, 0)
#        self.SetSizerAndFit(sizer1)
#
#    def updatePointScanner(self):
#        ps.pixelsize[0] = float(self.tPixelSizeX.GetValue())
#        ps.pixelsize[1] = float(self.tPixelSizeY.GetValue())
#        ps.pixels[0] = int(self.tXPixels.GetValue())
#        ps.pixels[1] = int(self.tYPixels.GetValue())
#
#    def OnTest(self, event):
#        self.updatePointScanner()
#        #print ps.pixels
#        ps.genCoords()
#        #print ps.nx
#        self.plotPan.draw()
#        self.plotPan.Refresh()
#
#    def OnGo(self, event):
#        self.updatePointScanner()
#        ps.genCoords()
#        self.EndModal(True)


def stop():
    ps.stop()
    MainFrame.pan_spool.OnBStopSpoolingButton(None)


stopTask = T(1000, stop)

#def ShowSFDialog():
#    ps.genCoords()
#    dlg = ShiftfieldPreviewDialog()
#    ret = dlg.ShowModal()
#    dlg.Destroy()
#
#    #stop after one full scan
#    stopTask.when = 40 + 3*ps.imsize
#    print stopTask.when





#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
#T(-1, ShowSFDialog),
#T(-1, SetCameraShutter,False),
#T(11, SetCameraShutter, True),
#T(12, ps.start),
#T(30, MainFrame.pan_spool.OnBAnalyse, None),
T(0, scope.stage.MoveTo,1, 13),
stopTask,
#T(maxint, ps.stop),
]

#optional - metadata entries
metaData = [
#('Protocol.DarkFrameRange', (0, 10)),
#('Protocol.DataStartsAt', 12)
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 100, metaData, randomise = False)
