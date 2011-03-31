#!/usr/bin/python
##################
# setBGImage.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################



from pylab import *
from PYME.DSView.dsviewer_npy import View3D
vp.view.showTracks = False
vp.view.showPoints = False
class fakeVp:
    pass

fvp = fakeVp()
fvp.do = vp.do
class fakeVp:
    def update(self, ev=None):
        glCanvas.setBackgroundImage((255*vp.do.Gains[0]*(vp.do.ds[:,:,vp.do.zp] - vp.do.Offs[0])).astype('uint8'))

fvp = fakeVp()
fvp.do = vp.do
fvp.update()
from PYME.DSView.Modules import playback
Traceback (most recent call last):
  File "<input>", line 1, in <module>
ImportError: No module named Modules
from PYME.DSView.modules import playback
pplay = playback.PlayPanel(MainWindow, fvp)
import wx.lib.agw.aui as aui
pinfo1 = aui.AuiPaneInfo().Name("plPanel").Top().Caption('Playback').DestroyOnClose(True).CloseButton(False)
pinfo1 = aui.AuiPaneInfo().Name("plPanel").Bottom().Caption('Playback').DestroyOnClose(True).CloseButton(False)
MainWindow._mgr.AddPane(pplay, pinfo1)
True
MainWindow._mgr.Update()
fvp.update()
vp.do.zp
589
vp.do.zp
317
fvp.update()
class fakeVp:
    def __init__(self, do):
        self.do = do

class fakeVp:
    def __init__(self, do):
        self.do = do
    def update(self, ev=None):
        glCanvas.setBackgroundImage((255*self.do.Gains[0]*(self.do.ds[:,:,self.do.zp] - self.do.Offs[0])).astype('uint8'))

fvp = fakeVp(vp.do)
fvp.update()
fvp.update()
class fakeVp:
    def __init__(self, do):
        self.do = do
    def update(self, ev=None):
        glCanvas.setBackgroundImage((255*self.do.Gains[0]*(self.do.ds[:,:,self.do.zp] - self.do.Offs[0])).astype('uint8'))
        glCanvas.Refresh()

fvp = fakeVp(vp.do)
fvp.update()
pplay.vp = fvp

