#!/usr/bin/python
##################
# shell.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx.py.shell

def Plug(dsviewer):
    sh = wx.py.shell.Shell(id=-1,
              parent=dsviewer, pos=wx.Point(0, 0), size=wx.Size(618, 451), style=0, locals=dsviewer.__dict__,
              introText='note that help, license etc below is for Python, not PYME\n\n')

    sh.Execute('from pylab import *')
    sh.Execute('from PYME.DSView import View3D')

    dsviewer.AddPage(page=sh, select=False, caption='Console')

    dsviewer.sh = sh