#!/usr/bin/python

##################
# prevtimer.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx

class myTimer(wx.Timer):
    def Notify(self):
        if (cam.ExpReady()):
            cam.ExtractColor(ds.getCurrentChannelSlice(0), 1)
            cam.ExtractColor(ds.getCurrentChannelSlice(1), 2)
            cam.ExtractColor(ds.getCurrentChannelSlice(2), 3)
            cam.ExtractColor(ds.getCurrentChannelSlice(3), 4)
            fr.vp.imagepanel.Refresh()
