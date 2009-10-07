#!/usr/bin/python

##################
# wxThread.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
import threading

class wxThread(threading.Thread):
    def run(self):
        self.app = wx.PySimpleApp()
        self.f1 = wx.Frame(None,-1,'ball_wx',wx.DefaultPosition,wx.Size(400,400))
        self.f1.Show()
        #self.f1.Iconize()
        self.app.MainLoop()