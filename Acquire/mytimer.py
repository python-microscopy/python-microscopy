#!/usr/bin/python

##################
# mytimer.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx

class mytimer(wx.Timer):
    def __init__(self):
        wx.Timer.__init__(self)
        self.WantNotification = []

    def Notify(self):
        for a in self.WantNotification:
                a()