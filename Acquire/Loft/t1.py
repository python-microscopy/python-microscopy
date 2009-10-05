#!/usr/bin/python

##################
# t1.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import sys
sys.path.append(".")
import example
ds = example.CDataStack.OpenFromFile("test.kdf")

do = example.CDisplayOpts()
do.setDisp1Chan(0)

rend = example.CLUT_RGBRenderer()
rend.setDispOpts(do)

import wx

im = wx.EmptyImage(640,480)
