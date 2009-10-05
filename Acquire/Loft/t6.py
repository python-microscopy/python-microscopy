#!/usr/bin/python

##################
# t6.py
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
import vfr
import previewaquisator

example.CShutterControl.init()
cam = example.CCamera()
cam.Init()

class chaninfo:
    names = ['bw']
    cols = [2|4|8|16]
    hw = [example.CShutterControl.CH1]

pa = previewaquisator.PreviewAquisator(chaninfo,cam)

pa.Prepare()

fr = vfr.ViewFrame(None, "Live Prev", pa.ds)

def refr(source):
    fr.vp.Refresh()

pa.WantFrameNotification.append(refr)
fr.Show()

#pa.start()

#pa.stop()
import piezo_e662
pfoc = piezo_e662.piezo_e662()
pfoc.initialise()
import simplesequenceaquisator
sa = simplesequenceaquisator.SimpleSequenceAquisitor(chaninfo, cam, pfoc)
sa.SetStartMode(sa.START_AND_END)
sa.SetStepSize(5)
sa.SetStartPos(10)
sa.SetEndPos(55)
sa.Prepare()
sa.start()