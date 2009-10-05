#!/usr/bin/python

##################
# _init.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import sys
sys.path.append(".")

#import example
#import vfr
#import previewaquisator
#import simplesequenceaquisator
import funcs
import piezo_e662
import piezo_e816
import psliders
import intsliders
import seqdialog

scope = funcs.microscope()

#Setup for binning = 2 
#scope.cam.SetHorizBin(1)
#scope.cam.SetVertBin(1)
scope.cam.SetCOC()
scope.cam.GetStatus()

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['col']
    cols = [2|4|8|16] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [funcs.example.CShutterControl.CH1] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo

#setup the piezos
#scope.pfoc = piezo_e816.piezo_e816(1,100)
#scope.pfoc.MoveTo(1,50)

#scope.piezos.append((scope.pfoc, 1, 'Pfoc'))

#scope.livepreview(sh.GetParent())

#psl = psliders.PiezoSliders(scope.piezos, parent=sh.GetParent())
#psl.Show()

isl = intsliders.IntegrationSliders(scope.chaninfo,parent=sh.GetParent())
isl.Show()

#sqd = seqdialog.SeqDialog(sh.GetParent(), scope)
#sqd.Show()