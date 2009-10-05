#!/usr/bin/python

##################
# init_smi.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#import sys
#sys.path.append(".")

import example
import wx
#import mytimer
#import psliders
#import intsliders
#import seqdialog
#import vfr
#import previewaquisator
#import simplesequenceaquisator
#import funcs
#import piezo_e662
#import piezo_e816

#scope = funcs.microscope()

#Setup for binning = 2 
#scope.cam.SetHorizBin(1)
#scope.cam.SetVertBin(1)
#scope.cam.SetCOC()
#scope.cam.GetStatus()

scope.scopedetails = {}
scope.scopedetails['Name'] = 'SMI1'
scope.scopedetails['Objective'] = '63x Oil'
scope.scopedetails['Camera'] = 'B/W Sensicam'
scope.scopedetails['TubeLensMagnification'] = 1.25
scope.scopedetails['VoxelSizeX'] = 0.085

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['ar', 'kr']
    cols = [1,1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [example.CShutterControl.CH1, example.CShutterControl.CH2]
    itimes = [100,100]

scope.chaninfo = chaninfo

#setup the piezos
scope.pz = example.CPiezoOp()
scope.pz.Init(1)

scope.piezos.append((scope.pz, scope.pz.GetChannelObject(), 'Object'))
scope.piezos.append((scope.pz, scope.pz.GetChannelPhase(), 'Phase'))

#time = mytimer.mytimer()

scope.step = example.CStepOp()

time.WantNotification.append(scope.step.ContIO)
#time.Start(100)

mb = wx.MessageDialog(sh.GetParent(), 'Continue with Calibration of stage?\nPLEASE CHECK that the slide holder has been removed\n(and then press OK)', 'Stage Callibration', wx.YES_NO|wx.NO_DEFAULT)
ret = mb.ShowModal()
if (ret == wx.ID_YES):
    scope.step.Init(1)
else:
    scope.step.Init(2)

