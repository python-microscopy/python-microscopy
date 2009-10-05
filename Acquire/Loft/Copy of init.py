#!/usr/bin/python

##################
# Copy of init.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import sys
sys.path.append(r'c:\pysmi_simulator\py_fit')

import scipy
import fakeCam
#import fakeCam
#reload(fakeCam)

import example
#example.CDataStack=fakeCam.CDataStack

scope.fakePiezo = fakeCam.FakePiezo(100)
scope.piezos.append((scope.fakePiezo, 1, 'Fake z-piezo'))

scope.cam = fakeCam.FakeCamera(70*scipy.arange(-128.0, 128.0), 70*scipy.arange(-128.0, 128.0), fakeCam.NoiseMaker(), scope.fakePiezo, fluors=[])
#import vfr
#import previewaquisator
#import simplesequenceaquisator
#import funcs
#import piezo_e662
#import piezo_e816
#import psliders
#import intsliders
#import seqdialog

#scope = funcs.microscope()

#Setup for binning = 2 
#scope.cam.SetHorizBin(1)
#scope.cam.SetVertBin(1)
#scope.cam.SetCOC()
#scope.cam.GetStatus()

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [example.CShutterControl.CH1] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo

#setup the piezos
#scope.pfoc = piezo_e816.piezo_e816(1,100)
#scope.pfoc.MoveTo(1,50)

#scope.piezos.append((scope.pfoc, 1, 'Pfoc'))
import lasersliders
lsf = lasersliders.LaserSliders(scope.cam)
lsf.Show()

import dSimControl
dsc = dSimControl.dSimControl(None, scope)
dsc.Show()

import remFit6
import Pyro
tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')


def postTask(hi=None):
    im = example.CDataStack_AsArray(scope.pa.ds,0)[:,:,0]
    t = remFit6.fitTask(im, 10)
    tq.postTask(t)

#scope.pa.WantFrameNotification.append(postTask)
