import sys
sys.path.append(".")

import example
import mytimer
import psliders
import intsliders
#import vfr
#import previewaquisator
#import simplesequenceaquisator
import funcs
#import piezo_e662
#import piezo_e816

scope = funcs.microscope()

#Setup for binning = 2 
scope.cam.SetHorizBin(1)
scope.cam.SetVertBin(1)
scope.cam.SetCOC()
scope.cam.GetStatus()

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

tim = mytimer.mytimer()

#scope.step = example.CStepOp()

#scope.step.Init(1)

#tim.WantNotification.append(scope.step.ContIO)
#tim.Start(100)

scope.livepreview()

psl = psliders.PiezoSliders(scope.piezos)
psl.Show()

isl = intsliders.IntegrationSliders(scope.chaninfo)
isl.Show()