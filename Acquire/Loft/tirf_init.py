import sys
sys.path.append(".")

#import example
#import vfr
#import previewaquisator
#import simplesequenceaquisator
import funcs
import piezo_e662
import piezo_e816

scope = funcs.funcs()

#Setup for binning = 2 
scope.cam.SetHorizBin(1)
scope.cam.SetVertBin(1)
scope.cam.SetCOC()
scope.cam.GetStatus()

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [example.CShutterControl.CH1] #unimportant - as we have no shutters

scope.chaninfo = chaninfo

#setup the piezos
#pphase = piezo_e816.piezo_e816(1)
scope.pfoc = piezo_e662.piezo_e662(2)
scope.pfoc.MoveTo(1,50)

scope.piezos.append((scope.pfoc, 1, 'Pfoc'))

scope.livepreview()