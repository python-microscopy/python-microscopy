#!/usr/bin/python

###############
# zScanner.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################
from PYME.DSView import View3D, ViewIm3D
#from PYME import cSMI
import numpy as np
#from PYME.Acquire import eventLog
from math import floor
from PYME.IO import MetaDataHandler
from PYME.IO import image
import time

from PYME.contrib import dispatch

class zScanner:
    def __init__(self, scope):
        self.scope = scope
        self.stackSettings = scope.stackSettings
        self.off = 0
        self.sc = 100
        self.sqrt = False
        
        self.frameNum = 0
        
        #self.ds = scope.frameWrangler.currentFrame
        self.shape_x, self.shape_y = scope.frameWrangler.currentFrame.shape[:2]
        
        self.running = False
 
        #legacy signalling - don't use in new code.        
        #self.WantFrameNotification = []
        #self.WantTickNotification = []
        
        #These signals should instead of the notification lists above
        self.onStack = dispatch.Signal() #dispatched on completion of a stack
        self.onSingleFrame = dispatch.Signal()  #dispatched when a frame is ready
        
    def _endSingle(self, **kwargs):
        print('es')
        self.Stop()
        #self.WantFrameNotification.remove(self._endSingle)
        self.onStack.disconnect(self._endSingle)
        self.stackSettings.piezoGoHome()
        
    def Single(self):
        self.stackSettings.SetPrevPos(self.stackSettings._CurPos())
        #self.WantFrameNotification.append(self._endSingle)
        self.onStack.connect(self._endSingle)
        self.Start()
        
    def Start(self):
        self.image = np.zeros((self.shape_x, self.shape_y, 2), 'uint16')

        mdh = MetaDataHandler.NestedClassMDHandler()
        mdh.setEntry('StartTime', time.time())
        mdh.setEntry('AcquisitionType', 'Stack')

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(mdh)

        self.img = image.ImageStack(data=self.image, mdh=mdh)

        #self.view = View3D(self.image, 'Z Stack', mdh = mdh)
        #self.view = ViewIm3D(self.img, 'Z Stack')
        self.running = True
        
        self.zPoss = np.arange(self.stackSettings.GetStartPos(), self.stackSettings.GetEndPos()+.95*self.stackSettings.GetStepSize(),self.stackSettings.GetStepSize()*self.stackSettings.GetDirection())
        #piezo = self.scope.positioning[self.stackSettings.GetScanChannel()]
        #self.piezo = piezo[0]
        #self.piezoChan = piezo[1]
        self.posChan = self.stackSettings.GetScanChannel()
        #self.startPos = self.piezo.GetPos(self.piezoChan)
        self.startPos = self.scope.GetPos()[self.posChan]
        
        self.scope.frameWrangler.stop()
        self.scope.frameWrangler.onFrame.connect(self.OnCameraFrame)
        
        self.OnAqStart()
        self.scope.frameWrangler.start()
        
    def Stop(self):
        self.scope.frameWrangler.stop()
        self.scope.frameWrangler.onFrame.disconnect(self.OnCameraFrame)
        self.scope.frameWrangler.start()

        self.OnAqStop()
        
        self.running = False
        
        
    def OnAqStart(self, **kwargs):      
        self.pos = 0
        self.callNum = 0

        self.nz = len(self.zPoss)

        self.image = np.zeros((self.shape_x, self.shape_y, self.nz), 'uint16')

        self.img.SetData(self.image)
        #self.view.do.SetDataStack(self.img.data)
        
        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(self.img.mdh)
            
        #new metadata handling
        self.img.mdh.setEntry('StartTime', time.time())
        self.img.mdh.setEntry('AcquisitionType', 'Stack')
        
        #self.view_xz = View3D(self.image)
        #self.view_xz.do.slice = 1
        #self.view_xz.do.yp = self.ds.shape[1]/2
        #self.view_xz.do.xp = self.ds.shape[0]/2
        #self.view_yz = View3D(self.image)
        #self.view_yz.do.slice = 2
        #self.view_yz.do.yp = self.ds.shape[1]/2
        #self.view_yz.do.xp = self.ds.shape[0]/2

        #self.piezo.MoveTo(self.piezoChan, self.zPoss[self.pos])
        print('setting initial position')
        self.scope.SetPos(**{self.posChan : self.zPoss[self.pos]})


    def OnCameraFrame(self, sender, frameData, **kwargs):
        fn = int(floor(self.callNum) % len(self.zPoss))
        self.frameNum = fn
        #print(fn)
        if self.sqrt:
            self.image[:, :,fn] = (self.sc*np.sqrt(frameData[0,:,:] - self.off)).astype('uint16')
        else:
            self.image[:, :,fn] = frameData[0,:,:]

        #if not fn == self.pos:

        self.callNum += 1
        fn = int(floor(self.callNum) % len(self.zPoss))

        #self.piezo.MoveTo(self.piezoChan, self.zPoss[fn])
        self._movePiezo(fn)
        
        
        
        if fn == 0: #we've wrapped around 
            #make a copy of callbacks so that if we remove one, we still call the others
            #callbacks = [] + self.WantFrameNotification              
            #for cb in callbacks:
            #    cb()
                
            self.onStack.send(self)
                
            
            #if 'decView' in dir(self.view):
            #    self.view.decView.wienerPanel.OnCalculate()
        #self.view_xz.Refresh()
        #self.view_yz.Refresh()
        #for cb in self.WantTickNotification:
        #    cb()
            
        self.onSingleFrame.send(self)
            
        #if fn == 0:
        #    self.view.do.Optimise()
        #self.view.Refresh()

    def _movePiezo(self, fn):
        #self.piezo.MoveTo(self.piezoChan, self.zPoss[fn])
        self.scope.SetPos(**{self.posChan : self.zPoss[fn]})
        
    def OnAqStop(self, **kwargs):
        self.img.mdh.setEntry('EndTime', time.time())

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStopMetadata:
           mdgen(self.img.mdh)
        #pass

    def destroy(self):
        self.Stop()
        
    def getCentroid(self):
        mx = self.image.max()
        mn = self.image.min()
        
        im2 = np.maximum(self.image.astype('f') - (mn + (mx - mn)/2), 0)
        im2s = im2.sum()
        
        z0 = (im2*self.zPoss[None,None,:]).sum()/im2s
        #x_1, x_2 = self.scope.cam.ROIx
        #y_1, y_2 = self.scope.cam.ROIy
        x_1, y_1, x_2, y_2 = self.scope.state['Camera.ROI']
        x = np.arange(float(x_1), x_2+1)
        y = np.arange(float(y_1), y_2+1)
        x0 = (im2*x[:,None,None]).sum()/im2s
        y0 = (im2*y[None,:,None]).sum()/im2s
        
        print((x0, y0, z0))
        print((x0 - x.mean(), y0 - y.mean(), z0 - self.zPoss.mean()))
        
        return x0 - x.mean(), y0 - y.mean(), z0 - self.zPoss.mean()
        
    def center(self):
        dx, dy, dz = self.getCentroid()
        
        #self.scope.frameWrangler.stop()
        
        self.zPoss -= dz
        
        #x1, x2 = self.scope.cam.ROIx
        #y1, y2 = self.scope.cam.ROIy
        
        x1, y1, x2, y2 = self.scope.state['Camera.ROI']
        
        dx = int(dx)
        dy = int(dy)
        #print dx, dy, x1 - dx,y1-dy,x2-dx,y2-dy
        
        #self.scope.cam.SetROI(x1 - dx - 1,y1-dy - 1,x2-dx,y2-dy)
        self.scope.state['Camera.ROI'] = (x1 - dx - 1,y1-dy - 1,x2-dx,y2-dy)
        
        #self.scope.frameWrangler.start()

        #self.view.Destroy()
        #self.view_xz.Destroy()
        #self.view_yz.Destroy()

class wavetableZScanner(zScanner):
    def __init__(self, scope, triggered=False):
        zScanner.__init__(self, scope)
        
        self.triggered = triggered
        
    def OnAqStart(self, caller=None):
        zScanner.OnAqStart(self, caller)

        self.piezo.PopulateWaveTable(self.piezoChan, self.zPoss)

        #self.scope.frameWrangler.stop()

        if self.triggered:
            #if we've got a hardware trigger rigged up, use it
            self.piezo.StartWaveOutput(self.piezoChan)
        else:
            #otherwise fudge so that step time is nominally the same as exposure time
            self.piezo.StartWaveOutput(self.piezoChan, self.scope.cam.tKin*1e3)

        #self.scope.frameWrangler.start()
        
    def OnAqStop(self, caller=None):
        self.piezo.StopWaveOutput()

    #def destroy(self):
        #self.piezo.StopWaveOutput()
    #    zScanner.destroy(self)

    def _movePiezo(self, fn):
        pass

def getBestScanner(scope):
    piezo = scope.positioning[scope.stackSettings.GetScanChannel()][0]
    
    if 'StartWaveOutput' in dir(piezo) and not scope.stackSettings.GetSeqLength() > piezo.MAXWAVEPOINTS: #piezo supports wavetable output
        return wavetableZScanner(scope, piezo.hasTrigger)
    else: #no wavetable - possibly poorly synchronised
        return zScanner(scope)



