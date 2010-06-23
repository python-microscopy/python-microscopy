from PYME.DSView.dsviewer_npy import View3D
from PYME import cSMI
import numpy as np
#from PYME.Acquire import eventLog
from math import floor

class zScanner:
    def __init__(self, scope):
        self.zPoss = np.arange(scope.sa.GetStartPos(), scope.sa.GetEndPos()+.95*scope.sa.GetStepSize(),scope.sa.GetStepSize())

        #if self.randomise:
        #    self.zPoss = self.zPoss[np.argsort(np.random.rand(len(self.zPoss)))]

        piezo = scope.sa.piezos[scope.sa.GetScanChannel()]
        self.piezo = piezo[0]
        self.piezoChan = piezo[1]
        self.startPos = self.piezo.GetPos(self.piezoChan)
        self.pos = 0

        self.scope = scope
        
        #pixels = np.array(pixels)

        self.callNum = 0
        self.ds = cSMI.CDataStack_AsArray(scope.pa.ds, 0)

        self.nz = len(self.zPoss)

        self.image = np.zeros((self.ds.shape[0], self.ds.shape[1], self.nz), 'uint16')

        self.view_xy = View3D(self.image)
        self.view_xz = View3D(self.image)
        self.view_xz.vp.do.slice = 1
        self.view_yz = View3D(self.image)
        self.view_yz.vp.do.slice = 2

        self.piezo.MoveTo(self.piezoChan, self.zPoss[self.pos])

        self.scope.pa.WantFrameNotification.append(self.tick)


    def tick(self, caller=None):
        fn = floor(self.callNum) % len(self.zPoss)
        self.image[:, :,fn] = self.ds[:,:,0]

        #if not fn == self.pos:

        self.callNum += 1
        fn = floor(self.callNum) % len(self.zPoss)

        #self.piezo.MoveTo(self.piezoChan, self.zPoss[fn])
        self._movePiezo(fn)
            
        self.view_xy.Refresh()
        self.view_xz.Refresh()
        self.view_yz.Refresh()

    def _movePiezo(self, fn):
        self.piezo.MoveTo(self.piezoChan, self.zPoss[fn])

    def destroy(self):
        self.scope.pa.WantFrameNotification.remove(self.tick)

        self.view_xy.Destroy()
        self.view_xz.Destroy()
        self.view_yz.Destroy()

class wavetableZScanner(zScanner):
    def __init__(self, scope, triggered=False):
        zScanner.__init__(self, scope)

        self.piezo.PopulateWaveTable(self.piezoChan, self.zPoss)

        self.scope.pa.stop()

        if triggered:
            #if we've got a hardware trigger rigged up, use it
            self.piezo.StartWaveOutput(self.piezoChan)
        else:
            #otherwise fudge so that step time is nominally the same as exposure time
            self.piezo.StartWaveOutput(self.piezoChan, scope.cam.tKin*1e3)

        self.scope.pa.start()

    def destroy(self):
        self.piezo.StopWaveOutput()
        zScanner.destroy(self)

    def _movePiezo(self, fn):
        pass





