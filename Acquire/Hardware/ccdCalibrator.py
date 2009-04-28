from PYME import cSMI
import numpy as np
import wx
from pylab import *

class ccdCalibrator:
    ''' class for calibrating the ccd

    usage:
    - use transmitted light to produce a roughly uniform field
    - make sure CCD temperature has settled to the the specified set point
    - adjust intensity and integration time so that camera is not saturated at
      the maximum em gain which is going to be calibrated for (220 when using
      default options). To give good results the image at max em-gain should
      however be bright - ~50% of saturation seems like a good point. Note that
      this WILL saturate the display (12 bits as opposed to 16) - use the
      histogram display.
    - instantiate the class on the console - ie:
      from PYME.Hardware import ccdCalibrator
      ccdCal = ccdCalibrator.ccdCalibrator(scope.pa, scope.cam)
    '''
    def __init__(self, pa, cam, gains = np.arange(0, 220, 5)):
        self.pa = pa
        self.cam = cam

        self.gains = gains
        self.pos = -1

        self.contMode = self.cam.contMode
        self.emgain = self.cam.GetEMGain()

        self.realGains = np.zeros(len(gains))

        self.dsa = cSMI.CDataStack_AsArray(self.pa.ds, 0)

        self.pa.stop()
        #self.cam.SetAcquisitionMode(self.cam.MODE_SINGLE_SHOT)

        self.pa.WantFrameNotification.append(self.tick)
        self.cam.SetShutter(0)

        self.cam.SetBaselineClamp(True) #otherwise baseline changes with em gain

        self.pd = wx.ProgressDialog('CCD Gain Calibration', 'Callibrating CCD Gain', len(self.gains))
        #self.pd.Show()
        self.pa.start()


    def finish(self):
        #self.pa.stop()
        self.pa.WantFrameNotification.remove(self.tick)
        
        #if self.contMode:
        #    self.cam.SetAcquisitionMode(self.cam.MODE_CONTINUOUS)

        self.cam.SetEMGain(self.emgain)

        self.realGains = self.realGains/self.realGains[0]

        figure()
        plot(self.gains, self.realGains)

        #self.pa.start()

    def tick(self, caller):
        self.pa.stop()
        imMean = self.dsa.mean()
        if self.pos == -1: #calculate background
            self.offset = imMean
            self.cam.SetShutter(1)
        else:
            self.realGains[self.pos] = imMean - self.offset

        self.pos += 1
        self.pd.Update(self.pos)
        if self.pos < len(self.gains):
            self.cam.SetEMGain(self.gains[self.pos])
        else:
            self.finish()

        self.pa.start()
            
            
