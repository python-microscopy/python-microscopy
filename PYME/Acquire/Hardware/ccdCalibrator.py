#!/usr/bin/python

##################
# ccdCalibrator.py
#
# Copyright David Baddeley, 2009
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
##################

import numpy as np
import datetime
import time

global scope
scope = None

import logging
logger = logging.getLogger(__name__)

def setScope(sc):
    global scope
    scope = sc

def getCalibratedCCDGain(nomGain, temperature):
    with scope.settingsDB as conn:
        ret = conn.execute("SELECT nominalGains, trueGains FROM CCDCalibration2 WHERE temperature=? AND serial=? ORDER BY time DESC", (temperature,scope.cam.GetSerialNumber())).fetchone()
    if ret is None or np.max(nomGain) > ret[0].max():
        return None
    else:
        return np.interp(nomGain, ret[0], ret[1])

class ccdCalibrator:
    """ class for calibrating the ccd

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
      ccdCal = ccdCalibrator.ccdCalibrator(scope.frameWrangler, scope.cam)
    """
    def __init__(self, gains = np.arange(0, 220, 5)):
        global scope
        self.pa = scope.frameWrangler
        self.cam = scope.cam

        self.gains = gains
        self.pos = -1

        self.frameN = 0

        self.contMode = self.cam.contMode
        self.emgain = self.cam.GetEMGain()

        if abs(self.cam.GetCCDTemp() - self.cam.GetCCDTempSetPoint()) >=4:
            import wx
            wx.MessageBox('Error ...', 'CCD Temperature has not settled', wx.OK|wx.ICON_HAND)
            return 

        self.realGains = np.zeros(len(gains))

        self.dsa = self.pa.dsa

        self.pa.stop()
        #self.cam.SetAcquisitionMode(self.cam.MODE_SINGLE_SHOT)

        #self.pa.WantFrameNotification.append(self.tick)
        self.pa.onFrame.connect(self.tick)
        self.cam.SetShutter(0)

        self.cam.SetBaselineClamp(True) #otherwise baseline changes with em gain

        self.pd = wx.ProgressDialog('CCD Gain Calibration', 'Calibrating CCD Gain', len(self.gains))
        #self.pd.Show()
        self.pa.start()


    def finish(self):
        import wx
        logger.debug('Disconnecting')
        self.pa.onFrame.disconnect(self.tick)
        self.cam.SetEMGain(self.emgain)
        time.sleep(0.5)
        logger.debug('Disconnected')
        wx.CallAfter(self.plot)
        
    def plot(self):
        import matplotlib.pyplot as plt
        #self.pa.stop()
        #self.pa.onFrame.disconnect(self.tick)
        
        #if self.contMode:
        #    self.cam.SetAcquisitionMode(self.cam.MODE_CONTINUOUS)

        self.pd.Destroy()

        self.realGains = self.realGains/self.realGains[0]

        plt.figure()
        plt.semilogy(self.gains, self.realGains)
        plt.xlabel('EMGain Setting')
        plt.ylabel('EMGain')

        self._saveCalibration()

        #self.pa.start()

    def tick(self, sender, frameData, **kwargs):
        import wx
        if self.frameN < 3:
            self.frameN +=1
            return

        self.pa.stop()
        self.frameN = 0
        imMean = frameData.mean()
        if self.pos == -1: #calculate background
            self.offset = imMean
            self.cam.SetShutter(1)
        else:
            self.realGains[self.pos] = imMean - self.offset

        self.pos += 1
        wx.CallAfter(self.pd.Update,self.pos)
        if self.pos < len(self.gains):
            logger.debug('Setting EM Gain to %d' % self.gains[self.pos])
            self.cam.SetEMGain(self.gains[self.pos])
        else:
            self.finish()

        self.pa.start()


    def _saveCalibration(self):
        with scope.settingsDB as conn:
            conn.execute("INSERT INTO CCDCalibration2 VALUES (?, ?, ?, ?, ?)", (datetime.datetime.now(), self.cam.GetCCDTempSetPoint(), self.cam.GetSerialNumber(), self.gains,self.realGains))
        


            
            
