#!/usr/bin/python

##################
# ccdAdjPanel.py
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

from PYME.contrib import wxPlotPanel
import wx
import numpy as np
import matplotlib
from PYME.Acquire.Hardware import EMCCDTheory
from PYME.Acquire.Hardware import ccdCalibrator
import scipy.special

class sizedCCDPanel(wx.Panel):
    def __init__(self, parent, scope, empan):
        wx.Panel.__init__(self, parent)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.ccdPan = ccdPlotPanel(self, scope, empan)

        vsizer.Add(self.ccdPan, 1, wx.EXPAND, 0)

        self.SetSizerAndFit(vsizer)


class ccdPlotPanel(wxPlotPanel.PlotPanel):
    def __init__(self, parent, scope, empan, **kwargs ):
        self.scope = scope
        #self.dp = dispPan
        self.empan = empan
        self.eventDuration = 0.05

        wxPlotPanel.PlotPanel.__init__( self, parent, **kwargs )

    def draw( self ):
            """Draw data."""
            #matplotlib.interactive(False)

            if not 'vsp' in dir(self.scope): # can't do anything
                #self.figure.show()
                #matplotlib.interactive(True)
                return

            emGainSettings = np.arange(0, 220, 5)

            emGains = ccdCalibrator.getCalibratedCCDGain(emGainSettings, self.scope.cam.GetCCDTempSetPoint())

            if emGains is None: # can't do anything
                #self.figure.show()
                #matplotlib.interactive(True)
                return

            matplotlib.interactive(False)

            if not hasattr( self, 'spEMGain' ):
                self.spEMSNR = self.figure.add_axes([.15,.55,.7,.3])
                self.spEMHeadroom = self.figure.add_axes([.15,.55,.7,.3], axisbg=None, frameon=False, label='headroom')
                self.spIntSNR = self.figure.add_axes([.15,.1,.7,.3])
                self.spIntFrameRate = self.spIntSNR.twinx()#self.figure.add_axes([.15,.1,.7,.3], axisbg=None, frameon=False, label='framerate')

            #a, ed = numpy.histogram(self.fitResults['tIndex'], self.Size[0]/2)
            self.spEMSNR.cla()
            self.spEMHeadroom.cla()
            self.spIntSNR.cla()
            self.spIntFrameRate.cla()

            

            currEMGain = ccdCalibrator.getCalibratedCCDGain(self.scope.cam.GetEMGain(), self.scope.cam.GetCCDTempSetPoint())

            Imin = self.scope.vsp.dmin
            Imax = self.scope.vsp.dmax
            Imean = self.scope.vsp.dmean

            off = self.scope.cam.ADOffset

            #snrMin = EMCCDTheory.SNR((np.minimum(Imin - off, 1)/currEMGain)*self.scope.cam.ElectronsPerCount, self.scope.cam.ReadoutNoise, emGains, self.scope.cam.NGainElements)
            snrMean = EMCCDTheory.SNR(((Imean - off)/currEMGain)*self.scope.cam.ElectronsPerCount, self.scope.cam.ReadoutNoise, emGains, self.scope.cam.NGainElements)
            snrMax = EMCCDTheory.SNR(((Imax - off)/currEMGain)*self.scope.cam.ElectronsPerCount, self.scope.cam.ReadoutNoise, emGains, self.scope.cam.NGainElements)

            
            #self.spEMSNR.plot(np.log10(emGains), 10*np.log10(snrMin), color='b')
            self.spEMSNR.plot(np.log10(emGains), 10*np.log10(snrMean), color='b', lw=2)
            self.spEMSNR.plot(np.log10(emGains), 10*np.log10(snrMax), color='b', lw=2)
            xticks = [1, 10, 100, 1000]
            self.spEMSNR.set_xticks(np.log10(xticks))
            self.spEMSNR.set_xticklabels([str(t) for t in xticks])
            self.spEMSNR.set_xlim(np.log10(emGains).min(), np.log10(emGains).max())
            self.spEMSNR.set_xlabel('True EM Gain')
            self.spEMSNR.set_ylabel('SNR [dB]', color = 'b')
            for t in self.spEMSNR.get_yticklabels():
                t.set_color('b')

            self.spEMHeadroom.yaxis.tick_right()
            self.spEMHeadroom.yaxis.set_label_position('right')
            self.spEMHeadroom.xaxis.tick_top()
            self.spEMHeadroom.xaxis.set_label_position('top')

            self.spEMHeadroom.semilogy(np.log10(emGains), np.maximum((self.scope.cam.SaturationThreshold - off)/((Imax - off)*emGains/currEMGain), 1), color='r', lw=2)
            #self.spEMHeadroom.xaxis.tick_top()
            xticks = [0, 50, 100, 150, 200]
            self.spEMHeadroom.set_xticks(np.log10(ccdCalibrator.getCalibratedCCDGain(xticks, self.scope.cam.GetCCDTempSetPoint())))
            self.spEMHeadroom.set_xticklabels([str(t) for t in xticks])
            self.spEMHeadroom.set_xlim(np.log10(emGains).min(), np.log10(emGains).max())
            self.spEMHeadroom.set_xlabel('EM Gain Setting')
            self.spEMHeadroom.set_ylabel('Headroom - Isat/Imax', color = 'r')
            self.spEMHeadroom.set_ylim(ymin=1)
            for t in self.spEMHeadroom.get_yticklabels():
                t.set_color('r')


            iTimes = np.logspace(-3, 0, 50)
            iScale = iTimes/self.scope.cam.GetIntegTime()

            #snrMeanI = EMCCDTheory.SNR((iScale*(Imean - off)/currEMGain)*self.scope.cam.ElectronsPerCount, self.scope.cam.ReadoutNoise, currEMGain, self.scope.cam.NGainElements)
            snrMaxI = EMCCDTheory.SNR((iScale*(Imax - off)/currEMGain)*self.scope.cam.ElectronsPerCount, self.scope.cam.ReadoutNoise, currEMGain, self.scope.cam.NGainElements,(iScale*(Imean - off)/currEMGain)*self.scope.cam.ElectronsPerCount)

            evtI = (Imax - Imean)/scipy.special.gammainc(1, self.scope.cam.GetIntegTime()/self.eventDuration)
            snrFlash = EMCCDTheory.SNR(((scipy.special.gammainc(1, iTimes/self.eventDuration)*evtI + iScale*(Imean - off))/currEMGain)*self.scope.cam.ElectronsPerCount, self.scope.cam.ReadoutNoise, currEMGain, self.scope.cam.NGainElements,(iScale*(Imean - off)/currEMGain)*self.scope.cam.ElectronsPerCount)

            evtI = (Imax - Imean)/scipy.special.gammainc(1, self.scope.cam.GetIntegTime()*5/self.eventDuration)
            snrFlash02 = EMCCDTheory.SNR(((scipy.special.gammainc(1, iTimes*5/self.eventDuration)*evtI + iScale*(Imean - off))/currEMGain)*self.scope.cam.ElectronsPerCount, self.scope.cam.ReadoutNoise, currEMGain, self.scope.cam.NGainElements,(iScale*(Imean - off)/currEMGain)*self.scope.cam.ElectronsPerCount)

            evtI = (Imax - Imean)/scipy.special.gammainc(1, self.scope.cam.GetIntegTime()/(self.eventDuration*5))
            snrFlash5 = EMCCDTheory.SNR(((scipy.special.gammainc(1, iTimes/(self.eventDuration*5))*evtI + iScale*(Imean - off))/currEMGain)*self.scope.cam.ElectronsPerCount, self.scope.cam.ReadoutNoise, currEMGain, self.scope.cam.NGainElements,(iScale*(Imean - off)/currEMGain)*self.scope.cam.ElectronsPerCount)


            #self.spIntSNR.plot(np.log10(iTimes), 10*np.log10(snrMeanI), color='b', lw=2)
            #self.spIntSNR.plot(np.log10(iTimes), 10*np.log10(snrMaxI), color='b', lw=2)
            self.spIntSNR.plot(np.log10(iTimes), 10*np.log10(snrFlash), color='b', lw=2)
            self.spIntSNR.plot(np.log10(iTimes), 10*np.log10(snrFlash02), color='b', lw=2, ls=';')
            self.spIntSNR.plot(np.log10(iTimes), 10*np.log10(snrFlash5), color='b', lw=2, ls='--')

            xticks = [1, 10, 100, 1000]
            self.spIntSNR.set_xticks(np.log10(np.array(xticks)*1e-3))
            self.spIntSNR.set_xticklabels([str(t) for t in xticks])
            self.spIntSNR.set_xlim(np.log10(iTimes).min(), np.log10(iTimes).max())
            self.spIntSNR.set_xlabel('Integration Time [ms]')
            self.spIntSNR.set_ylabel('SNR [dB]', color = 'b')
            for t in self.spIntSNR.get_yticklabels():
                t.set_color('b')


            self.spIntFrameRate.yaxis.tick_right()
            self.spIntFrameRate.yaxis.set_label_position('right')

            tFrame = np.maximum(iTimes, (1e-6/self.scope.cam.HorizShiftSpeeds[0][0][self.scope.cam.HSSpeed]*self.scope.cam.GetPicWidth()*self.scope.cam.GetPicHeight()))
            tFrame += self.scope.cam.GetCCDHeight()*self.scope.cam.vertShiftSpeeds[self.scope.cam.VSSpeed]*1e-6

            self.spIntFrameRate.plot(np.log10(iTimes), 1./tFrame, color='g', lw=2)

            xticks = [1, 10, 100, 1000]
            #self.spIntSNR.set_xticks(np.log10(np.array(xticks)*1e-3))
            #self.spIntSNR.set_xticklabels([str(t) for t in xticks])
            #self.spIntSNR.set_xlim(np.log10(iTimes).min(), np.log10(iTimes).max())
            #self.spIntSNR.set_xlabel('Integration Time [ms]')
            self.spIntFrameRate.set_ylabel('FPS', color = 'g')
            for t in self.spIntFrameRate.get_yticklabels():
                t.set_color('g')

            #self.subplot1.set_yticks([0, a.max()])
            #self.subplot2.plot(ed[:-1], numpy.cumsum(a), color='g' )
            #self.subplot2.set_xticks([0, ed.max()])
            #self.subplot2.set_yticks([0, a.sum()])

            #self.figure.show()

            matplotlib.interactive(True)

