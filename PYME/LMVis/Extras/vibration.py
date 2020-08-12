#!/usr/bin/python
##################
# photophysics.py
#
# Copyright David Baddeley, 2010
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

#import wx
import numpy as np
import math

class VibrationAnalyser:
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Extras>Diagnostics','Plot vibration spectra', self.VibrationSpecgram)

    def VibrationSpecgram(self, event):
        # import pylab as pl
        import matplotlib.pyplot as pl
        
        pipeline = self.visFr.pipeline
        
        x = pipeline['x']
        y = pipeline['y']
        
        x -= x.mean()
        y -= y.mean()
        
        xStd = []
        yStd = []
        window = 80
        
        for i in range(len(x)-window):
            xStd.append(x[i:i+window].std())
            yStd.append(y[i:i+window].std())
        
        xDrift = abs(x[0:window].mean()-x[-window-1:-1].mean())
        yDrift = abs(y[0:window].mean()-y[-window-1:-1].mean())
        
        
        ### create Spectrogram data and axes bounds    
        # set parameters for specgram
        NFFT = 1024
        noverlap = int(NFFT*.9)
        Fs=1/pipeline.mdh.getEntry('Camera.CycleTime')
        
        # create specgram data
        pl.figure('Specgram for X and Y', figsize=(16,6))
        xData, xFreq, xBins, a = pl.specgram(x, NFFT = NFFT, noverlap = noverlap, Fs = Fs, detrend=pl.detrend_linear)
        yData, yFreq, yBins, b = pl.specgram(y, NFFT = NFFT, noverlap = noverlap, Fs = Fs, detrend=pl.detrend_linear)
        pl.clf()
        
        # convert to dB    
        xData = 10*np.log10(xData)
        yData = 10*np.log10(yData)
        
        # create upper limit for colorbar
        ColorbarMax = max(25, xData.max(), yData.max())
        
        
        ### plot spectrograms
        # X        
        pl.subplot(121)
        pl.imshow(xData, aspect='auto', clim=[-15,ColorbarMax], origin='lower', interpolation='nearest', extent=[xBins[0], xBins[-1], xFreq[0], xFreq[-1]])
        pl.title('X')
        pl.ylabel('frequency [Hz]')
        #pl.yticks(freqLabels[0], freqLabels[1])
        pl.xlabel('framebin [s]\nx-drift(min-max): %d nm, vibration-std(80fr): %d nm' %(xDrift,np.average(xStd)))
        #pl.xticks(binLabels[0], binLabels[1])
        pl.colorbar()
        
        # Y
        pl.subplot(122)
        pl.imshow(yData, aspect='auto', clim=[-15,ColorbarMax], origin='lower', interpolation='nearest', extent=[xBins[0], xBins[-1], xFreq[0], xFreq[-1]])
        pl.title('Y')
        pl.ylabel('frequency [Hz]')
        #pl.yticks(freqLabels[0], freqLabels[1])
        pl.xlabel('framebin [s]\ny-drift(min-max): %d nm, vibration-std(80fr): %d nm' %(yDrift,np.average(yStd)))
        #pl.xticks(binLabels[0], binLabels[1])
        pl.colorbar()
        
        # plot x, y vs time to visualize drift        
        pl.figure('x- and y-drift visualization')      
        pl.plot(pipeline['t'].astype('f') / Fs, x)
        pl.xlabel('Time [s]')
        pl.plot(pipeline['t'].astype('f') / Fs, y)
        pl.ylabel('relative position [nm]')
        


def Plug(visFr):
    """Plugs this module into the gui"""
    VibrationAnalyser(visFr)



