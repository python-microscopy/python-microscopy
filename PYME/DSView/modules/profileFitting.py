#!/usr/bin/python
##################
# profilePlotting.py
#
# Copyright David Baddeley, 2011
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

import wx

#from PYME.Acquire.mytimer import mytimer
# import pylab
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np

from PYME.DSView.dsviewer import ViewIm3D, ImageStack

from ._base import Plugin
class Fitter(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        
        dsviewer.AddMenuItem('Fitting', "Raw Intensity Decay", self.OnRawDecay)
        dsviewer.AddMenuItem('Fitting', "Simple Decay", self.OnRawDecaySimp)
        dsviewer.AddMenuItem('Fitting', "Gaussian", self.OnGaussianFit)
        
    def OnRawDecay(self, event):
        from PYME.Analysis.BleachProfile import rawIntensity
        I = self.image.data[:,0,0].squeeze()
        imo = self.image.parent
        
        rawIntensity.processIntensityTrace(I, imo.mdh, dt=imo.mdh['Camera.CycleTime'])
        
    def OnGaussianFit(self, event):
        import PYME.Analysis._fithelpers as fh
        
        def gmod(p,x):
            A, x0, sig, b = p
            
            return A*np.exp(-(x-x0)**2/(2*sig**2)) + b
        
        plt.figure()
        
        cols = ['b','g','r', 'C0', 'C1']
        xv = self.image.xvals
        
        for chan in range(self.image.data.shape[3]):    
            I = self.image.data_xyztc[:,:,0,0, chan].squeeze().astype('f')
            
            res = fh.FitModel(gmod, [I.max()-I.min(), xv[I.argmax()], xv[1] - xv[0], I.min()], I, xv)
            
            plt.plot(xv, I, cols[chan] + 'x', label=self.image.names[chan])
            plt.plot(xv, gmod(res[0], self.image.xvals), cols[chan],
                       label='A: %2.3g, x0 :%2.4g, \n sig: %2.4g,  b: %2.3g\nFWHM: %2.3g' % (tuple(res[0]) + (2.35*res[0][2],)))
            
            print((res[0]))
            #imo = self.image.parent
        plt.legend()
        #rawIntensity.processIntensityTrace(I, imo.mdh, dt=imo.mdh['Camera.CycleTime'])
        plt.show()
        
    def OnRawDecaySimp(self, event):
        #from pylab import *
        import matplotlib.pyplot as plt
        I = self.image.data[:].squeeze()
        t = self.image.xvals
        
        dt = t[1] - t[0]
        
        #prebleach
        
        plt.figure()
        bStart, bEnd = self.image.parent.mdh['Protocol.BleachFrames']
        
        Ib = 1.0*I[bStart:bEnd]
        tb = t[bStart:bEnd]
        
        #scale to [0,1]
        Ib-= Ib.min()
        Ib /= Ib.max()
        tau_shelve = (Ib > 1./np.e).sum()*dt
        plt.plot(tb, Ib)
        plt.figtext(.7, .8, 'Tau = %3.4f s'%tau_shelve)
        plt.xlabel('Time [s]')
        plt.title('Prebleach Intensity')
        
        #total decay
        plt.figure()
        multiplier = np.ones_like(I)
        multiplier[bStart:bEnd] = self.image.parent.mdh['Camera.TrueEMGain']
        
        pStart, pEnd = self.image.parent.mdh['Protocol.PrebleachFrames']
        
        multiplier[pStart:pEnd] = 100

        plt.plot(I*multiplier)
        #figure()
        #plot(multiplier)
        
        #actual imaging
        plt.figure()
        Ii = I[self.image.parent.mdh['Protocol.DataStartsAt']:]
        ti = t[self.image.parent.mdh['Protocol.DataStartsAt']:]
        
        Ii = Ii - self.image.parent.mdh['Camera.ADOffset']
        
        Ii = Ii/Ii.max()

        plt.plot(ti, Ii)
        
        n100 = abs(ti-100).argmin()
        print((n100, Ii[n100]))
        plt.figtext(.5, .8, 'I100/Imax = %3.4f'%(Ii[n100]))
        plt.plot(ti[n100], Ii[n100], 'xr')
        
        
        
        


def Plug(dsviewer):
    return Fitter(dsviewer)
    
