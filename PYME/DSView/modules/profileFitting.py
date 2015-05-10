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
import pylab
from scipy import ndimage
import numpy as np

from PYME.DSView.dsviewer_npy_nb import ViewIm3D, ImageStack

class fitter:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        #self.view = dsviewer.view
        self.do = dsviewer.do
        self.image = dsviewer.image
        
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
            
            return A*pylab.exp(-(x-x0)**2/(2*sig**2)) + b
        
        pylab.figure()
        
        cols = ['b','g','r']
        xv = self.image.xvals
        
        for chan in range(self.image.data.shape[3]):    
            I = self.image.data[:,0,0, chan].squeeze().astype('f')
            
            res = fh.FitModel(gmod, [I.max()-I.min(), xv[I.argmax()], xv[1] - xv[0], I.min()], I, xv)
            
            pylab.plot(xv, I, cols[chan] + 'x', label=self.image.names[chan])
            pylab.plot(xv, gmod(res[0], self.image.xvals), cols[chan], label='%2.3g, %2.3g, \n%2.3g, %2.3g' % tuple(res[0]))
            
            print((res[0]))
            #imo = self.image.parent
        pylab.legend()
        #rawIntensity.processIntensityTrace(I, imo.mdh, dt=imo.mdh['Camera.CycleTime'])
        pylab.show()
        
    def OnRawDecaySimp(self, event):
        from pylab import *
        I = self.image.data[:].squeeze()
        t = self.image.xvals
        
        dt = t[1] - t[0]
        
        #prebleach
        
        figure()
        bStart, bEnd = self.image.parent.mdh['Protocol.BleachFrames']
        
        Ib = 1.0*I[bStart:bEnd]
        tb = t[bStart:bEnd]
        
        #scale to [0,1]
        Ib-= Ib.min()
        Ib /= Ib.max()
        tau_shelve = (Ib > 1./np.e).sum()*dt
        plot(tb, Ib)
        figtext(.7, .8, 'Tau = %3.4f s'%tau_shelve)
        xlabel('Time [s]')
        title('Prebleach Intensity')
        
        #total decay
        figure()
        multiplier = np.ones_like(I)
        multiplier[bStart:bEnd] = self.image.parent.mdh['Camera.TrueEMGain']
        
        pStart, pEnd = self.image.parent.mdh['Protocol.PrebleachFrames']
        
        multiplier[pStart:pEnd] = 100
        
        plot(I*multiplier)
        #figure()
        #plot(multiplier)
        
        #actual imaging
        figure()
        Ii = I[self.image.parent.mdh['Protocol.DataStartsAt']:]
        ti = t[self.image.parent.mdh['Protocol.DataStartsAt']:]
        
        Ii = Ii - self.image.parent.mdh['Camera.ADOffset']
        
        Ii = Ii/Ii.max()
        
        plot(ti, Ii)
        
        n100 = abs(ti-100).argmin()
        print((n100, Ii[n100]))
        figtext(.5, .8, 'I100/Imax = %3.4f'%(Ii[n100]))
        plot(ti[n100], Ii[n100], 'xr')
        
        
        
        


def Plug(dsviewer):
    dsviewer.fitter = fitter(dsviewer)
    
