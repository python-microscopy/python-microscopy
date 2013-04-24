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

        fit_menu = wx.Menu()

        FIT_RAW_INTENSITY_DECAY = wx.NewId()
        FIT_GAUSS = wx.NewId()
        
        fit_menu.Append(FIT_RAW_INTENSITY_DECAY, "Raw Intensity Decay", "", wx.ITEM_NORMAL)
        fit_menu.Append(FIT_GAUSS, "Gaussian", "", wx.ITEM_NORMAL)

        dsviewer.menubar.Insert(dsviewer.menubar.GetMenuCount()-1, fit_menu, 'Fitting')

        wx.EVT_MENU(dsviewer, FIT_RAW_INTENSITY_DECAY, self.OnRawDecay)
        wx.EVT_MENU(dsviewer, FIT_GAUSS, self.OnGaussianFit)
        
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
            
            print res[0]
            #imo = self.image.parent
        pylab.legend()
        #rawIntensity.processIntensityTrace(I, imo.mdh, dt=imo.mdh['Camera.CycleTime'])
        pylab.show()
        




def Plug(dsviewer):
    dsviewer.fitter = fitter(dsviewer)
    
