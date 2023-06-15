#!/usr/bin/python
##################
# coloc.py
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
import numpy as np
import wx
#import pylab

from PYME.DSView.dsviewer import ViewIm3D, ImageStack

class ColocSettingsDialog(wx.Dialog):
    def __init__(self, parent, pxSize=100, names = []):
        wx.Dialog.__init__(self, parent, title='Colocalisation Settings')
        
        sizer1 = wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Minimum Distance:'), 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL,5)
        self.tMin = wx.TextCtrl(self, -1, '-600')
        hsizer.Add(self.tMin, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        
        sizer1.Add(hsizer, 0, wx.EXPAND)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Maximum Distance:'), 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL,5)
        self.tMax = wx.TextCtrl(self, -1, '2000')
        hsizer.Add(self.tMax, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        
        sizer1.Add(hsizer, 0, wx.EXPAND)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Bin Size:'), 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL,5)
        self.tStep = wx.TextCtrl(self, -1, '%d' % (2*pxSize))
        hsizer.Add(self.tStep, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        
        sizer1.Add(hsizer, 0, wx.EXPAND)
        
        if len(names) > 0:
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, '1st Channel:'), 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL,5)
            self.cChan1 = wx.Choice(self, -1, choices=names)
            self.cChan1.SetSelection(0)
            hsizer.Add(self.cChan1, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
            
            sizer1.Add(hsizer, 0, wx.EXPAND)
            
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, '2st Channel:'), 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL,5)
            self.cChan2 = wx.Choice(self, -1, choices=names)
            self.cChan2.SetSelection(1)
            hsizer.Add(self.cChan2, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
            
            sizer1.Add(hsizer, 0, wx.EXPAND)
        
        bOK = wx.Button(self, wx.ID_OK, 'OK')
        
        sizer1.Add(bOK, 0, wx.ALL|wx.ALIGN_RIGHT, 5)
        
        self.SetSizerAndFit(sizer1)
        
    def GetBins(self):
        return numpy.arange(float(self.tMin.GetValue()), float(self.tMax.GetValue()), float(self.tStep.GetValue()))
        
    def GetChans(self):
        return [self.cChan1.GetSelection(), self.cChan2.GetSelection()]

from ._base import Plugin
class RingCorrelator(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(dsviewer)
        
        PROC_COLOCALISE = wx.NewIdRef()
        
        
        dsviewer.mProcessing.Append(PROC_COLOCALISE, "Fourier Ring Correlation", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, PROC_COLOCALISE, self.OnColoc)



    
    def OnColoc(self, event):
        import matplotlib.pyplot as plt
        from PYME.Analysis import binAvg        
        voxelsize = self.image.voxelsize

        #assume we have exactly 2 channels #FIXME - add a selector
        #grab image data
        imA = self.image.data[:,:,:,0].squeeze()
        imB = self.image.data[:,:,:,1].squeeze()
        
        X, Y = np.mgrid[0:float(imA.shape[0]), 0:float(imA.shape[1])]
        X = X/X.shape[0]
        Y = Y/X.shape[1]
        X = (X - .5)
        Y = Y - .5
        R = np.sqrt(X**2 + Y**2)

        H1 = np.fft.fftn(imA)
        H2 = np.fft.fftn(imB)
        
        #rB = np.linspace(0,R.max())
        rB = np.linspace(0, 0.5, 100)
    
        bn, bm, bs = binAvg.binAvg(R, np.fft.fftshift(H1*H2.conjugate()).real, rB)
        
        bn1, bm1, bs1 = binAvg.binAvg(R, np.fft.fftshift((H1*H1.conjugate()).real), rB)
        bn2, bm2, bs2 = binAvg.binAvg(R, np.fft.fftshift((H2*H2.conjugate()).real), rB)
       

        plt.figure()
        
        ax = plt.gca()
        
        #FRC
        FRC = bm/np.sqrt(bm1*bm2)
        ax.plot(rB[:-1], FRC)
        
        #noise envelope???????????
        ax.plot(rB[:-1], 2./np.sqrt(bn/2), ':')
        
        dfrc = np.diff(FRC)
        monotone = np.where(dfrc > 0)[0][0] + 1
        
        #print FRC[:monotone], FRC[:(monotone+1)]
        
        intercept = np.interp(1.0 - 1/7.0, 1-FRC[:monotone], rB[:monotone])
        
        print('Intercept= %3.2f (%3.2f nm)' % (intercept, voxelsize[0]/intercept))
        
        xt = np.array([10.,  15, 20, 30, 40, 50, 70, 90, 120, 150, 200, 300, 500])
        rt = voxelsize[0]/xt
        
        plt.xticks(rt[::-1],['%d' % xi for xi in xt[::-1]], rotation='vertical')

        ax.plot([0, rt[0]], np.ones(2)/7.0)
        
        plt.grid()
        plt.xlabel('Resolution [nm]')
        
        plt.ylabel('FRC')
        
        plt.figtext(0.5, 0.5, 'FRC intercept at %3.1f nm' % (voxelsize[0]/intercept))
        
        
        plt.show()
        
        


def Plug(dsviewer):
    return RingCorrelator(dsviewer)



