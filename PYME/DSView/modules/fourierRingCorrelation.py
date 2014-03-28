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
import pylab

from PYME.DSView.dsviewer_npy_nb import ViewIm3D, ImageStack

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

class RingCorrelator:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        PROC_COLOCALISE = wx.NewId()
        
        
        dsviewer.mProcessing.Append(PROC_COLOCALISE, "Fourier Ring Correlation", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, PROC_COLOCALISE, self.OnColoc)



    
    def OnColoc(self, event):
        from PYME.Analysis import binAvg        
        voxelsize = self.image.voxelsize

        #assume we have exactly 2 channels #FIXME - add a selector
        #grab image data
        imA = self.image.data[:,:,:,0].squeeze()
        imB = self.image.data[:,:,:,1].squeeze()
        
        X, Y = np.mgrid[0:float(imA.shape[0]), 0:float(imA.shape[1])]
        X = X/X.shape[0]
        Y = Y/X.shape[1]
        X = X - .5
        Y = Y - .5
        R = np.sqrt(X**2 + Y**2)

        H1 = pylab.fftn(imA)
        H2 = pylab.fftn(imB)
        
        rB = np.linspace(0,R.max())
        
        bn, bm, bs = binAvg.binAvg(R, pylab.fftshift(abs(H1*H2)), rB)
        
        bn1, bm1, bs1 = binAvg.binAvg(R, pylab.fftshift(abs(H1*H1)), rB)
        bn2, bm2, bs2 = binAvg.binAvg(R, pylab.fftshift(abs(H2*H2)), rB)
       
       
        

        pylab.figure()
        
        ax = pylab.gca()
        
        ax.plot(rB[:-1], bm/np.sqrt(bm1*bm2))
        ax.plot(rB[:-1], 2./np.sqrt(bn/2))
        
        xt = np.array([10., 15, 20, 30, 50, 80, 100, 150])
        rt = voxelsize[0]/xt
        
        pylab.xticks(rt[::-1],['%d' % xi for xi in xt[::-1]])
        
        
        pylab.show()
        
#        im = ImageStack(plots, titleStub='Radial Distribution')
#        im.xvals = bins[:-1]
#
#
#        im.xlabel = 'Distance [nm]'
#
#        im.ylabel = 'Fraction'
#        im.defaultExt = '.txt'
#
#        im.mdh['voxelsize.x'] = (bins[1] - bins[0])*1e-3
#        im.mdh['ChannelNames'] = pnames
#        im.mdh['Profile.XValues'] = im.xvals
#        im.mdh['Profile.XLabel'] = im.xlabel
#        im.mdh['Profile.YLabel'] = im.ylabel
#        
#        im.mdh['Colocalisation.Channels'] = names
#        im.mdh['Colocalisation.Thresholds'] = [tA, tB]
#        im.mdh['Colocalisation.Pearson'] = pearson
#        im.mdh['Colocalisation.Manders'] = [MA, MB]
#
#        im.mdh['OriginalImage'] = self.image.filename
#
#        ViewIm3D(im, mode='graph')



def Plug(dsviewer):
    dsviewer.ringCorrelator = RingCorrelator(dsviewer)



