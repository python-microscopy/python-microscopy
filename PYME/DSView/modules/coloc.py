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
import numpy
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

class colocaliser:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        PROC_COLOCALISE = wx.NewId()
        PROC_COLOCALISE_EDT = wx.NewId()
        
        dsviewer.mProcessing.AppendSeparator()
        dsviewer.mProcessing.Append(PROC_COLOCALISE, "&Colocalisation", "", wx.ITEM_NORMAL)
        dsviewer.mProcessing.Append(PROC_COLOCALISE_EDT, "EDT Colocalisation", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, PROC_COLOCALISE, self.OnColocBasic)
        wx.EVT_MENU(dsviewer, PROC_COLOCALISE_EDT, self.OnColoc)



    
    def OnColoc(self, event):
        from PYME.Analysis.Colocalisation import correlationCoeffs, edtColoc
        voxelsize = [1e3*self.image.mdh.getEntry('voxelsize.x') ,1e3*self.image.mdh.getEntry('voxelsize.y'), 1e3*self.image.mdh.getEntry('voxelsize.z')]
        
        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % n for n in range(self.image.data.shape[3])]
        
        dlg = ColocSettingsDialog(self.dsviewer, voxelsize[0], names)
        dlg.ShowModal()
        
        bins = dlg.GetBins()
        chans = dlg.GetChans()
        dlg.Destroy()

        #assume we have exactly 2 channels #FIXME - add a selector
        #grab image data
        imA = self.image.data[:,:,:,chans[0]].squeeze()
        imB = self.image.data[:,:,:,chans[1]].squeeze()

        #assume threshold is half the colour bounds - good if using threshold mode
        tA = self.do.Offs[chans[0]] + .5/self.do.Gains[chans[0]] #pylab.mean(self.ivps[0].clim)
        tB = self.do.Offs[chans[1]] + .5/self.do.Gains[chans[1]] #pylab.mean(self.ivps[0].clim)
        
        nameA = names[chans[0]]
        nameB = names[chans[1]]

        voxelsize = voxelsize[:imA.ndim] #trunctate to number of dimensions

        print('Calculating Pearson and Manders coefficients ...')        
        pearson = correlationCoeffs.pearson(imA, imB)
        MA, MB = correlationCoeffs.thresholdedManders(imA, imB, tA, tB)

        print('Performing distance transform ...')        
        bnA, bmA, binsA = edtColoc.imageDensityAtDistance(imB, imA > tA, voxelsize, bins)
        print('Performing distance transform (reversed) ...') 
        bnB, bmB, binsB = edtColoc.imageDensityAtDistance(imA, imB > tB, voxelsize, bins)
        
        #print binsB, bmB
        
        plots = []
        pnames = []

        pylab.figure()
        pylab.figtext(.1, .95, 'Pearson: %2.2f   M1: %2.2f M2: %2.2f' % (pearson, MA, MB))
        pylab.subplot(211)
        p = bmA/bmA.sum()
        #print p
        pylab.bar(binsA[:-1], p, binsA[1] - binsA[0])
        pylab.xlabel('Distance from edge of %s [nm]' % nameA)
        pylab.ylabel('Density of %s' % nameB)
        plots.append(p.reshape(-1, 1,1))
        pnames.append('Dens. %s from %s' % (nameB, nameA))

        pylab.subplot(212)
        p = bmB/bmB.sum()
        pylab.bar(binsB[:-1], p, binsB[1] - binsB[0])
        pylab.xlabel('Distance from edge of %s [nm]' % nameB)
        pylab.ylabel('Density of %s' % nameA)
        plots.append(p.reshape(-1, 1,1))
        pnames.append('Dens. %s from %s' % (nameA, nameB))

        pylab.figure()
        pylab.figtext(.1, .95, 'Pearson: %2.2f   M1: %2.2f M2: %2.2f' % (pearson, MA, MB))
        pylab.subplot(211)
        fA = bmA*bnA
        p = fA/fA.sum()
        pylab.bar(binsA[:-1], p, binsA[1] - binsA[0])
        pylab.xlabel('Distance from edge of %s [nm]' % nameA)
        pylab.ylabel('Fraction of %s' % nameB)
        plots.append(p.reshape(-1, 1,1))
        pnames.append('Frac. %s from %s' % (nameB, nameA))

        pylab.subplot(212)
        fB = bmB*bnB
        p = fB/fB.sum()
        pylab.bar(binsB[:-1], p, binsB[1] - binsB[0])
        pylab.xlabel('Distance from edge of %s [nm]' % nameB)
        pylab.ylabel('Fraction of %s' % nameA)
        plots.append(p.reshape(-1, 1,1))
        pnames.append('Frac. %s from %s' % (nameA, nameB))
        
        pylab.show()
        
        im = ImageStack(plots, titleStub='Radial Distribution')
        im.xvals = bins[:-1]


        im.xlabel = 'Distance [nm]'

        im.ylabel = 'Fraction'
        im.defaultExt = '.txt'

        im.mdh['voxelsize.x'] = (bins[1] - bins[0])*1e-3
        im.mdh['ChannelNames'] = pnames
        im.mdh['Profile.XValues'] = im.xvals
        im.mdh['Profile.XLabel'] = im.xlabel
        im.mdh['Profile.YLabel'] = im.ylabel
        
        im.mdh['Colocalisation.Channels'] = names
        im.mdh['Colocalisation.Thresholds'] = [tA, tB]
        im.mdh['Colocalisation.Pearson'] = pearson
        im.mdh['Colocalisation.Manders'] = [MA, MB]

        im.mdh['OriginalImage'] = self.image.filename

        ViewIm3D(im, mode='graph')
        
    def OnColocBasic(self, event):
        from PYME.Analysis.Colocalisation import correlationCoeffs, edtColoc
        voxelsize = [1e3*self.image.mdh.getEntry('voxelsize.x') ,1e3*self.image.mdh.getEntry('voxelsize.y'), 1e3*self.image.mdh.getEntry('voxelsize.z')]
        
        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % n for n in range(self.image.data.shape[3])]
        
        dlg = ColocSettingsDialog(self.dsviewer, voxelsize[0], names)
        dlg.ShowModal()
        
        bins = dlg.GetBins()
        chans = dlg.GetChans()
        dlg.Destroy()

        #assume we have exactly 2 channels #FIXME - add a selector
        #grab image data
        imA = self.image.data[:,:,:,chans[0]].squeeze()
        imB = self.image.data[:,:,:,chans[1]].squeeze()

        #assume threshold is half the colour bounds - good if using threshold mode
        tA = self.do.Offs[chans[0]] + .5/self.do.Gains[chans[0]] #pylab.mean(self.ivps[0].clim)
        tB = self.do.Offs[chans[1]] + .5/self.do.Gains[chans[1]] #pylab.mean(self.ivps[0].clim)
        
        nameA = names[chans[0]]
        nameB = names[chans[1]]

        voxelsize = voxelsize[:imA.ndim] #trunctate to number of dimensions

        print('Calculating Pearson and Manders coefficients ...')        
        pearson = correlationCoeffs.pearson(imA, imB)
        MA, MB = correlationCoeffs.thresholdedManders(imA, imB, tA, tB)
        
        I1 = imA.ravel()
        I2 = imB.ravel()
        h1 = np.histogram2d(np.clip(I1/I1.mean(), 0, 100), np.clip(I2/I2.mean(), 0, 100), 200)

        pylab.figure()
        pylab.figtext(.1, .95, 'Pearson: %2.2f   M1: %2.2f M2: %2.2f' % (pearson, MA, MB))
        pylab.subplot(111)
        
        pylab.imshow(np.log10(h1[0] + .1).T)
        pylab.xlabel('%s' % nameA)
        pylab.ylabel('%s' % nameB)

        
        pylab.show()
        



def Plug(dsviewer):
    dsviewer.coloc = colocaliser(dsviewer)



