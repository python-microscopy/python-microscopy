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
from __future__ import print_function
import numpy
import numpy as np
import wx
# import pylab
import matplotlib.pyplot as plt

from PYME.DSView.dsviewer import ViewIm3D, ImageStack

class ColocSettingsDialog(wx.Dialog):
    def __init__(self, parent, pxSize=100, names = [], have_mask=False, z_size=1, coloc_vs_z=False, show_bins=True):
        wx.Dialog.__init__(self, parent, title='Colocalisation Settings')
        self.zsize = z_size
        
        sizer1 = wx.BoxSizer(wx.VERTICAL)
        
        if show_bins:
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, 'Minimum Distance:'), 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL,5)
            self.tMin = wx.TextCtrl(self, -1, '-500')
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
        
        if have_mask:
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, 'Use Mask:'), 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            self.cbUseMask = wx.CheckBox(self, -1)
            self.cbUseMask.SetValue(False)
            hsizer.Add(self.cbUseMask, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
    
            sizer1.Add(hsizer, 0, wx.EXPAND)


        if self.zsize > 1:
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, 'z start:'), 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            self.tzs = wx.TextCtrl(self, -1, '0', size=[30, -1])
            hsizer.Add(self.tzs, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            hsizer.Add(wx.StaticText(self, -1, 'z end:'), 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            self.tze = wx.TextCtrl(self, -1, '%d' % (self.zsize - 1), size=[30, -1])
            hsizer.Add(self.tze, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            
            sizer1.Add(hsizer, 0, wx.EXPAND)
        
        
        if len(names) > 0:
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, '1st Channel:'), 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL,5)
            self.cChan1 = wx.Choice(self, -1, choices=names)
            self.cChan1.SetSelection(0)
            hsizer.Add(self.cChan1, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
            
            sizer1.Add(hsizer, 0, wx.EXPAND)
            
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, '2nd Channel:'), 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL,5)
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
    
    def GetUseMask(self):
        cbMask = getattr(self, 'cbUseMask', None)
        
        if cbMask is None:
            return False
        else:
            return cbMask.GetValue()

    def GetZrange(self):
        if self.zsize <= 1:
            return (None,None)
        zs = int(self.tzs.GetValue())
        ze = int(self.tze.GetValue())
        if zs < 0:
            zs = 0
        if zs > self.zsize-1:
            zs = self.zsize - 1
        if ze < 0:
            ze = 0
        if ze > self.zsize-1:
            ze = self.zsize - 1
        if ze <= zs:
            ze = zs+1 # it is still possible that ze now greater than max size
        return (zs,ze)

from ._base import Plugin
class Colocaliser(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        
        dsviewer.mProcessing.AppendSeparator()
        dsviewer.AddMenuItem('Processing', "&Colocalisation", self.OnColocBasic)
        dsviewer.AddMenuItem('Processing', "EDT Colocalisation", self.OnColoc)
        dsviewer.AddMenuItem('Processing', "EDT Colocalisation (z cropped)", lambda e: self.OnColoc(restrict_z=True))
        dsviewer.AddMenuItem('Processing', 'FRC', self.OnFRC)
    


    
    def OnColoc(self, event=None, restrict_z=False, coloc_vs_z = False):
        from PYME.Analysis.Colocalisation import correlationCoeffs, edtColoc
        from scipy import interpolate
        voxelsize = self.image.voxelsize_nm
        
        names = self.image.names
            
        if not getattr(self.image, 'labels', None) is None:
            have_mask = True
            
            mask = (self.image.labels > 0.5).squeeze()
        else:
            have_mask = False
            mask = None
        
        if not restrict_z:
            dlg = ColocSettingsDialog(self.dsviewer, voxelsize[0], names, have_mask=have_mask)
        else:
            dlg = ColocSettingsDialog(self.dsviewer, voxelsize[0], names, have_mask=have_mask, z_size=self.image.data.shape[2])
        dlg.ShowModal()
        
        bins = dlg.GetBins()
        chans = dlg.GetChans()
        use_mask = dlg.GetUseMask()

        zs, ze = (0, self.image.data.shape[2])
        if (self.image.data.shape[2] > 1) and restrict_z:
            zs, ze = dlg.GetZrange()
            
        zs,ze = (0,1)
        if self.image.data.shape[2] > 1:
            zs,ze = dlg.GetZrange()
        dlg.Destroy()
        
        print('Use mask: %s' % use_mask)
        
        if not use_mask:
            mask=None
        

        #assume we have exactly 2 channels #FIXME - add a selector
        #grab image data
        imA = self.image.data[:,:,zs:ze,chans[0]].squeeze()
        imB = self.image.data[:,:,zs:ze,chans[1]].squeeze()

        #assume threshold is half the colour bounds - good if using threshold mode
        tA = self.do.Offs[chans[0]] + .5/self.do.Gains[chans[0]] #plt.mean(self.ivps[0].clim)
        tB = self.do.Offs[chans[1]] + .5/self.do.Gains[chans[1]] #plt.mean(self.ivps[0].clim)
        
        nameA = names[chans[0]]
        nameB = names[chans[1]]

        voxelsize = voxelsize[:imA.ndim] #trunctate to number of dimensions

        print('Calculating Pearson and Manders coefficients ...')        
        pearson = correlationCoeffs.pearson(imA, imB, roi_mask=mask)
        MA, MB = correlationCoeffs.thresholdedManders(imA, imB, tA, tB, roi_mask=mask)

        if coloc_vs_z:
            MAzs, MBzs = ([],[])
            FAzs, FBzs = ([],[])
            if self.image.data.shape[2] > 1:
                for z in range(self.image.data.shape[2]):
                    imAz = self.image.data[:,:,z,chans[0]].squeeze()
                    imBz = self.image.data[:,:,z,chans[1]].squeeze()
                    MAz, MBz = correlationCoeffs.thresholdedManders(imAz, imBz, tA, tB)
                    FAz, FBz = correlationCoeffs.maskFractions(imAz, imBz, tA, tB)
                    MAzs.append(MAz)
                    MBzs.append(MBz)
                    FAzs.append(FAz)
                    FBzs.append(FBz)

                print("M(A->B) %s" % MAzs)
                print("M(B->A) %s" % MBzs)
                print("Species A: %s, Species B: %s" %(nameA,nameB))
    
                plt.figure()
                plt.subplot(211)
                if 'filename' in self.image.__dict__:
                    plt.title(self.image.filename)
                # nameB with nameA
                cAB, = plt.plot(MAzs,'o', label = '%s with %s' %(nameB,nameA))
                # nameA with nameB
                cBA, = plt.plot(MBzs,'*', label = '%s with %s' %(nameA,nameB))
                plt.legend([cBA,cAB])
                plt.xlabel('z slice level')
                plt.ylabel('Manders coloc fraction')
                plt.ylim(0,None)
    
                plt.subplot(212)
                if 'filename' in self.image.__dict__:
                    plt.title(self.image.filename)
                # nameB with nameA
                fA, = plt.plot(FAzs,'o', label = '%s mask fraction' %(nameA))
                # nameA with nameB
                fB, = plt.plot(FBzs,'*', label = '%s mask fraction' %(nameB))
                plt.legend([fA,fB])
                plt.xlabel('z slice level')
                plt.ylabel('Mask fraction')
                plt.ylim(0,None)
                plt.show()

        print('Performing distance transform ...')
        #bnA, bmA, binsA = edtColoc.imageDensityAtDistance(imB, imA > tA, voxelsize, bins, roi_mask=mask)
        #bnAA, bmAA, binsA = edtColoc.imageDensityAtDistance(imA, imA > tA, voxelsize, bins, roi_mask=mask)
        
        bins_, enrichment_BA, enclosed_BA, enclosed_area_A = edtColoc.image_enrichment_and_fraction_at_distance(imB, imA > tA, voxelsize,
                                                                                               bins, roi_mask=mask)
        bins_, enrichment_AA, enclosed_AA, _ = edtColoc.image_enrichment_and_fraction_at_distance(imA, imA > tA, voxelsize,
                                                                                               bins, roi_mask=mask)
        
        print('Performing distance transform (reversed) ...')
        #bnB, bmB, binsB = edtColoc.imageDensityAtDistance(imA, imB > tB, voxelsize, bins, roi_mask=mask)
        #bnBB, bmBB, binsB = edtColoc.imageDensityAtDistance(imB, imB > tB, voxelsize, bins, roi_mask=mask)

        bins_, enrichment_AB, enclosed_AB, enclosed_area_B = edtColoc.image_enrichment_and_fraction_at_distance(imA, imB > tB, voxelsize,
                                                                                               bins, roi_mask=mask)
        bins_, enrichment_BB, enclosed_BB, _ = edtColoc.image_enrichment_and_fraction_at_distance(imB, imB > tB, voxelsize,
                                                                                               bins, roi_mask=mask)
        
        #print binsB, bmB
        
        plots = []
        pnames = []
        
        
        # B from mA
        ####################
        plots_ = {}


        #plots_['Frac. %s from mask(%s)' % (nameB, nameA)] =
        plots.append(enclosed_BA.reshape(-1, 1, 1))
        pnames.append('Frac. %s from mask(%s)' % (nameB, nameA))

        plots.append(enrichment_BA.reshape(-1, 1, 1))
        pnames.append('Enrichment of %s at distance from mask(%s)' % (nameB, nameA))
        
            
        edtColoc.plot_image_dist_coloc_figure(bins_, enrichment_BA, enrichment_AA, enclosed_BA, enclosed_AA, enclosed_area_A, pearson, MA, MB, nameA, nameB)

        plots.append(enclosed_AB.reshape(-1, 1, 1))
        pnames.append('Frac. %s from mask(%s)' % (nameA, nameB))

        plots.append(enrichment_AB.reshape(-1, 1, 1))
        pnames.append('Enrichment of %s at distance from mask(%s)' % (nameA, nameB))

        edtColoc.plot_image_dist_coloc_figure(bins_, enrichment_AB, enrichment_BB, enclosed_AB, enclosed_BB, enclosed_area_B, pearson,
                                              MA, MB, nameB, nameA)
        
        plt.show()
        
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
        try:
            im.mdh['Colocalisation.ThresholdMode'] = self.do.ThreshMode
        except:
            pass

        im.mdh['OriginalImage'] = self.image.filename

        ViewIm3D(im, mode='graph')
        
    def OnColocBasic(self, event):
        from PYME.Analysis.Colocalisation import correlationCoeffs, edtColoc
        voxelsize = self.image.voxelsize_nm
        
        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % n for n in range(self.image.data.shape[3])]
        
        dlg = ColocSettingsDialog(self.dsviewer, voxelsize[0], names, show_bins=False)
        dlg.ShowModal()
        
        chans = dlg.GetChans()
        dlg.Destroy()

        #assume we have exactly 2 channels #FIXME - add a selector
        #grab image data
        imA = self.image.data[:,:,:,chans[0]].squeeze()
        imB = self.image.data[:,:,:,chans[1]].squeeze()

        #assume threshold is half the colour bounds - good if using threshold mode
        tA = self.do.Offs[chans[0]] + .5/self.do.Gains[chans[0]] #plt.mean(self.ivps[0].clim)
        tB = self.do.Offs[chans[1]] + .5/self.do.Gains[chans[1]] #plt.mean(self.ivps[0].clim)
        
        nameA = names[chans[0]]
        nameB = names[chans[1]]

        print('Calculating Pearson and Manders coefficients ...')        
        pearson = correlationCoeffs.pearson(imA, imB)
        MA, MB = correlationCoeffs.thresholdedManders(imA, imB, tA, tB)
        mutual_information = correlationCoeffs.mutual_information(imA, imB)
        
        I1 = imA.ravel()
        I2 = imB.ravel()
        h1 = np.histogram2d(np.clip(I1/I1.mean(), 0, 100), np.clip(I2/I2.mean(), 0, 100), 200)

        plt.figure()
        plt.figtext(.1, .95, 'Pearson: %2.2f   M1: %2.2f M2: %2.2f    MI: %2.3f' % (pearson, MA, MB, mutual_information))
        plt.subplot(111)
        
        plt.imshow(np.log10(h1[0] + .1).T)
        plt.xlabel('%s' % nameA)
        plt.ylabel('%s' % nameB)

        
        plt.show()

    def OnFRC(self, event):
        import matplotlib.pyplot as plt
        from PYME.Analysis import binAvg
        from PYME.Analysis.Colocalisation import correlationCoeffs
        
        voxelsize = self.image.voxelsize

        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % n for n in range(self.image.data.shape[3])]

        dlg = ColocSettingsDialog(self.dsviewer, voxelsize[0], names, show_bins=False)
        dlg.ShowModal()

        chans = dlg.GetChans()
        dlg.Destroy()
    
        #assume we have exactly 2 channels #FIXME - add a selector
        #grab image data
        imA = self.image.data[:, :, :, chans[0]].squeeze()
        imB = self.image.data[:, :, :, chans[1]].squeeze()
        
        correlationCoeffs.fourier_ring_correlation(imA, imB, voxelsize, True)
        
        



def Plug(dsviewer):
    return Colocaliser(dsviewer)



