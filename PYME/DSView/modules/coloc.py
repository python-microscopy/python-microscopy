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
import pylab

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

class colocaliser:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        dsviewer.mProcessing.AppendSeparator()
        dsviewer.AddMenuItem('Processing', "&Colocalisation", self.OnColocBasic)
        dsviewer.AddMenuItem('Processing', "EDT Colocalisation", self.OnColoc)
        dsviewer.AddMenuItem('Processing', "EDT Colocalisation (z cropped)", lambda e: self.OnColoc(restrict_z=True))
        dsviewer.AddMenuItem('Processing', 'FRC', self.OnFRC)
    


    
    def OnColoc(self, event=None, restrict_z=False, coloc_vs_z = False):
        from PYME.Analysis.Colocalisation import correlationCoeffs, edtColoc
        from scipy import interpolate
        voxelsize = [1e3*self.image.mdh.getEntry('voxelsize.x') ,1e3*self.image.mdh.getEntry('voxelsize.y'), 1e3*self.image.mdh.getEntry('voxelsize.z')]
        
        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % n for n in range(self.image.data.shape[3])]
            
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
        tA = self.do.Offs[chans[0]] + .5/self.do.Gains[chans[0]] #pylab.mean(self.ivps[0].clim)
        tB = self.do.Offs[chans[1]] + .5/self.do.Gains[chans[1]] #pylab.mean(self.ivps[0].clim)
        
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
    
                pylab.figure()
                pylab.subplot(211)
                if 'filename' in self.image.__dict__:
                    pylab.title(self.image.filename)
                # nameB with nameA
                cAB, = pylab.plot(MAzs,'o', label = '%s with %s' %(nameB,nameA))
                # nameA with nameB
                cBA, = pylab.plot(MBzs,'*', label = '%s with %s' %(nameA,nameB))
                pylab.legend([cBA,cAB])
                pylab.xlabel('z slice level')
                pylab.ylabel('Manders coloc fraction')
                pylab.ylim(0,None)
    
                pylab.subplot(212)
                if 'filename' in self.image.__dict__:
                    pylab.title(self.image.filename)
                # nameB with nameA
                fA, = pylab.plot(FAzs,'o', label = '%s mask fraction' %(nameA))
                # nameA with nameB
                fB, = pylab.plot(FBzs,'*', label = '%s mask fraction' %(nameB))
                pylab.legend([fA,fB])
                pylab.xlabel('z slice level')
                pylab.ylabel('Mask fraction')
                pylab.ylim(0,None)
                pylab.show()

        print('Performing distance transform ...')
        #bnA, bmA, binsA = edtColoc.imageDensityAtDistance(imB, imA > tA, voxelsize, bins, roi_mask=mask)
        #bnAA, bmAA, binsA = edtColoc.imageDensityAtDistance(imA, imA > tA, voxelsize, bins, roi_mask=mask)
        
        bins_, enrichment_BA, enclosed_BA = edtColoc.image_enrichment_and_fraction_at_distance(imB, imA > tA, voxelsize,
                                                                                               bins, roi_mask=mask)
        bins_, enrichment_AA, enclosed_AA = edtColoc.image_enrichment_and_fraction_at_distance(imA, imA > tA, voxelsize,
                                                                                               bins, roi_mask=mask)
        
        print('Performing distance transform (reversed) ...')
        #bnB, bmB, binsB = edtColoc.imageDensityAtDistance(imA, imB > tB, voxelsize, bins, roi_mask=mask)
        #bnBB, bmBB, binsB = edtColoc.imageDensityAtDistance(imB, imB > tB, voxelsize, bins, roi_mask=mask)

        bins_, enrichment_AB, enclosed_AB = edtColoc.image_enrichment_and_fraction_at_distance(imA, imB > tB, voxelsize,
                                                                                               bins, roi_mask=mask)
        bins_, enrichment_BB, enclosed_BB = edtColoc.image_enrichment_and_fraction_at_distance(imB, imB > tB, voxelsize,
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
        
            
        edtColoc.plot_image_dist_coloc_figure(bins_, enrichment_BA, enrichment_AA, enclosed_BA, enclosed_AA, pearson, MA, MB, nameA, nameB)

        plots.append(enclosed_AB.reshape(-1, 1, 1))
        pnames.append('Frac. %s from mask(%s)' % (nameA, nameB))

        plots.append(enrichment_AB.reshape(-1, 1, 1))
        pnames.append('Enrichment of %s at distance from mask(%s)' % (nameA, nameB))

        edtColoc.plot_image_dist_coloc_figure(bins_, enrichment_AB, enrichment_BB, enclosed_AB, enclosed_BB, pearson,
                                              MA, MB, nameB, nameA)
        
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
        try:
            im.mdh['Colocalisation.ThresholdMode'] = self.do.ThreshMode
        except:
            pass

        im.mdh['OriginalImage'] = self.image.filename

        ViewIm3D(im, mode='graph')
        
    def OnColocBasic(self, event):
        from PYME.Analysis.Colocalisation import correlationCoeffs, edtColoc
        voxelsize = [1e3*self.image.mdh.getEntry('voxelsize.x') ,1e3*self.image.mdh.getEntry('voxelsize.y'), 1e3*self.image.mdh.getEntry('voxelsize.z')]
        
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
        tA = self.do.Offs[chans[0]] + .5/self.do.Gains[chans[0]] #pylab.mean(self.ivps[0].clim)
        tB = self.do.Offs[chans[1]] + .5/self.do.Gains[chans[1]] #pylab.mean(self.ivps[0].clim)
        
        nameA = names[chans[0]]
        nameB = names[chans[1]]

        print('Calculating Pearson and Manders coefficients ...')        
        pearson = correlationCoeffs.pearson(imA, imB)
        MA, MB = correlationCoeffs.thresholdedManders(imA, imB, tA, tB)
        mutual_information = correlationCoeffs.mutual_information(imA, imB)
        
        I1 = imA.ravel()
        I2 = imB.ravel()
        h1 = np.histogram2d(np.clip(I1/I1.mean(), 0, 100), np.clip(I2/I2.mean(), 0, 100), 200)

        pylab.figure()
        pylab.figtext(.1, .95, 'Pearson: %2.2f   M1: %2.2f M2: %2.2f    MI: %2.3f' % (pearson, MA, MB, mutual_information))
        pylab.subplot(111)
        
        pylab.imshow(np.log10(h1[0] + .1).T)
        pylab.xlabel('%s' % nameA)
        pylab.ylabel('%s' % nameB)

        
        pylab.show()

    def OnFRC(self, event):
        import matplotlib.pyplot as plt
        from PYME.Analysis import binAvg
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
    
        X, Y = np.mgrid[0:float(imA.shape[0]), 0:float(imA.shape[1])]
        X = X / X.shape[0]
        Y = Y / X.shape[1]
        X = (X - .5)
        Y = Y - .5
        R = np.sqrt(X ** 2 + Y ** 2)
    
        H1 = np.fft.fftn(imA)
        H2 = np.fft.fftn(imB)
    
        #rB = np.linspace(0,R.max())
        rB = np.linspace(0, 0.5, 100)
    
        bn, bm, bs = binAvg.binAvg(R, np.fft.fftshift(H1 * H2.conjugate()).real, rB)
    
        bn1, bm1, bs1 = binAvg.binAvg(R, np.fft.fftshift((H1 * H1.conjugate()).real), rB)
        bn2, bm2, bs2 = binAvg.binAvg(R, np.fft.fftshift((H2 * H2.conjugate()).real), rB)
    
        plt.figure()
    
        ax = plt.gca()
    
        #FRC
        FRC = bm / np.sqrt(bm1 * bm2)
        ax.plot(rB[:-1], FRC)
    
        #noise envelope???????????
        ax.plot(rB[:-1], 2. / np.sqrt(bn / 2), ':')
    
        dfrc = np.diff(FRC)
        monotone = np.where(dfrc > 0)[0][0] + 1
    
        #print FRC[:monotone], FRC[:(monotone+1)]
    
        intercept_m = np.interp(1.0 - 1 / 7.0, 1 - FRC[:monotone], rB[:monotone])
    
        print('Intercept_m= %3.2f (%3.2f nm)' % (intercept_m, voxelsize[0] / intercept_m))

        from scipy import ndimage
        f_s = np.sign(FRC - 1. / 7.)

        intercept = np.interp(0.0, - ndimage.gaussian_filter(f_s, 10), rB[:-1])

        print('Intercept= %3.2f (%3.2f nm)' % (intercept, voxelsize[0] / intercept))
    
        xt = np.array([10., 15, 20, 30, 40, 50, 70, 90, 120, 150, 200, 300, 500])
        rt = voxelsize[0] / xt
    
        plt.xticks(rt[::-1], ['%d' % xi for xi in xt[::-1]], rotation='vertical')
    
        ax.plot([0, rt[0]], np.ones(2) / 7.0)
    
        plt.grid()
        plt.xlabel('Resolution [nm]')
    
        plt.ylabel('FRC')
        
        plt.plot([intercept, intercept], [0,1], '--')
    
        plt.figtext(0.5, 0.5, 'FRC intercept at %3.1f nm' % (voxelsize[0] / intercept))
    
        # plt.figure()
        #
        #
        # plt.plot(rB[:-1], FRC-1./7.)
        # plt.plot(rB[:-1], f_s)
        # plt.plot(rB[:-1], ndimage.gaussian_filter(f_s, 1))
        # plt.plot(rB[:-1], ndimage.gaussian_filter(f_s, 5))
        # plt.plot(rB[:-1], ndimage.gaussian_filter(f_s, 10))
        #
        # plt.show()
        
        



def Plug(dsviewer):
    dsviewer.coloc = colocaliser(dsviewer)



