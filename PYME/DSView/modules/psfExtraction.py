#!/usr/bin/python
##################
# psfExtraction.py
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
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
import numpy

from ._base import Plugin

import logging
logger = logging.getLogger(__name__)

class PsfExtractor(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)

        try:  # stack multiview channels
            self.numChan = self.image.mdh['Multiview.NumROIs']
        except:
            self.ChanOffsetZnm = None
            self.numChan = self.image.data.shape[3]

        self.PSFLocs = []
        self.psfROISize = [30,30,30]
        
        dsviewer.view.add_overlay(self.DrawOverlays, 'PSF ROIs')

        dsviewer.paneHooks.append(self.GenPSFPanel)

    def GenPSFPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="PSF Extraction", pinned = True)

        pan = wx.Panel(item, -1)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        #if self.multiChannel: #we have channels            
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Channel:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.chChannel = wx.Choice(pan, -1, choices=self.do.names)
        hsizer.Add(self.chChannel, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
    
        vsizer.Add(hsizer, 0,wx.EXPAND|wx.ALL, 0)        
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#
#        hsizer.Add(wx.StaticText(pan, -1, 'Threshold:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#        self.tThreshold = wx.TextCtrl(pan, -1, value='50', size=(40, -1))
#
        bTagPSF = wx.Button(pan, -1, 'Tag', style=wx.BU_EXACTFIT)
        bTagPSF.Bind(wx.EVT_BUTTON, self.OnTagPSF)
        hsizer.Add(bTagPSF, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bTagPoints = wx.Button(pan, -1, 'Tag Points', style=wx.BU_EXACTFIT)
        bTagPoints.Bind(wx.EVT_BUTTON, self.OnTagPoints)
        hsizer.Add(bTagPoints, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bClearTagged = wx.Button(pan, -1, 'Clear', style=wx.BU_EXACTFIT)
        bClearTagged.Bind(wx.EVT_BUTTON, self.OnClearTags)
        hsizer.Add(bClearTagged, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0,wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'ROI Half Size:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPSFROI = wx.TextCtrl(pan, -1, value='30,30,30', size=(40, -1))
        hsizer.Add(self.tPSFROI, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tPSFROI.Bind(wx.EVT_TEXT, self.OnPSFROI)

        vsizer.Add(hsizer, 0,wx.EXPAND|wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Blur:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPSFBlur = wx.TextCtrl(pan, -1, value='.5,.5,1', size=(40, -1))
        hsizer.Add(self.tPSFBlur, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0,wx.EXPAND|wx.ALL, 0)
        
        # hsizer = wx.BoxSizer(wx.HORIZONTAL)
        # hsizer.Add(wx.StaticText(pan, -1, 'PSF Type:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        # self.chType = wx.Choice(pan, -1, choices=['Widefield', 'Confocal'], size=(40, -1))
        # hsizer.Add(self.chType, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        # if 'Camera.IntegrationTime' in self.image.mdh.getEntryNames():
        #     #most likely widefield
        #     self.chType.SetSelection(0)
        # else:
        #     #most likely confocal
        #     #confocal is a safe default as widefield just does some fancy extra
        #     #background correction
        #     self.chType.SetSelection(1)

        # vsizer.Add(hsizer, 0,wx.EXPAND|wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Method:'), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.chMethod = wx.Choice(pan, -1, choices=['Standard', 'Split', 'Multi-channel'], size=(40, -1))
        hsizer.Add(self.chMethod, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.chMethod.SetSelection(0) # most common option - also a good default with a splitter assuming aberrations are not very different between channels.
        
        if self.image.data.shape[3] == 4:
            #most likely the high-throughput system, default to multi-view extraction
            self.chMethod.SetSelection(0)

        vsizer.Add(hsizer, 0, wx.EXPAND | wx.ALL, 0)

        self.cbAlignZ = wx.CheckBox(pan, -1, 'Align Z')
        self.cbAlignZ.SetValue(True)
        vsizer.Add(self.cbAlignZ, 0, wx.ALL, 5)
        
        self.cbBackgroundCorrect = wx.CheckBox(pan, -1, 'Widefield background correction:')
        self.cbBackgroundCorrect.SetValue(False)
        vsizer.Add(self.cbBackgroundCorrect, 0, wx.ALL, 5)

        self.cbExpandROI = wx.CheckBox(pan, -1, 'Expand z ROI')
        self.cbExpandROI.SetValue(False)
        vsizer.Add(self.cbExpandROI, 0, wx.ALL, 5)


        hsizer = wx.BoxSizer(wx.HORIZONTAL)        
        bExtract = wx.Button(pan, -1, 'Extract', style=wx.BU_EXACTFIT)
        bExtract.Bind(wx.EVT_BUTTON, self.OnExtract)
        hsizer.Add(bExtract, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5) 
        
        # bExtractSplit = wx.Button(pan, -1, 'Extract Split', style=wx.BU_EXACTFIT)
        # bExtractSplit.Bind(wx.EVT_BUTTON, self.OnExtractSplitPSF)
        # hsizer.Add(bExtractSplit, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        
        vsizer.Add(hsizer, 0,wx.ALL|wx.ALIGN_RIGHT, 5)
        
        bAxialShift = wx.Button(pan, -1, 'Estimate axial shift', style=wx.BU_EXACTFIT)
        bAxialShift.Bind(wx.EVT_BUTTON, self.OnCalcShift)
        vsizer.Add(bAxialShift, 0,wx.ALL|wx.ALIGN_RIGHT, 5)

        # bExtractMultColour = wx.Button(pan, -1, 'Extract Multi Colour', style=wx.BU_EXACTFIT)
        # bExtractMultColour.Bind(wx.EVT_BUTTON, self.OnExtractMultiviewPSF)
        # vsizer.Add(bExtractMultColour, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        #bCalibrateMultiview = wx.Button(pan, -1, 'Calibrate multiview astigmatism', style=wx.BU_EXACTFIT)
        #bCalibrateMultiview.Bind(wx.EVT_BUTTON, self.OnCalibrateMultiview)
        #vsizer.Add(bCalibrateMultiview, 0,wx.ALL|wx.ALIGN_RIGHT, 5)

        pan.SetSizer(vsizer)
        vsizer.Fit(pan)

        #_pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        item.AddNewElement(pan)
        _pnl.AddPane(item)
#
#        self.cbSNThreshold = wx.CheckBox(item, -1, 'SNR Threshold')
#        self.cbSNThreshold.SetValue(False)
#
#        _pnl.AddFoldPanelWindow(item, self.cbSNThreshold, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)


    def OnTagPSF(self, event):
        from PYME.Analysis.PSFEst import extractImages
        #if we already have a location there, un-tag it
        for i, p in enumerate(self.PSFLocs):
            if ((numpy.array(p[:2]) - numpy.array((self.do.xp, self.do.yp)))**2).sum() < 100:
                self.PSFLocs.pop(i)

                #self.view.psfROIs = self.PSFLocs
                self.view.Refresh()
                return
                
        #if we have a muilt-colour stack, 
        chnum = self.chChannel.GetSelection()
                
        rsx, rsy, rsz = [int(s) for s in self.tPSFROI.GetValue().split(',')]
        #print self.do.xp-rsx, self.do.xp+rsx + 1, self.do.yp-rsy, self.do.yp+rsy+1, chnum
        #print self.image.data[(self.do.xp-rsx):(self.do.xp+rsx + 1),(self.do.yp-rsy):(self.do.yp+rsy+1), :, chnum]
        dx, dy, dz = extractImages.getIntCenter(self.image.data[(self.do.xp-rsx):(self.do.xp+rsx + 1),(self.do.yp-rsy):(self.do.yp+rsy+1), :, chnum])
        self.PSFLocs.append((self.do.xp + dx, self.do.yp + dy, dz))
        self.view.psfROIs = self.PSFLocs
        self.view.Refresh()

    def OnTagPoints(self, event):
        from PYME.Analysis.PSFEst import extractImages
        chnum = self.chChannel.GetSelection()
        rsx, rsy, rsz = [int(s) for s in self.tPSFROI.GetValue().split(',')]
        try:
            pts = self.dsviewer.blobFinding.points
        except AttributeError:
            raise AttributeError('Could not find blobFinding.points, make sure the `blobFinding` module is loaded and you have clicked `Find`')

        for xp, yp, zp in pts:
            if ((xp > rsx) and (xp < (self.image.data.shape[0] - rsx)) and
                (yp > rsy) and (yp < (self.image.data.shape[1] - rsy))):
                    
                    dx, dy, dz = extractImages.getIntCenter(self.image.data[(xp-rsx):(xp+rsx + 1),(yp-rsy):(yp+rsy+1), :, chnum])
                    self.PSFLocs.append((xp + dx, yp + dy, dz))
        
        #self.view.psfROIs = self.PSFLocs
        self.view.Refresh()

    def OnClearTags(self, event):
        self.PSFLocs = []
        #self.view.psfROIs = self.PSFLocs
        self.view.Refresh()

    def OnPSFROI(self, event):
        try:
            self.psfROISize = [int(s) for s in self.tPSFROI.GetValue().split(',')]
            #self.view.psfROISize = psfROISize
            self.view.Refresh()
        except:
            pass
        
    def DrawOverlays(self, view, dc):
        #PSF ROIs
        if (len(self.PSFLocs) > 0):
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.SetPen(wx.Pen(wx.TheColourDatabase.FindColour('GREEN'),1))
            
            if(view.do.slice == view.do.SLICE_XY):
                a_x = 0
                a_y = 1
            elif(view.do.slice == view.do.SLICE_XZ):
                a_x = 0
                a_y = 2
            elif(view.do.slice == view.do.SLICE_YZ):
                a_x = 1
                a_y = 2
                
            for p in self.PSFLocs:
                #dc.DrawRectangle(sc*p[0]-self.psfROISize[0]*sc - x0,sc*p[1] - self.psfROISize[1]*sc - y0, 2*self.psfROISize[0]*sc,2*self.psfROISize[1]*sc)
                xp0, yp0 = view.pixel_to_screen_coordinates(p[a_x]-self.psfROISize[a_x],p[a_y] - self.psfROISize[a_y])
                xp1, yp1 = view.pixel_to_screen_coordinates(p[a_x]+self.psfROISize[a_x],p[a_y] + self.psfROISize[a_y])
                dc.DrawRectangle(xp0, yp0, xp1-xp0,yp1-yp0)

        
    def OnCalcShift(self, event):
        if (len(self.PSFLocs) > 0):
            # import pylab
            import matplotlib.pyplot as plt
            
            x,y,z = self.PSFLocs[0]
            x, y = int(x), int(y)  # int cast for indexing
            
            z_ = numpy.arange(self.image.data.shape[2])*self.image.mdh['voxelsize.z']*1.e3
            z_ -= z_.mean()

            plt.figure()
            p_0 = 1.0*self.image.data[x,y,:,0].squeeze()
            p_0  -= p_0.min()
            p_0 /= p_0.max()

            #print (p_0*z_).sum()/p_0.sum()

            p0b = numpy.maximum(p_0 - 0.5, 0)
            z0 = (p0b*z_).sum()/p0b.sum()

            p_1 = 1.0*self.image.data[x,y,:,1].squeeze()
            p_1 -= p_1.min()
            p_1 /= p_1.max()
            
            p1b = numpy.maximum(p_1 - 0.5, 0)
            z1 = (p1b*z_).sum()/p1b.sum()
            
            dz = z1 - z0

            print(('z0: %f, z1: %f, dz: %f' % (z0,z1,dz)))
            
            plt.plot(z_, p_0)
            plt.plot(z_, p_1)
            plt.vlines(z0, 0, 1)
            plt.vlines(z1, 0, 1)
            plt.figtext(.7,.7, 'dz = %3.2f' % dz)
            
    def OnExtract(self, event):
        method = self.chMethod.GetStringSelection()
        
        if method == 'Standard':
            return self.OnExtractPSF(event)
        elif method == 'Split':
            return  self.OnExtractSplitPSF(event)
        elif method == 'Multi-channel':
            return  self.OnExtractMultiviewPSF(event)

    def OnExtractPSF(self, event):
        if (len(self.PSFLocs) > 0):
            from PYME.Analysis.PSFEst import extractImages
            chnum = self.chChannel.GetSelection()

            psfROISize = [int(s) for s in self.tPSFROI.GetValue().split(',')]
            psfBlur = [float(s) for s in self.tPSFBlur.GetValue().split(',')]
            #print psfROISize
            psf, offsets = extractImages.getPSF3D(self.image.data[:,:,:,chnum], self.PSFLocs, psfROISize, psfBlur,
                                                  expand_z=self.cbExpandROI.GetValue())
            
            if self.cbBackgroundCorrect.GetValue():
                #widefield image - do special background subtraction
                psf = extractImages.backgroundCorrectPSFWF(psf)

            from PYME.DSView.dsviewer import ImageStack, ViewIm3D
            from PYME.IO.MetaDataHandler import NestedClassMDHandler

            mdh = NestedClassMDHandler(self.image.mdh)
            self.write_metadata(mdh, 'default')
            mdh['ImageType']='PSF'

            im = ImageStack(data = psf, mdh = mdh, titleStub = 'Extracted PSF')
            im.defaultExt = '*.tif' #we want to save as PSF by default
            
            ViewIm3D(im, mode='psf', parent=wx.GetTopLevelParent(self.dsviewer))

    def OnExtractMultiviewPSF(self, event):
        if (len(self.PSFLocs) > 0):
            from PYME.Analysis.PSFEst import extractImages

            psfROISize = [int(s) for s in self.tPSFROI.GetValue().split(',')]
            psfBlur = [float(s) for s in self.tPSFBlur.GetValue().split(',')]

            psfs = []

            for chnum in range(self.image.data.shape[3]):
                alignZ = (chnum > 0) and self.cbAlignZ.GetValue()
                #always align the first channel

                psf, offsets = extractImages.getPSF3D(self.image.data[:, :, :, chnum], self.PSFLocs, psfROISize, psfBlur,
                                                      centreZ=alignZ, expand_z=self.cbExpandROI.GetValue())

                if self.cbBackgroundCorrect.GetValue():
                    #widefield image - do special background subtraction
                    psf = extractImages.backgroundCorrectPSFWF(psf)

                psfs.append(psf)


            from PYME.DSView.dsviewer import ImageStack, ViewIm3D
            from PYME.IO.MetaDataHandler import NestedClassMDHandler

            mdh = NestedClassMDHandler(self.image.mdh)
            self.write_metadata(mdh, 'Multiview')
            mdh['ImageType'] = 'PSF'

            im = ImageStack(data=psfs, mdh=mdh, titleStub='Extracted PSF')
            im.defaultExt = '*.tif' #we want to save as PSF by default
            ViewIm3D(im, mode='psf', parent=wx.GetTopLevelParent(self.dsviewer))

    def OnCalibrateMultiview(self, event):
        """
        CalibrateMultiview loops over all channels described in the metadata and runs CalibrateAstigmatism on each one.
        Results are stored in a library of dictionaries and are saved into a jason astigmatism map file (.am)
        Args:
            event: GUI event

        Returns:
            nothing

        """
        # first make sure the user is calling the right function
        try:
            chanSizeX = self.image.mdh['Multiview.ROISize'][0]
        except KeyError:
            raise KeyError('You are not looking at multiview data, or the metadata is incomplete')

        if len(self.PSFLocs) > 0:
            print('Place cursor on bead in first multiview channel and press Calibrate Multiview Astigmatism')
            self.OnClearTags(event)
            return

        from PYME.Analysis.PSFEst import extractImages
        from PYME.DSView.modules import psfTools
        import scipy.interpolate as terp
        import numpy as np
        from PYME.IO.FileUtils import nameUtils
        import os
        import json

        zrange = np.nan*np.ones(2)

        rsx, rsy, rsz = [int(s) for s in self.tPSFROI.GetValue().split(',')]
        # astigDat = []
        astigLib = {}
        for ii in range(self.numChan):
            # grab PSFs, currently relying on user to pick psf in first channel
            xmin, xmax = [(self.do.xp-rsx + ii*chanSizeX), (self.do.xp+rsx + ii*chanSizeX + 1)]
            # dz returns offset in units of frames, dx dy are in pixels
            dx, dy, dz = extractImages.getIntCenter(self.image.data[xmin:xmax,
                                                    (self.do.yp-rsy):(self.do.yp+rsy+1), :, 0])
            self.PSFLocs.append((self.do.xp + dx, self.do.yp + dy, dz))
            self.view.psfROIs = self.PSFLocs
            self.view.Refresh()

            psfROISize = [int(s) for s in self.tPSFROI.GetValue().split(',')]
            psfBlur = [float(s) for s in self.tPSFBlur.GetValue().split(',')]

            psf = extractImages.getPSF3D(self.image.data[:, :, :, 0],
                                         [self.PSFLocs[ii]], psfROISize, psfBlur)


            if self.cbBackgroundCorrect.GetValue():
                #widefield image - do special background subtraction
                psf = extractImages.backgroundCorrectPSFWF(psf)

            from PYME.DSView.dsviewer import ImageStack, ViewIm3D

            im = ImageStack(data=psf, mdh=self.image.mdh, titleStub='Extracted PSF, Chan %i' % ii)
            im.defaultExt = '*.psf' #we want to save as PSF by default
            ViewIm3D(im, mode='psf', parent=wx.GetTopLevelParent(self.dsviewer))

            calibrater = psfTools.PSFTools(self.dsviewer, im)
            astigLib['PSF%i' % ii] = (calibrater.OnCalibrateAstigmatism(event, plotIt=False))
            # the next line is only ok if each frame is at a different z position, which it should be
            astigLib['PSF%i' % ii]['z'] += dz * self.image.mdh['voxelsize.z'] * 1.e3

            # -- spline interpolate --
            # find region of dsigma which is monotonic
            dsig = terp.UnivariateSpline(astigLib['PSF%i' % ii]['z'], astigLib['PSF%i' % ii]['dsigma'])#, s=1.5*len(astigDat[ii]['z']))

            # mask where the sign is the same as the center
            zvec = np.linspace(astigLib['PSF%i' % ii]['z'].min(), astigLib['PSF%i' % ii]['z'].max(), 1000)
            sgn = np.sign(np.diff(dsig(zvec)))
            halfway = len(sgn)/2
            notmask = sgn != sgn[halfway]

            # find z range for spline generation
            lowerZ = zvec[np.where(notmask[:halfway])[0].max()]
            upperZ = zvec[(len(sgn)/2 + np.where(notmask[halfway:])[0].min() - 1)]
            astigLib['PSF%i' % ii]['zrange'] = [lowerZ, upperZ]
            zrange = [np.nanmin([lowerZ, zrange[0]]), np.nanmax([upperZ, zrange[1]])]

            #lowsubZ , upsubZ = np.absolute(astigDat[ii]['z'] - zvec[lowerZ]), np.absolute(astigDat[ii]['z'] - zvec[upperZ])
            #lowZLoc = np.argmin(lowsubZ)
            #upZLoc = np.argmin(upsubZ)

            #
            #astigLib['sigxTerp%i' % ii] = terp.UnivariateSpline(astigLib['PSF%i' % ii]['z'], astigLib['PSF%i' % ii]['sigmax'],
            #                                                    bbox=[lowerZ, upperZ])
            #astigLib['sigyTerp%i' % ii] = terp.UnivariateSpline(astigLib['PSF%i' % ii]['z'], astigLib['PSF%i' % ii]['sigmay'],
            #                                                    bbox=[lowerZ, upperZ])
            astigLib['PSF%i' % ii]['z'] = astigLib['PSF%i' % ii]['z'].tolist()

        astigLib['zRange'] = np.round(zrange).tolist()
        astigLib['numChan'] = self.numChan


        psfTools.plotAstigCalibration(astigLib)



        # save to json file
        #defFile = os.path.splitext(os.path.split(self.visFr.GetTitle())[-1])[0] + '.am'

        fdialog = wx.FileDialog(None, 'Save Astigmatism Calibration as ...',
            wildcard='AstigMAPism file (*.am)|*.am', style=wx.FD_SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath())  #, defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()

            fid = open(fpath, 'wb')
            json.dump(astigLib, fid)
            fid.close()


            
    def OnExtractSplitPSF(self, event):
        if (len(self.PSFLocs) > 0):
            from PYME.Analysis.PSFEst import extractImages
            chnum = self.chChannel.GetSelection()

            psfROISize = [int(s) for s in self.tPSFROI.GetValue().split(',')]
            psfBlur = [float(s) for s in self.tPSFBlur.GetValue().split(',')]
            #print psfROISize

            psfs = []
            offsetsAllChannel = []

            #extract first channel (always aligned)
            psf, offsets = extractImages.getPSF3D(self.image.data[:,:,:,0].squeeze(), self.PSFLocs, psfROISize, psfBlur, centreZ=True)
            if self.chType.GetSelection() == 0:
                #widefield image - do special background subtraction
                psf = extractImages.backgroundCorrectPSFWF(psf)

            psfs.append(psf)
            offsetsAllChannel.append(offsets)
            alignZ = self.cbAlignZ.GetValue()
            z_offset = offsets[2]

            #extract subsequent channels, aligning if necessary, otherwise offsetting by the calculated offset for the first channel
            for i in range(1, self.image.data.shape[3]):
                psf, offsets = extractImages.getPSF3D(self.image.data[:,:,:,i].squeeze(), self.PSFLocs, psfROISize, psfBlur, centreZ=alignZ, z_offset=z_offset)

                if self.cbBackgroundCorrect.GetValue():
                    #widefield image - do special background subtraction
                    psf = extractImages.backgroundCorrectPSFWF(psf)

                psfs.append(psf)
                offsetsAllChannel.append(offsets)
                
            psf = numpy.concatenate(psfs, 0)
            offsetsAllChannel = numpy.asarray(offsetsAllChannel)
            offsetsAllChannel -= offsetsAllChannel[0]
            print(offsetsAllChannel)

#            from pylab import *
#            import cPickle
#            imshow(psf.max(2))

            from PYME.DSView.dsviewer import ImageStack, ViewIm3D
            from PYME.IO.MetaDataHandler import NestedClassMDHandler

            mdh = NestedClassMDHandler(self.image.mdh)
            self.write_metadata(mdh, 'Split', offsetsAllChannel[:,2])
            mdh['ImageType'] = 'PSF'

            im = ImageStack(data = psf, mdh = mdh, titleStub = 'Extracted PSF')
            im.defaultExt = '*.tif' #we want to save as PSF by default
            ViewIm3D(im, mode='psf', parent=wx.GetTopLevelParent(self.dsviewer))

    def write_metadata(self, mdh, mode, axialshift=None):
        mdh['PSFExtraction.Mode'] = mode
        mdh['PSFExtraction.ROI'] = [int(s) for s in self.tPSFROI.GetValue().split(',')]
        mdh['PSFExtraction.Blur'] = [float(s) for s in self.tPSFBlur.GetValue().split(',')]
        #mdh['PSFExtraction.Type'] = self.chType.GetStringSelection()
        mdh['PSFExtraction.WidefieldBackgroundCorrection'] = self.cbBackgroundCorrect.GetValue()
        mdh['PSFExtraction.Method'] = self.chMethod.GetStringSelection()
        mdh['PSFExtraction.Locations'] = self.PSFLocs
#        mdh['PSF_Extraction.Normalize'] = self.chNormalize.GetStringSelection()
        if axialshift is not None:
            try:
                mdh['PSFExtraction.shift.units'] = self.image.mdh['voxelsize.units']
                mdh['PSFExtraction.shift.z'] = -axialshift*self.image.mdh['voxelsize.z']
            except:
                mdh['PSFExtraction.shift.units'] = 'pixel'
                mdh['PSFExtraction.shift.z'] = -axialshift


def Plug(dsviewer):
    return PsfExtractor(dsviewer)

