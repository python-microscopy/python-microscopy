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
import PYME.misc.autoFoldPanel as afp
import numpy

class psfExtractor:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.view = dsviewer.view
        self.do = dsviewer.do
        self.image = dsviewer.image
        
        self.multiChannel = self.image.data.shape[3] > 1

        self.PSFLocs = []
        self.psfROISize = [30,30,30]
        
        dsviewer.do.overlays.append(self.DrawOverlays)

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
    
        vsizer.Add(hsizer, 0,wx.EXPAND|wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 0)        
        
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

        vsizer.Add(hsizer, 0,wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'ROI Size:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPSFROI = wx.TextCtrl(pan, -1, value='30,30,30', size=(40, -1))
        hsizer.Add(self.tPSFROI, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tPSFROI.Bind(wx.EVT_TEXT, self.OnPSFROI)

        vsizer.Add(hsizer, 0,wx.EXPAND|wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Blur:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPSFBlur = wx.TextCtrl(pan, -1, value='.5,.5,1', size=(40, -1))
        hsizer.Add(self.tPSFBlur, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0,wx.EXPAND|wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'PSF Type:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.chType = wx.Choice(pan, -1, choices=['Widefield', 'Confocal'], size=(40, -1))
        hsizer.Add(self.chType, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        if 'Camera.IntegrationTime' in self.image.mdh.getEntryNames():
            #most likely widefield
            self.chType.SetSelection(0)
        else:
            #most likely confocal
            #confocal is a safe default as widefield just does some fancy extra
            #background correction
            self.chType.SetSelection(1)

        vsizer.Add(hsizer, 0,wx.EXPAND|wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 0)


        hsizer = wx.BoxSizer(wx.HORIZONTAL)        
        bExtract = wx.Button(pan, -1, 'Extract', style=wx.BU_EXACTFIT)
        bExtract.Bind(wx.EVT_BUTTON, self.OnExtractPSF)
        hsizer.Add(bExtract, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5) 
        
        bExtractSplit = wx.Button(pan, -1, 'Extract Split', style=wx.BU_EXACTFIT)
        bExtractSplit.Bind(wx.EVT_BUTTON, self.OnExtractSplitPSF)
        hsizer.Add(bExtractSplit, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        
        vsizer.Add(hsizer, 0,wx.ALL|wx.ALIGN_RIGHT, 5)
        
        bAxialShift = wx.Button(pan, -1, 'Estimate axial shift', style=wx.BU_EXACTFIT)
        bAxialShift.Bind(wx.EVT_BUTTON, self.OnCalcShift)
        vsizer.Add(bAxialShift, 0,wx.ALL|wx.ALIGN_RIGHT, 5)

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
        from PYME.PSFEst import extractImages
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
        dx, dy, dz = extractImages.getIntCenter(self.image.data[(self.do.xp-rsx):(self.do.xp+rsx + 1),(self.do.yp-rsy):(self.do.yp+rsy+1), :, chnum])
        self.PSFLocs.append((self.do.xp + dx, self.do.yp + dy, dz))
        self.view.psfROIs = self.PSFLocs
        self.view.Refresh()

    def OnTagPoints(self, event):
        from PYME.PSFEst import extractImages
        chnum = self.chChannel.GetSelection()
        rsx, rsy, rsz = [int(s) for s in self.tPSFROI.GetValue().split(',')]
        for xp, yp, zp in self.view.points:
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
                xp0, yp0 = view._PixelToScreenCoordinates(p[a_x]-self.psfROISize[a_x],p[a_y] - self.psfROISize[a_y])
                xp1, yp1 = view._PixelToScreenCoordinates(p[a_x]+self.psfROISize[a_x],p[a_y] + self.psfROISize[a_y])
                dc.DrawRectangle(xp0, yp0, xp1-xp0,yp1-yp0)

        
    def OnCalcShift(self, event):
        if (len(self.PSFLocs) > 0):
            import pylab
            
            x,y,z = self.PSFLocs[0]
            
            z_ = numpy.arange(self.image.data.shape[2])*self.image.mdh['voxelsize.z']*1.e3
            z_ -= z_.mean()
            
            pylab.figure()
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
            
            pylab.plot(z_, p_0)
            pylab.plot(z_, p_1)
            pylab.vlines(z0, 0, 1)
            pylab.vlines(z1, 0, 1)
            pylab.figtext(.7,.7, 'dz = %3.2f' % dz)
            

    def OnExtractPSF(self, event):
        if (len(self.PSFLocs) > 0):
            from PYME.PSFEst import extractImages
            chnum = self.chChannel.GetSelection()

            psfROISize = [int(s) for s in self.tPSFROI.GetValue().split(',')]
            psfBlur = [float(s) for s in self.tPSFBlur.GetValue().split(',')]
            #print psfROISize
            psf = extractImages.getPSF3D(self.image.data[:,:,:,chnum], self.PSFLocs, psfROISize, psfBlur)
            
            if self.chType.GetSelection() == 0:
                #widefield image - do special background subtraction
                psf = extractImages.backgroundCorrectPSFWF(psf)

#            from pylab import *
#            import cPickle
#            imshow(psf.max(2))

            from PYME.DSView.dsviewer_npy_nb import ImageStack, ViewIm3D

            im = ImageStack(data = psf, mdh = self.image.mdh, titleStub = 'Extracted PSF')
            im.defaultExt = '*.psf' #we want to save as PSF by default
            ViewIm3D(im, mode='psf', parent=wx.GetTopLevelParent(self.dsviewer))
            
    def OnExtractSplitPSF(self, event):
        if (len(self.PSFLocs) > 0):
            from PYME.PSFEst import extractImages
            chnum = self.chChannel.GetSelection()

            psfROISize = [int(s) for s in self.tPSFROI.GetValue().split(',')]
            psfBlur = [float(s) for s in self.tPSFBlur.GetValue().split(',')]
            #print psfROISize

            psfs = []            
            
            for i in range(self.image.data.shape[3]):
                psf = extractImages.getPSF3D(self.image.data[:,:,:,i], self.PSFLocs, psfROISize, psfBlur)
                
                if self.chType.GetSelection() == 0:
                    #widefield image - do special background subtraction
                    psf = extractImages.backgroundCorrectPSFWF(psf)
                    
                psfs.append(psf)
                
            psf = numpy.concatenate(psfs, 0)

#            from pylab import *
#            import cPickle
#            imshow(psf.max(2))

            from PYME.DSView.dsviewer_npy_nb import ImageStack, ViewIm3D

            im = ImageStack(data = psf, mdh = self.image.mdh, titleStub = 'Extracted PSF')
            im.defaultExt = '*.psf' #we want to save as PSF by default
            ViewIm3D(im, mode='psf', parent=wx.GetTopLevelParent(self.dsviewer))

#            fdialog = wx.FileDialog(None, 'Save PSF as ...',
#                wildcard='PSF file (*.psf)|*.psf|H5P file (*.h5p)|*.h5p', style=wx.SAVE|wx.HIDE_READONLY)
#            succ = fdialog.ShowModal()
#            if (succ == wx.ID_OK):
#                fpath = fdialog.GetPath()
#                #save as a pickle containing the data and voxelsize
#
#                if fpath.endswith('.psf'):
#                    fid = open(fpath, 'wb')
#                    cPickle.dump((psf, self.image.mdh.voxelsize), fid, 2)
#                    fid.close()
#                else:
#                    import tables
#                    h5out = tables.openFile(fpath,'w')
#                    filters=tables.Filters(5,'zlib',shuffle=True)
#
#                    xSize, ySize, nFrames = psf.shape
#
#                    ims = h5out.createEArray(h5out.root,'PSFData',tables.Float32Atom(),(0,xSize,ySize), filters=filters, expectedrows=nFrames)
#                    for frameN in range(nFrames):
#                        ims.append(psf[:,:,frameN][None, :,:])
#                        ims.flush()
#
#                    outMDH = MetaDataHandler.HDFMDHandler(h5out)
#
#                    outMDH.copyEntriesFrom(self.image.mdh)
#                    outMDH.setEntry('psf.originalFile', self.seriesName)
#
#                    h5out.flush()
#                    h5out.close()

def Plug(dsviewer):
    dsviewer.psfExtractor = psfExtractor(dsviewer)

