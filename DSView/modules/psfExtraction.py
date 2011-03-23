#!/usr/bin/python
##################
# psfExtraction.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import wx
import PYME.misc.autoFoldPanel as afp

class psfExtractor:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.vp = dsviewer.vp
        self.image = dsviewer.image

        self.PSFLocs = []

        dsviewer.paneHooks.append(self.GenPSFPanel)

    def GenPSFPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="PSF Extraction", pinned = True)

        pan = wx.Panel(item, -1)

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#
#        hsizer.Add(wx.StaticText(pan, -1, 'Threshold:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#        self.tThreshold = wx.TextCtrl(pan, -1, value='50', size=(40, -1))
#
        bTagPSF = wx.Button(pan, -1, 'Tag', style=wx.BU_EXACTFIT)
        bTagPSF.Bind(wx.EVT_BUTTON, self.OnTagPSF)
        hsizer.Add(bTagPSF, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

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

        bExtract = wx.Button(pan, -1, 'Extract', style=wx.BU_EXACTFIT)
        bExtract.Bind(wx.EVT_BUTTON, self.OnExtractPSF)
        vsizer.Add(bExtract, 0,wx.ALL|wx.ALIGN_RIGHT, 5)

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
        rsx, rsy, rsz = [int(s) for s in self.tPSFROI.GetValue().split(',')]
        dx, dy, dz = extractImages.getIntCenter(self.image.data[(self.vp.do.xp-rsx):(self.vp.do.xp+rsx + 1),(self.vp.do.yp-rsy):(self.vp.do.yp+rsy+1), :])
        self.PSFLocs.append((self.vp.do.xp + dx, self.vp.do.yp + dy, dz))
        self.vp.view.psfROIs = self.PSFLocs
        self.vp.view.Refresh()

    def OnClearTags(self, event):
        self.PSFLocs = []
        self.vp.view.psfROIs = self.PSFLocs
        self.vp.view.Refresh()

    def OnPSFROI(self, event):
        try:
            psfROISize = [int(s) for s in self.tPSFROI.GetValue().split(',')]
            self.vp.view.psfROISize = psfROISize
            self.vp.Refresh()
        except:
            pass

    def OnExtractPSF(self, event):
        if (len(self.PSFLocs) > 0):
            from PYME.PSFEst import extractImages

            psfROISize = [int(s) for s in self.tPSFROI.GetValue().split(',')]
            psfBlur = [float(s) for s in self.tPSFBlur.GetValue().split(',')]
            #print psfROISize
            psf = extractImages.getPSF3D(self.image.data, self.PSFLocs, psfROISize, psfBlur)

#            from pylab import *
#            import cPickle
#            imshow(psf.max(2))

            from PYME.DSView.dsviewer_npy_nb import ImageStack, ViewIm3D

            im = ImageStack(data = psf, mdh = self.image.mdh)
            im.defaultExt = '*.psf' #we want to save as PSF by default
            ViewIm3D(im, 'Extracted PSF')

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

