#!/usr/bin/python
##################
# coloc.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
#import numpy
import wx
#import pylab
#from PYME.DSView.image import ImageStack

class filterer:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        PROC_GAUSSIAN_FILTER = wx.NewId()
        PROC_APPLY_THRESHOLD = wx.NewId()
        PROC_LABEL = wx.NewId()
        
        dsviewer.mProcessing.Append(PROC_GAUSSIAN_FILTER, "&Gaussian Filter", "", wx.ITEM_NORMAL)
        dsviewer.mProcessing.Append(PROC_APPLY_THRESHOLD, "Generate &Mask", "", wx.ITEM_NORMAL)
        dsviewer.mProcessing.Append(PROC_LABEL, "&Label", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, PROC_GAUSSIAN_FILTER, self.OnGaussianFilter)
        wx.EVT_MENU(dsviewer, PROC_APPLY_THRESHOLD, self.OnApplyThreshold)
        wx.EVT_MENU(dsviewer, PROC_LABEL, self.OnLabel)

    def OnGaussianFilter(self, event):
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from PYME.DSView.image import ImageStack
        from PYME.DSView import ViewIm3D

        dlg = wx.TextEntryDialog(self.dsviewer, 'Blur size [pixels]:', 'Gaussian Blur', '[1,1,1]')

        if dlg.ShowModal() == wx.ID_OK:
            sigmas = eval(dlg.GetValue())
            #print sigmas
            #print self.images[0].img.shape
            filt_ims = [np.atleast_3d(gaussian_filter(self.image.data[:,:,:,chanNum].squeeze(), sigmas)) for chanNum in range(self.image.data.shape[3])]
            
            im = ImageStack(filt_ims, titleStub = 'Filtered Image')
            im.mdh.copyEntriesFrom(self.image.mdh)
            im.mdh['Parent'] = self.image.filename
            im.mdh['Processing.GaussianFilter'] = sigmas

            if self.dsviewer.mode == 'visGUI':
                mode = 'visGUI'
            else:
                mode = 'lite'

            dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas)

            #set scaling to (0,1)
            for i in range(im.data.shape[3]):
                dv.do.Gains[i] = 1.0

            #imfc = MultiChannelImageViewFrame(self.parent, self.parent.glCanvas, filt_ims, self.image.names, title='Filtered Image - %3.1fnm bins' % self.image.pixelSize)

            #self.parent.generatedImages.append(imfc)
            #imfc.Show()

        dlg.Destroy()

    def OnApplyThreshold(self, event):
        import numpy as np
        #from scipy.ndimage import gaussian_filter
        from PYME.DSView.image import ImageStack
        from PYME.DSView import ViewIm3D

        #dlg = wx.TextEntryDialog(self.dsviewer, 'Blur size [pixels]:', 'Gaussian Blur', '[1,1,1]')

        #if dlg.ShowModal() == wx.ID_OK:
            #sigmas = eval(dlg.GetValue())
            #print sigmas
            #print self.images[0].img.shape

        #roi = [[self.do.selection_begin_x, self.do.selection_end_x + 1],[self.do.selection_begin_y, self.do.selection_end_y +1], [0, self.image.data.shape[2]]]

        filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].squeeze() > (self.dsviewer.do.Offs[chanNum] + 0.5/self.dsviewer.do.Gains[chanNum])) for chanNum in range(self.image.data.shape[3])]
        
        #print sum(filt_ims).shape


        im = ImageStack(sum(filt_ims) > 0.5, titleStub = 'Thresholded Image')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        #im.mdh['Processing.CropROI'] = roi

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas)

        #set scaling to (0,1)
        for i in range(im.data.shape[3]):
            dv.do.Gains[i] = 1.0

            #imfc = MultiChannelImageViewFrame(self.parent, self.parent.glCanvas, filt_ims, self.image.names, title='Filtered Image - %3.1fnm bins' % self.image.pixelSize)

            #self.parent.generatedImages.append(imfc)
            #imfc.Show()

        #dlg.Destroy()

    def OnLabel(self, event):
        import numpy as np
        from scipy import ndimage
        from PYME.DSView.image import ImageStack
        from PYME.DSView import ViewIm3D

        #dlg = wx.TextEntryDialog(self.dsviewer, 'Blur size [pixels]:', 'Gaussian Blur', '[1,1,1]')

        #if dlg.ShowModal() == wx.ID_OK:
            #sigmas = eval(dlg.GetValue())
            #print sigmas
            #print self.images[0].img.shape

        #roi = [[self.do.selection_begin_x, self.do.selection_end_x + 1],[self.do.selection_begin_y, self.do.selection_end_y +1], [0, self.image.data.shape[2]]]

        #filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].squeeze() > self.dsviewer.do.Offs[chanNum]) for chanNum in range(self.image.data.shape[3])]
        filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].squeeze() > (self.dsviewer.do.Offs[chanNum] + 0.5/self.dsviewer.do.Gains[chanNum])) for chanNum in range(self.image.data.shape[3])]

        #print sum(filt_ims).shape
        mask = sum(filt_ims) > 0.5
        labs, nlabs = ndimage.label(mask)

        im = ImageStack(labs, titleStub = 'Thresholded Image')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        #im.mdh['Processing.CropROI'] = roi

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas)

        #set scaling to (0,1)
        for i in range(im.data.shape[3]):
            dv.do.Gains[i] = 1.0

            #imfc = MultiChannelImageViewFrame(self.parent, self.parent.glCanvas, filt_ims, self.image.names, title='Filtered Image - %3.1fnm bins' % self.image.pixelSize)

            #self.parent.generatedImages.append(imfc)
            #imfc.Show()

        #dlg.Destroy()



def Plug(dsviewer):
    dsviewer.filtering = filterer(dsviewer)



