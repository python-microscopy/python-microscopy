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

class cropper:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        PROC_CROP = wx.NewId()
        
        
        dsviewer.mProcessing.Append(PROC_CROP, "&Crop", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, PROC_CROP, self.OnCrop)

    def OnCrop(self, event):
        import numpy as np
        #from scipy.ndimage import gaussian_filter
        from PYME.DSView.image import ImageStack
        from PYME.DSView import ViewIm3D

        #dlg = wx.TextEntryDialog(self.dsviewer, 'Blur size [pixels]:', 'Gaussian Blur', '[1,1,1]')

        #if dlg.ShowModal() == wx.ID_OK:
            #sigmas = eval(dlg.GetValue())
            #print sigmas
            #print self.images[0].img.shape

        roi = [[self.do.selection_begin_x, self.do.selection_end_x + 1],[self.do.selection_begin_y, self.do.selection_end_y +1], [0, self.image.data.shape[2]]]

        filt_ims = [np.atleast_3d(self.image.data[roi[0][0]:roi[0][1],roi[1][0]:roi[1][1],:,chanNum].squeeze()) for chanNum in range(self.image.data.shape[3])]

        im = ImageStack(filt_ims, titleStub = 'Filtered Image')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        im.mdh['Processing.CropROI'] = roi

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



def Plug(dsviewer):
    dsviewer.cropper = cropper(dsviewer)



