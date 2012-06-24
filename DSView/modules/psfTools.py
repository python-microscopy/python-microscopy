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

class PSFTools:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        PROC_EXTRACT_PUPIL = wx.NewId()
        #PROC_APPLY_THRESHOLD = wx.NewId()
        #PROC_LABEL = wx.NewId()
        
        dsviewer.mProcessing.Append(PROC_EXTRACT_PUPIL, "Extract &Pupil Function", "", wx.ITEM_NORMAL)
        #dsviewer.mProcessing.Append(PROC_APPLY_THRESHOLD, "Generate &Mask", "", wx.ITEM_NORMAL)
        #dsviewer.mProcessing.Append(PROC_LABEL, "&Label", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, PROC_EXTRACT_PUPIL, self.OnExtractPupil)
        #wx.EVT_MENU(dsviewer, PROC_APPLY_THRESHOLD, self.OnApplyThreshold)
        #wx.EVT_MENU(dsviewer, PROC_LABEL, self.OnLabel)

    def OnExtractPupil(self, event):
        import numpy as np
        import pylab
        from PYME.PSFGen import fourierHNA
        
        from PYME.DSView.image import ImageStack
        from PYME.DSView import ViewIm3D

        z_ = np.arange(self.image.data.shape[2])*self.image.mdh['voxelsize.z']*1.e3
        z_ -= z_.mean()        
        
        pupil = fourierHNA.ExtractPupil(self.image.data[:,:,:], z_, self.image.mdh['voxelsize.x']*1e3, 680, 1.49)
        
        pylab.figure()
        pylab.subplot(121)
        pylab.imshow(np.abs(pupil), interpolation='nearest')
        pylab.subplot(122)
        pylab.imshow(np.angle(pupil)*(np.abs(pupil) > 0), interpolation='nearest')
        
        im = ImageStack([np.abs(pupil), np.angle(pupil)*(np.abs(pupil) > 0)], titleStub = 'Extracted Pupil')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        #im.mdh['Processing.CropROI'] = roi
        mode = 'lite'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas)

        

    



def Plug(dsviewer):
    dsviewer.PSFTools = PSFTools(dsviewer)



