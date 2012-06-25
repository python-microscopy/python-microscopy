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

class ZernikeView(wx.Panel):
    def __init__(self, dsviewer):
        from PYME.misc import zernike
        import numpy as np
        
        self.dsviewer = dsviewer
        mag = dsviewer.image.data[:,:,0]
        phase = dsviewer.image.data[:,:,1]
        
        xm = np.where(mag.max(1) > 0)[0]
        ym = np.where(mag.max(0) > 0)[0]
        
        print xm, ym, mag.shape

        mag = mag[xm[0]:(xm[-1]+1), ym[0]:(ym[-1]+1)]        
        phase = phase[xm[0]:(xm[-1]+1), ym[0]:(ym[-1]+1)]
        
        #im = mag*np.exp(1j*phase)
        
        coeffs, res, im = zernike.calcCoeffs(phase, 25, mag)
        
        s = ''
        
        for i, c, r, in zip(xrange(25), coeffs, res):
            s += '%d\t%s%3.3f\tresidual=%3.2f\n' % (i, zernike.NameByNumber[i].ljust(30), c, r)
        
        wx.Panel.__init__(self, dsviewer)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        vsizer.Add(wx.StaticText(self, -1, s), 1, wx.EXPAND)
        
        self.SetSizerAndFit(vsizer)
        
        

class PupilTools:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        PROC_PUPIL_TO_PSF = wx.NewId()
        #PROC_APPLY_THRESHOLD = wx.NewId()
        #PROC_LABEL = wx.NewId()
        
        dsviewer.mProcessing.Append(PROC_PUPIL_TO_PSF, "Generate PSF from pupil", "", wx.ITEM_NORMAL)
        #dsviewer.mProcessing.Append(PROC_APPLY_THRESHOLD, "Generate &Mask", "", wx.ITEM_NORMAL)
        #dsviewer.mProcessing.Append(PROC_LABEL, "&Label", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, PROC_PUPIL_TO_PSF, self.OnPSFFromPupil)
        #wx.EVT_MENU(dsviewer, PROC_APPLY_THRESHOLD, self.OnApplyThreshold)
        #wx.EVT_MENU(dsviewer, PROC_LABEL, self.OnLabel)

    def OnPSFFromPupil(self, event):
        import numpy as np
        #import pylab
        from PYME.PSFGen import fourierHNA
        
        from PYME.DSView.image import ImageStack
        from PYME.DSView import ViewIm3D

        z_ = np.arange(61)*self.image.mdh['voxelsize.z']*1.e3
        z_ -= z_.mean()        
        
        ps = fourierHNA.PsfFromPupilVect(self.image.data[:,:,0]*np.exp(1j*self.image.data[:,:,1]), z_, self.image.mdh['voxelsize.x']*1e3, 700)
        
        im = ImageStack(ps, titleStub = 'Generated PSF')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        #im.mdh['Processing.CropROI'] = roi
        mode = 'psf'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas)

        

    



def Plug(dsviewer):
    dsviewer.PupilTools = PupilTools(dsviewer)
    dsviewer.zern = ZernikeView(dsviewer)
    dsviewer.AddPage(dsviewer.zern, False, 'Zernike Moments')



