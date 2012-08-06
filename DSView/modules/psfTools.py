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
from enthought.traits.api import HasTraits, Float, Int
from enthought.traits.ui.api import View, Item
from enthought.traits.ui.menu import OKButton

class PSFTools(HasTraits):
    wavelength = Float(700)
    NA = Float(1.49)
    pupilSize = Float(0)
    iterations = Int(50)
    
    view = View(Item('wavelength'),
                Item('NA'),
                Item('pupilSize'),
                Item('iterations'), buttons=[OKButton])
    
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        PROC_EXTRACT_PUPIL = wx.NewId()
        PROC_CALC_CRB = wx.NewId()
        #PROC_APPLY_THRESHOLD = wx.NewId()
        #PROC_LABEL = wx.NewId()
        
        dsviewer.mProcessing.Append(PROC_EXTRACT_PUPIL, "Extract &Pupil Function", "", wx.ITEM_NORMAL)
        dsviewer.mProcessing.Append(PROC_CALC_CRB, "Calculate Cramer-Rao Bounds", "", wx.ITEM_NORMAL)
        #dsviewer.mProcessing.Append(PROC_APPLY_THRESHOLD, "Generate &Mask", "", wx.ITEM_NORMAL)
        #dsviewer.mProcessing.Append(PROC_LABEL, "&Label", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, PROC_EXTRACT_PUPIL, self.OnExtractPupil)
        wx.EVT_MENU(dsviewer, PROC_CALC_CRB, self.OnCalcCRB)
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
        
        self.configure_traits(kind='modal')
        
        pupil = fourierHNA.ExtractPupil(np.maximum(self.image.data[:,:,:] - .001, 0), z_, self.image.mdh['voxelsize.x']*1e3, self.wavelength, self.NA, nIters=self.iterations, size=self.pupilSize)
        
        pylab.figure()
        pylab.subplot(121)
        pylab.imshow(np.abs(pupil), interpolation='nearest')
        pylab.subplot(122)
        pylab.imshow(np.angle(pupil)*(np.abs(pupil) > 0), interpolation='nearest')
        
        im = ImageStack([np.abs(pupil), np.angle(pupil)*(np.abs(pupil) > 0)], titleStub = 'Extracted Pupil')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        #im.mdh['Processing.CropROI'] = roi
        mode = 'pupil'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))
        
    def OnCalcCRB(self, event):
        #print 'f'
        from PYME.Analysis import cramerRao
        from PYME.PSFGen import fourierHNA
        #print 'b'
        import numpy as np
        
        d = self.image.data[:,:,:]
        I = d[:,:,d.shape[2]/2].sum()
        
        vs = 1e3*np.array([self.image.mdh['voxelsize.x'], self.image.mdh['voxelsize.y'],self.image.mdh['voxelsize.z']])
        
        #print 'fi'        
        FI = cramerRao.CalcFisherInformZn2(d*(2e3/I), 100, voxelsize=vs)
        #print 'crb'
        crb = cramerRao.CalcCramerReoZ(FI)
        #print 'crbd'
        
        import pylab
        z_ = np.arange(d.shape[2])*self.image.mdh['voxelsize.z']*1.0e3
        z_ = z_ - z_.mean()
        
        print 'p'
        pylab.figure()
        pylab.plot(z_, np.sqrt(crb[:,0]), label='x')
        pylab.plot(z_, np.sqrt(crb[:,1]), label='y')
        pylab.plot(z_, np.sqrt(crb[:,2]), label='z')
        pylab.legend()
        
        pylab.xlabel('Defocus [nm]')
        pylab.ylabel('Std. Dev. [nm]')
        pylab.title('Cramer-Rao bound for 2000 photons')
        
        ps_as = fourierHNA.GenAstigPSF(z_, vs[0], 1.5)        
        crb_as = np.sqrt(cramerRao.CalcCramerReoZ(cramerRao.CalcFisherInformZn2(ps_as*2000/267., 500, voxelsize=vs)))
        pylab.plot(z_, crb_as[:,0], 'b:')
        pylab.plot(z_, crb_as[:,1], 'g:')
        pylab.plot(z_, crb_as[:,2], 'r:')
        

    



def Plug(dsviewer):
    dsviewer.PSFTools = PSFTools(dsviewer)



