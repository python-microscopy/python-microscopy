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

        

    



def Plug(dsviewer):
    dsviewer.PSFTools = PSFTools(dsviewer)



