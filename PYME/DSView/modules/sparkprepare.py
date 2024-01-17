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
#from PYME.IO.image import ImageStack

class filterer:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        PROC_GAUSSIAN_FILTER = wx.NewIdRef()
        
        
        dsviewer.mProcessing.Append(PROC_GAUSSIAN_FILTER, "&Spark", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, PROC_GAUSSIAN_FILTER, self.OnGaussianFilter)

    def OnGaussianFilter(self, event):
        import numpy as np
        from scipy.ndimage import gaussian_filter1d, convolve1d
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D

        #dlg = wx.TextEntryDialog(self.dsviewer, 'Blur size [pixels]:', 'Gaussian Blur', '[1,1,1]')

        if True: #dlg.ShowModal() == wx.ID_OK:
            #sigmas = eval(dlg.GetValue())
            #print sigmas
            #print self.images[0].img.shape
            #filt_ims = [np.atleast_3d(gaussian_filter(self.image.data[:,:,:,chanNum].squeeze(), sigmas)) for chanNum in range(self.image.data.shape[3])]
            ims = self.image.data[:,:,0,0].astype('f')
            
            ims = ims - np.median(ims, 0)
            
            ims = gaussian_filter1d(ims, 2, 1)
            ims = convolve1d(ims, np.ones(10), 0)
            
            ims = np.rollaxis(ims, 0, 3)
            
            #ims = (ims - ims.min()).astype('uint16')
            
            im = ImageStack(ims, titleStub = 'Filtered Image')
            im.mdh.copyEntriesFrom(self.image.mdh)
            im.mdh['Parent'] = self.image.filename
            #im.mdh['Processing.'] = sigmas

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
    dsviewer.filtering = filterer(dsviewer)



