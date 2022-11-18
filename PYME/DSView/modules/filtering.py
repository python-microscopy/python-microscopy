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

from ._base import Plugin

class Filterer(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self,dsviewer)
                
        dsviewer.AddMenuItem('Processing', '&Gaussian Filter', self.OnGaussianFilter)        
        dsviewer.AddMenuItem('Processing', 'Generate &Mask', self.OnApplyThreshold)
        dsviewer.AddMenuItem('Processing', '&Label', self.OnLabelSizeThreshold)        
        dsviewer.AddMenuItem('Processing', 'Set Labels', self.OnSetLabels)        
        dsviewer.AddMenuItem('Processing', '&Watershed', self.OnLabelWatershed)        
        dsviewer.AddMenuItem('Processing', 'Mean Projection', self.OnMeanProject)
        dsviewer.AddMenuItem('Processing', 'Max Projection', self.OnMaxProject)
        dsviewer.AddMenuItem('Processing', 'Average Frames by Step', self.OnAverageFramesByStep)
        dsviewer.AddMenuItem('Processing', 'Resample Z Stack', self.OnResampleZStack)

    def OnGaussianFilter(self, event):
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D

        filter_size = '[1,1,1]'
        if self.image.data.shape[2] == 1:
            filter_size = '[1,1]'

        dlg = wx.TextEntryDialog(self.dsviewer, 'Blur size [pixels]:', 'Gaussian Blur', filter_size)

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

            

        dlg.Destroy()
        
    def OnMeanProject(self, event):
        self.Project('mean')
        
    def OnMaxProject(self, event):
        self.Project('max')
    
    def Project(self, projType):
        import numpy as np
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        import os

        if projType == 'mean':        
            filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].mean(2)) for chanNum in range(self.image.data.shape[3])]
        elif projType == 'max':
            filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].max(2)) for chanNum in range(self.image.data.shape[3])]

        fns = os.path.split(self.image.filename)[1]        
        
        im = ImageStack(filt_ims, titleStub = '%s - %s' %(fns, projType))
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        im.mdh['Processing.Projection'] = projType

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas)

        #set scaling to (0,1)
        for i in range(im.data.shape[3]):
            dv.do.Gains[i] = 1.0


    def OnApplyThreshold(self, event):
        import numpy as np
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D

       
        filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].squeeze() > self.dsviewer.do.thresholds[chanNum]) for chanNum in range(self.image.data.shape[3])]
        

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        elif self.dsviewer.mode == 'graph':
            mode = 'graph'
            filt_ims = [fi.squeeze() for fi in filt_ims]
        else:
            mode = 'lite'
            
        im = ImageStack(sum(filt_ims) > 0.5, titleStub = 'Thresholded Image')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))

        #set scaling to (0,1)
        for i in range(im.data.shape[3]):
            dv.do.Gains[i] = 1.0

            

    def OnLabel(self, event):
        import numpy as np
        from scipy import ndimage
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D

        filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].squeeze() > self.dsviewer.do.thresholds[chanNum]) for chanNum in range(self.image.data.shape[3])]

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

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))

        #set scaling to (0,1)
        for i in range(im.data.shape[3]):
            dv.do.Gains[i] = 1.0

            
        
    def OnLabelSizeThreshold(self, event):
        import numpy as np
        from scipy import ndimage
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D

        #dlg = wx.TextEntryDialog(self.dsviewer, 'Blur size [pixels]:', 'Gaussian Blur', '[1,1,1]')

        #if dlg.ShowModal() == wx.ID_OK:
            #sigmas = eval(dlg.GetValue())
            #print sigmas
            #print self.images[0].img.shape

        #roi = [[self.do.selection.start.x, self.do.selection.finish.x + 1],[self.do.selection.start.y, self.do.selection.finish.y +1], [0, self.image.data.shape[2]]]

        #filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].squeeze() > self.dsviewer.do.Offs[chanNum]) for chanNum in range(self.image.data.shape[3])]

        dlg = wx.TextEntryDialog(self.dsviewer, 'Minimum region size [pixels]:', 'Labelling', '1')

        if dlg.ShowModal() == wx.ID_OK:
            rSize = int(dlg.GetValue())
            
            
            self.image.labelThresholds = self.do.thresholds
        
            #filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].squeeze() > (self.dsviewer.do.Offs[chanNum] + 0.5/self.dsviewer.do.Gains[chanNum])) for chanNum in range(self.image.data.shape[3])]
            filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].squeeze() > self.image.labelThresholds[chanNum]) for chanNum in range(self.image.data.shape[3])]
            
            #self.image.labelThresholds = [(self.dsviewer.do.Offs[chanNum] + 0.5/self.dsviewer.do.Gains[chanNum]) for chanNum in range(self.image.data.shape[3])]
    
            #print sum(filt_ims).shape
            mask = sum(filt_ims) > 0.5
            labs, nlabs = ndimage.label(mask)
            
            if rSize > 1:
                m2 = 0*mask
                objs = ndimage.find_objects(labs)
                for i, o in enumerate(objs):
                    r = labs[o] == i+1
                    #print r.shape
                    if r.sum() > rSize:
                        m2[o] = r
                        
                labs, nlabs = ndimage.label(m2)
                
            #store a copy in the image for measurements etc ...
            self.image.labels = labs
            
            im = ImageStack(labs, titleStub = 'Labelled Image')
            im.mdh.copyEntriesFrom(self.image.mdh)
            im.mdh['Parent'] = self.image.filename
            
            im.mdh['Labelling.MinSize'] = rSize
            im.mdh['Labelling.Thresholds'] = self.image.labelThresholds
            #im.mdh['Processing.CropROI'] = roi
    
            if self.dsviewer.mode == 'visGUI':
                mode = 'visGUI'
            else:
                mode = 'lite'
    
            dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))
    
            #set scaling to (0,1)
            for i in range(im.data.shape[3]):
                dv.do.Gains[i] = 1.0
                
    def OnSetLabels(self, event):
        from PYME.IO import image
        names = image.openImages.keys()
        
        dlg = wx.SingleChoiceDialog(self.dsviewer, 'Select an image', 'Set Labels', names)
        
        if dlg.ShowModal() == wx.ID_OK:
            #store a copy in the image for measurements etc ...

            im = image.openImages[names[dlg.GetSelection()]]
            
            self.image.labels = im.data[:,:,:].astype('uint16')
            
                
    def OnLabelWatershed(self, event):
        import numpy as np
        from PYME.contrib.cpmath import watershed
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        
        nChans = self.image.data.shape[3]
    
        filt_ims = [np.atleast_3d(self.image.data[:,:,:,chanNum].squeeze()) for chanNum in range(nChans)]
        
        img = (-sum([im/im.max() for im in filt_ims])*(2**15)/nChans).astype('int16')
        
        mask = (sum([filt_ims[chanNum] > self.do.thresholds[chanNum] for chanNum in range(nChans)]) > .5).astype('int16')
        
        #self.image.labelThresholds = [(self.dsviewer.do.Offs[chanNum] + 0.5/self.dsviewer.do.Gains[chanNum]) for chanNum in range(self.image.data.shape[3])]

        #print sum(filt_ims).shape
        
        labs = watershed.fast_watershed(img, self.image.labels.astype('int16'), mask=mask)
            
        #store a copy in the image for measurements etc ...
        self.image.labels = labs
        
        im = ImageStack(labs, titleStub = 'Labelled Image')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        
        im.mdh['Labelling.WatershedThresholds'] = self.do.thresholds
        
        #im.mdh['Labelling.MinSize'] = rSize
        #im.mdh['Labelling.Thresholds'] = self.image.labelThresholds
        #im.mdh['Processing.CropROI'] = roi

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))

        #set scaling to (0,1)
        for i in range(im.data.shape[3]):
            dv.do.Gains[i] = 1.0

    def OnAverageFramesByStep(self, event):
        """
        Averages frames acquired at the same z-position, as determined by the associated events, or (fall-back)
        metadata. See PYME.recipes.processing.AverageFramesByStep
        Parameters
        ----------
        event : wx.Event

        Returns
        -------

        """
        from PYME.DSView import ViewIm3D
        from PYME.recipes.processing import AverageFramesByZStep

        averaged = AverageFramesByZStep().apply_simple(input_image=self.image)

        ViewIm3D(averaged, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))

    def OnResampleZStack(self, event):
        from PYME.DSView import ViewIm3D
        from PYME.recipes.processing import ResampleZ

        dialog = wx.TextEntryDialog(None, 'Z Spacing [um]:', 'Enter Desired Spacing', str(0.05))
        if dialog.ShowModal() == wx.ID_OK:
            regular = ResampleZ(z_sampling=float(dialog.GetValue())).apply_simple(input=self.image)

            ViewIm3D(regular, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))


def Plug(dsviewer):
    return Filterer(dsviewer)



