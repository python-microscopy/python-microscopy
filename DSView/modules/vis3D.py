#!/usr/bin/python
##################
# vis3D.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy
import wx
import pylab

class visualiser:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.vp.do

        self.image = dsviewer.image
        self.tq = None

        td_menu = wx.Menu()

        VIEW_ISOSURFACE = wx.NewId()
        VIEW_VOLUME = wx.NewId()
        #dsviewer.mExtras.AppendSeparator()
        td_menu.Append(VIEW_ISOSURFACE, "3D Isosurface", "", wx.ITEM_NORMAL)
        td_menu.Append(VIEW_VOLUME, "3D Volume", "", wx.ITEM_NORMAL)

        dsviewer.menubar.Insert(dsviewer.menubar.GetMenuCount()-1, td_menu, '3D')
        #mDeconvolution.AppendSeparator()
        #dsviewer.save_menu.Append(DECONV_SAVE, "Deconvolution", "", wx.ITEM_NORMAL)
        #self.menubar.Append(mDeconvolution, "Deconvolution")

        wx.EVT_MENU(dsviewer, VIEW_ISOSURFACE, self.On3DIsosurf)
        wx.EVT_MENU(dsviewer, VIEW_VOLUME, self.On3DVolume)
        #wx.EVT_MENU(dsviewer, DECONV_SAVE, self.saveDeconvolution)

        #dsviewer.updateHooks.append(self.update)


    def checkTQ(self):
        import Pyro.core
        if self.tq == None:
            if 'PYME_TASKQUEUENAME' in os.environ.keys():
                taskQueueName = os.environ['PYME_TASKQUEUENAME']
            else:
                taskQueueName = 'taskQueue'
            self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)

    def On3DIsosurf(self, event):
        from enthought.mayavi import mlab

        f = mlab.figure()

        for i in range(self.image.data.shape[3]):
            c = mlab.contour3d(self.image.data[:,:,:,i].astype('f'), contours=[self.do.Offs[i] + .5/self.do.Gains[i]], color = pylab.cm.gist_rainbow(float(i)/self.image.data.shape[3])[:3])
            c.mlab_source.dataset.spacing = (self.image.mdh.getEntry('voxelsize.x') ,self.image.mdh.getEntry('voxelsize.y'), self.image.mdh.getEntry('voxelsize.z'))

    def On3DVolume(self, event):
        from enthought.mayavi import mlab

        f = mlab.figure()

        for i in range(self.image.data.shape[3]):
            #c = mlab.contour3d(im.img, contours=[pylab.mean(ivp.clim)], color = pylab.cm.gist_rainbow(float(i)/len(self.images))[:3])
            v = mlab.pipeline.volume(mlab.pipeline.scalar_field(numpy.minimum(255*(self.image.data[:,:,:,i] -self.do.Offs[i])*self.do.Gains[i], 254).astype('uint8')))
            v.volume.scale = (self.image.mdh.getEntry('voxelsize.x') ,self.image.mdh.getEntry('voxelsize.y'), self.image.mdh.getEntry('voxelsize.z'))



def Plug(dsviewer):
    dsviewer.vis3D = visualiser(dsviewer)



