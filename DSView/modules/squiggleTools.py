#!/usr/bin/python
##################
# profilePlotting.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx

#from PYME.Acquire.mytimer import mytimer
import pylab
from scipy import ndimage
import numpy as np

from PYME.DSView.dsviewer_npy_nb import ViewIm3D, ImageStack

class squiggle:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        #self.view = dsviewer.view
        self.do = dsviewer.do
        self.image = dsviewer.image

        PLOT_PROFILE = wx.NewId()
        dsviewer.mProcessing.Append(PLOT_PROFILE, "Plot wavy profile", "", wx.ITEM_NORMAL)
        
        dsviewer.Bind(wx.EVT_MENU, self.OnPlotProfile, id=PLOT_PROFILE)

        #accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL,  ord('k'), PLOT_PROFILE )])
        #self.dsviewer.SetAcceleratorTable(accel_tbl)


    def OnPlotProfile(self, event=None):
        #lx, ly, hx, hy = self.do.GetSliceSelection()
        pts = np.array(self.do.selection_trace)

        w = int(np.floor(0.5*self.do.selectionWidth))

        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % d for d in range(self.image.data.shape[3])]

        try:
            voxx = self.image.mdh.getEntry('voxelsize.x')
        except:
            voxx=1

        #d_x = w
        #d_y = w
        

        #pylab.figure()
        plots = []
        t = np.arange(np.ceil(len(pts)))

        for chanNum in range(self.image.data.shape[3]):

            #x_0 = min(lx, hx)
            #y_0 = min(ly, hy)

            #d__x = abs(d_x) + 1
            #d__y = abs(d_y) + 1

            #print dx, dy, d__x, d__y, w

            #if(self.do.slice == self.do.SLICE_XY):
            #    ims = self.image.data[(min(lx, hx) - d__x):(max(lx,hx)+d__x+1), (min(ly, hy)-d__y):(max(ly,hy)+d__y+1), self.do.zp, chanNum].squeeze()

            #splf = ndimage.spline_filter(ims) 

            #p = np.zeros(len(t))


            #x_c = t*dx + lx - x_0
            #y_c = t*dy + ly - y_0

            #print splf.shape


            #for i in range(-w, w+1):
                #print np.vstack([x_c + d__x +i*dy, y_c + d__y + i*dx])
            #    p += ndimage.map_coordinates(splf, pts.T, prefilter=False)

            #p = p/(2*w + 1)
            
            p = ndimage.map_coordinates(self.image.data[:,:, self.do.zp, chanNum].squeeze(), pts.T)



            plots.append(p.reshape(-1, 1,1))

        #pylab.legend(names)

        im = ImageStack(plots, titleStub='New Profile')
        im.xvals = t*voxx

        if not voxx == 1:
            im.xlabel = 'Distance [um]'
        else:
            im.xlabel = 'Distance [pixels]'

        im.ylabel = 'Intensity'
        im.defaultExt = '.txt'

        im.mdh['voxelsize.x'] = voxx
        im.mdh['ChannelNames'] = names
        im.mdh['Profile.XValues'] = im.xvals
        im.mdh['Profile.XLabel'] = im.xlabel
        im.mdh['Profile.YLabel'] = im.ylabel
        #im.mdh['Profile.StartX'] = lx
        #im.mdh['Profile.StartY'] = ly
        #im.mdh['Profile.EndX'] = hx
        #im.mdh['Profile.EndY'] = hy
        #im.mdh['Profile.Width'] = 2*w + 1

        im.mdh['OriginalImage'] = self.image.filename

        ViewIm3D(im, mode='graph')













        #dsviewer.paneHooks.append(self.GenProfilePanel)

#    def GenProfilePanel(self, _pnl):
#        item = afp.foldingPane(_pnl, -1, caption="Intensity Profile", pinned = True)
##        item = self._pnl.AddFoldPanel("Intensity Profile", collapsed=False,
##                                      foldIcons=self.Images)
#
#        bPlotProfile = wx.Button(item, -1, 'Plot')
#
#        bPlotProfile.Bind(wx.EVT_BUTTON, self.OnPlotProfile)
#        #self._pnl.AddFoldPanelWindow(item, bPlotProfile, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
#        item.AddNewElement(bPlotProfile)
#        _pnl.AddPane(item)


    def OnZPlotProfile(self, event):
        x,p,d, pi = self.vp.GetProfile(50, background=[7,7])

        pylab.figure(1)
        pylab.clf()
        pylab.step(x,p)
        pylab.step(x, 10*d - 30)
        pylab.ylim(-35,pylab.ylim()[1])

        pylab.xlim(x.min(), x.max())

        pylab.xlabel('Time [%3.2f ms frames]' % (1e3*self.mdh.getEntry('Camera.CycleTime')))
        pylab.ylabel('Intensity [counts]')

        fr = self.fitResults[pi]

        if not len(fr) == 0:
            pylab.figure(2)
            pylab.clf()

            pylab.subplot(211)
            pylab.errorbar(fr['tIndex'], fr['fitResults']['x0'] - self.vp.do.xp*1e3*self.mdh.getEntry('voxelsize.x'), fr['fitError']['x0'], fmt='xb')
            pylab.xlim(x.min(), x.max())
            pylab.xlabel('Time [%3.2f ms frames]' % (1e3*self.mdh.getEntry('Camera.CycleTime')))
            pylab.ylabel('x offset [nm]')

            pylab.subplot(212)
            pylab.errorbar(fr['tIndex'], fr['fitResults']['y0'] - self.vp.do.yp*1e3*self.mdh.getEntry('voxelsize.y'), fr['fitError']['y0'], fmt='xg')
            pylab.xlim(x.min(), x.max())
            pylab.xlabel('Time [%3.2f ms frames]' % (1e3*self.mdh.getEntry('Camera.CycleTime')))
            pylab.ylabel('y offset [nm]')

            pylab.figure(3)
            pylab.clf()

            pylab.errorbar(fr['fitResults']['x0'] - self.vp.do.xp*1e3*self.mdh.getEntry('voxelsize.x'),fr['fitResults']['y0'] - self.vp.do.yp*1e3*self.mdh.getEntry('voxelsize.y'), fr['fitError']['x0'], fr['fitError']['y0'], fmt='xb')
            #pylab.xlim(x.min(), x.max())
            pylab.xlabel('x offset [nm]')
            pylab.ylabel('y offset [nm]')


def Plug(dsviewer):
    dsviewer.squiggle = squiggle(dsviewer)
    
