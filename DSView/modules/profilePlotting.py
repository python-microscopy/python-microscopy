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


from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg

from matplotlib.figure import Figure

#from PYME.Acquire.mytimer import mytimer
import pylab
from scipy import ndimage
import numpy as np

class MyNavigationToolbar(NavigationToolbar2WxAgg):
    """
    Extend the default wx toolbar with your own event handlers
    """
    ON_SAVE_DATA = wx.NewId()
    def __init__(self, canvas, cankill):
        NavigationToolbar2WxAgg.__init__(self, canvas)

        self.AddSimpleTool(self.ON_SAVE_DATA, _load_bitmap('stock_left.xpm'),
                           'Click me', 'Activate custom contol')
        EVT_TOOL(self, self.ON_SAVE_DATA, self.OnSaveData)

    def OnSaveData(self, evt):

        np.savetxt
        evt.Skip()

class profiler:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        self.view = dsviewer.vp.view
        self.do = dsviewer.do
        self.image = dsviewer.image

        PLOT_PROFILE = wx.NewId()
        dsviewer.mProcessing.Append(PLOT_PROFILE, "Plot &Profile\tCtrl-K", "", wx.ITEM_NORMAL)
        
        dsviewer.Bind(wx.EVT_MENU, self.OnPlotProfile, id=PLOT_PROFILE)

        #accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL,  ord('k'), PLOT_PROFILE )])
        #self.dsviewer.SetAcceleratorTable(accel_tbl)


    def OnPlotProfile(self, event=None):
        lx, ly, hx, hy = self.view.GetSliceSelection()

        w = int(np.floor(0.5*self.view.selectionWidth))

        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % d for d in range(self.image.data.shape[3])]

        try:
            voxx = self.mdh.getEntry('voxelsize.x')
        except:
            voxx=1

        Dx = hx - lx
        Dy = hy - ly

        l = np.sqrt((Dx**2 + Dy**2))

        dx = Dx/l
        dy = Dy/l
        
        if Dx == 0 and Dy == 0: #special case - profile is orthogonal to current plane
            d_x = w
            d_y = w
        else:
            d_x = w*dy
            d_y = w*dx

        #print lx, hx, ly, hy

        pylab.figure()
            

        for chanNum in range(self.image.data.shape[3]):

            x_0 = min(lx, hx)
            y_0 = min(ly, hy)

            if(self.do.slice == self.do.SLICE_XY):
                ims = self.image.data[(min(lx, hx) - d_x):(max(lx,hx)+d_x), (min(ly, hy)-d_y):(max(ly,hy)+d_y), self.do.zp, chanNum].squeeze()

            splf = ndimage.spline_filter(ims)

            t = np.arange(np.ceil(l))

            p = np.zeros(len(t))


            x_c = t*dx + lx - x_0
            y_c = t*dy + ly - y_0

            print splf.shape


            for i in range(-w, w+1):
                #print np.vstack([x_c + d_x +i*dy, y_c + d_y + i*dx])
                p += ndimage.map_coordinates(splf, np.vstack([x_c + d_x +i*dy, y_c + d_y + i*dx]), prefilter=False)

            p = p/(2*w + 1)



            pylab.plot(t*voxx, p)

        pylab.legend(names)













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
    dsviewer.profilePlotter = profiler(dsviewer)
    
