#!/usr/bin/python
##################
# profilePlotting.py
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

import wx

#from PYME.Acquire.mytimer import mytimer
import pylab
from scipy import ndimage
import numpy as np

from PYME.DSView.dsviewer_npy_nb import ViewIm3D, ImageStack

class profiler:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        #self.view = dsviewer.view
        self.do = dsviewer.do
        self.image = dsviewer.image
        
        dsviewer.AddMenuItem('Processing', "Plot &Profile\tCtrl-K", self.OnPlotProfile)
        dsviewer.AddMenuItem('Processing', "Plot Axial Profile\tCtrl-Shift-K", self.OnPlotAxialProfile)

        #accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL,  ord('k'), PLOT_PROFILE )])
        #self.dsviewer.SetAcceleratorTable(accel_tbl)


    def OnPlotProfile(self, event=None):
        lx, ly, hx, hy = self.do.GetSliceSelection()

        w = int(np.floor(0.5*self.do.selectionWidth))

        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % d for d in range(self.image.data.shape[3])]

        try:
            voxx = self.image.mdh.getEntry('voxelsize.x')
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
            d_x = w*abs(dy)
            d_y = w*abs(dx)

        #print lx, hx, ly, hy

        #pylab.figure()
        plots = []
        t = np.arange(np.ceil(l))

        for chanNum in range(self.image.data.shape[3]):

            x_0 = min(lx, hx)
            y_0 = min(ly, hy)

            d__x = abs(d_x) + 1
            d__y = abs(d_y) + 1

            print((dx, dy, d__x, d__y, w))

            if(self.do.slice == self.do.SLICE_XY):
                ims = self.image.data[(min(lx, hx) - d__x):(max(lx,hx)+d__x+1), (min(ly, hy)-d__y):(max(ly,hy)+d__y+1), self.do.zp, chanNum].squeeze()

            splf = ndimage.spline_filter(ims) 

            p = np.zeros(len(t))


            x_c = t*dx + lx - x_0
            y_c = t*dy + ly - y_0

            print((splf.shape))


            for i in range(-w, w+1):
                #print np.vstack([x_c + d__x +i*dy, y_c + d__y + i*dx])
                p += ndimage.map_coordinates(splf, np.vstack([x_c + d__x +i*dy, y_c + d__y - i*dx]), prefilter=False)

            p = p/(2*w + 1)



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
        im.mdh['Profile.StartX'] = lx
        im.mdh['Profile.StartY'] = ly
        im.mdh['Profile.EndX'] = hx
        im.mdh['Profile.EndY'] = hy
        im.mdh['Profile.Width'] = 2*w + 1

        im.mdh['OriginalImage'] = self.image.filename

        ViewIm3D(im, mode='graph', parent=wx.GetTopLevelParent(self.dsviewer))




    def OnPlotAxialProfile(self, event=None):
        lx, ly, hx, hy = self.do.GetSliceSelection()

        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % d for d in range(self.image.data.shape[3])]

       
        plots = []
        t = np.arange(self.image.data.shape[2])
        
        try:
            stack = (self.image.mdh['AcquisitionType'] == 'Stack')
        except:
            stack = False                
        
        if stack:
            dt = self.image.mdh['voxelsize.z']
        else:
            try:        
                dt = self.image.mdh['Camera.CycleTime']
            except:
                dt = 1
            
        for chanNum in range(self.image.data.shape[3]):
            plots.append(np.zeros((len(t), 1, 1)))

        dlg = wx.ProgressDialog('Extracting Axial Profile', 'Progress', len(t) - 1)        
        for i in t:      
            for chanNum in range(self.image.data.shape[3]):
                plots[chanNum][i] = self.image.data[lx:hx, ly:hy, i, chanNum].mean()
                if (i % 10) == 0:
                    dlg.Update(i, '%d of %d frames' % (i, t.size))
                    
        dlg.Destroy()
                

        #pylab.legend(names)

        im = ImageStack(plots, titleStub='New Profile')
        im.xvals = t*dt

        if stack:
            im.xlabel = 'Position [um]'
        elif not dt == 1:
            im.xlabel = 'Time [s]'
        else:
            im.xlabel = 'Time [frames]'

        im.ylabel = 'Intensity'
        im.defaultExt = '.txt'

        im.mdh['voxelsize.x'] = dt
        im.mdh['ChannelNames'] = names
        im.mdh['Profile.XValues'] = im.xvals
        im.mdh['Profile.XLabel'] = im.xlabel
        im.mdh['Profile.YLabel'] = im.ylabel
        im.mdh['Profile.X1'] = lx
        im.mdh['Profile.Y1'] = ly
        im.mdh['Profile.X2'] = hx
        im.mdh['Profile.Y2'] = hy

        im.mdh['OriginalImage'] = self.image.filename
        
        im.parent = self.image

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
    dsviewer.profilePlotter = profiler(dsviewer)
    
