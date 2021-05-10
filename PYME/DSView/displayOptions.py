#!/usr/bin/python

##################
# displayOptions.py
#
# Copyright David Baddeley, 2009
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

from matplotlib import cm
import numpy as np
#import tables

from PYME.contrib import dispatch

from PYME.IO import dataWrap

import logging
logger = logging.getLogger(__name__)

try:
    # location in Python 2.7 and 3.1
    from weakref import WeakSet
except ImportError:
    # separately installed
    from weakrefset import WeakSet

def fast_grey(data):
    return data[:,:,None]*np.ones((1,1,4))

fast_grey.name = 'fastGrey'

def labeled(data):
    return (data > 0)[:,:,None]*cm.gist_rainbow(data % 1)

labeled.name = 'labeled'

class MyWeakSet(WeakSet):
    def append(self, item):
        self.add(item)
     

class DisplayOpts(object):
    UPRIGHT, ROT90 = range(2)
    SLICE_XY, SLICE_XZ, SLICE_YZ = range(3)

    ACTION_POSITION, ACTION_SELECTION = range(2)
    SELECTION_RECTANGLE, SELECTION_LINE, SELECTION_SQUIGGLE = range(3)

    def __init__(self, datasource, xp=0, yp=0, zp=0, aspect=1):
        self.WantChangeNotification = []# MyWeakSet() #[]
        
        self.Chans = []
        self.Gains = []
        self.Offs = []
        self.cmaps = []
        self.names = []
        self.show = []
        
        self._xp=0
        self._yp=0

        self._zp=0
        self._tp=0
        
        self.maximumProjection=False
        self.colourMax = False
        self.cmax_offset = 0.0
        self.cmax_scale = 1.0
        
        self._complexMode = 'coloured'
        
        self.inOnChange = False
        self.syncedWith = []

        self.SetDataStack(datasource)
        self.SetAspect(aspect)
        self.ResetSelection()

        self.orientation = self.UPRIGHT
        self.slice = self.SLICE_XY
        self.scale = 0

        self.leftButtonAction = self.ACTION_POSITION
        self.selectionMode = self.SELECTION_RECTANGLE

        self.selectionWidth = 1

        self.showSelection=False

        #signals (currently just for selection end
        # this lets people know that we've made a selection. Used to allow modification of the selection by some filter
        # e.g. using active contours in the annotation module
        self.on_selection_end = dispatch.Signal()
        
        self.overlays = []
        


    @property
    def zp(self):
        return self._zp

    @zp.setter
    def zp(self, value):
        if (value < 0) or (value >= self.ds.shape[2]):
            raise IndexError('Z position out of bounds')
        
        self._zp = value
        #print 'z changed'
        self.OnChange()

    @property
    def xp(self):
        return self._xp

    @xp.setter
    def xp(self, value):
        if (value < 0) or (value >= self.ds.shape[0]):
            raise IndexError('X position out of bounds')
        
        self._xp = value
        #print 'z changed'
        self.OnChange()

    @property
    def yp(self):
        return self._yp

    @yp.setter
    def yp(self, value):
        if (value < 0) or (value >= self.ds.shape[1]):
            raise IndexError('Y position out of bounds')
        
        self._yp = value
        #print 'z changed'
        self.OnChange()

    @property
    def tp(self):
        return self._tp

    @tp.setter
    def tp(self, value):
        if (value < 0) or (value >= self.ds.shape[3]):
            raise IndexError('Z position out of bounds')
    
        self._tp = value
        #print 'z changed'
        self.OnChange()
        
    @property
    def complexMode(self):
        return self._complexMode

    @complexMode.setter
    def complexMode(self, value):
        """
        
        Parameters
        ----------
        value : str
            one of 'real', 'imag', 'abs', 'angle', 'coloured', 'imag coloured'

        """
        self._complexMode = value
        #print 'z changed'
        self.OnChange()
        
    @property
    def thresholds(self):
        return [(self.Offs[chanNum] + 0.5/self.Gains[chanNum]) for chanNum in range(len(self.Offs))]

    def ResetSelection(self):
        self.selection_begin_x = 0
        self.selection_begin_y = 0
        self.selection_begin_z = 0

        self.selection_end_x = self.ds.shape[0] - 1
        self.selection_end_y = self.ds.shape[1] - 1
        self.selection_end_z = self.ds.shape[2] - 1
        
        self.selection_trace = []

    def SetSelection(self, begin, end):
        (b_x, b_y, b_z) = begin
        (e_x, e_y, e_z) = end
        self.selection_begin_x = b_x
        self.selection_begin_y = b_y
        self.selection_begin_z = b_z

        self.selection_end_x = e_x
        self.selection_end_y = e_y
        self.selection_end_z = e_z
        
    @property
    def selection(self):
        return self.selection_begin_x, self.selection_end_x, self.selection_begin_y, self.selection_end_y, self.selection_begin_z, self.selection_end_z
    
    @property
    def sorted_selection(self):
        return sorted([self.selection_begin_x, self.selection_end_x]) + \
               sorted([self.selection_begin_y, self.selection_end_y]) + \
               sorted([self.selection_begin_z, self.selection_end_z])
        
    def EndSelection(self):
        self.on_selection_end.send(self)

    def GetSliceSelection(self):
        if(self.slice == self.SLICE_XY):
            lx = self.selection_begin_x
            ly = self.selection_begin_y
            hx = self.selection_end_x
            hy = self.selection_end_y
        elif(self.slice == self.SLICE_XZ):
            lx = self.selection_begin_x
            ly = self.selection_begin_z
            hx = self.selection_end_x
            hy = self.selection_end_z
        elif(self.slice == self.SLICE_YZ):
            lx = self.selection_begin_y
            ly = self.selection_begin_z
            hx = self.selection_end_y
            hy = self.selection_end_z

        return lx, ly, hx, hy

    def SetDataStack(self, datasource):
        self.ds = dataWrap.Wrap(datasource) #make sure data is wrapped

        nchans = self.ds.shape[self.ds.ndim -1]
        
        self.nz = self.ds.shape[2]
        if (self.ds.ndim >=5):
            self.nt = self.ds.shape[3]
        else:
            self.nt = 1

        if not nchans == len(self.Chans):
            if nchans == 1:
                self.Chans = [0]
                self.Gains = [1]
                self.Offs = [0]
                self.cmaps = [cm.gray]
                try:
                    if np.iscomplexobj(self.ds[0,0, 0, 0, 0]):
                        self.cmaps = [cm.jet]
                        self.cmax_offset = -np.pi
                        self.cmax_scale = 1./(2*np.pi)
                except IndexError:
                    pass
                self.show = [True]
            else:
                self.Chans = []
                self.Gains = []
                self.Offs = []
                self.cmaps = []
                self.show = []

                cms = [cm.r, cm.g, cm.b]

                for i in range(nchans):
                    self.Chans.append(i)
                    self.Gains.append(1.)
                    self.Offs.append(0.)
                    self.cmaps.append(cms[i%len(cms)])
                    self.show.append(True)

        self.names = ['Chan %d' %i for i in range(nchans)]
        

        self.OnChange()

    def SetGain(self, chan, gain):
        self.Gains[chan] = gain
        self.OnChange()
        
    def Show(self, chan, sh=True):
        self.show[chan] = sh
        self.OnChange()

    def SetOffset(self, chan, offset):
        self.Offs[chan] = offset
        self.OnChange()

    def SetCMap(self, chan, cmap):
        self.cmaps[chan] = cmap
        self.OnChange()

    def SetOrientation(self, orientation):
        self.orientation = orientation
        self.OnChange()

    def SetSlice(self, slice):
        self.slice = slice
        self.OnChange()

    def SetAspect(self, aspect):
        if np.isscalar(aspect):
            self.aspect = [1., 1., aspect]
        elif len(aspect) == 3:
            self.aspect = aspect
        else:
            self.aspect = [1., 1., 1.]
            
        self.OnChange()

    def SetScale(self, scale):
        self.scale = scale
        self.OnChange()

    def OnChange(self):
        if not self.inOnChange:
            self.inOnChange = True
            try:
                for fcn in self.WantChangeNotification:
                    #print('do - notifying %s' % fcn)
                    fcn()
                    
                for syn in self.syncedWith:
                    syn.Synchronise(self)
            
            except Exception as e: 
                logger.exception('Error in OnChange')
            finally:
                self.inOnChange = False
            
            
    def GetActiveChans(self):
        return [(self.Chans[i],self.Offs[i],self.Gains[i],self.cmaps[i]) for i in range(len(self.show)) if self.show[i]]
        
    def Synchronise(self, other):
        self.xp = other.xp
        self.yp = other.yp
        self.zp = other.zp
        self.tp = other.tp
            

    def _optimal_display_range(self, d, method='percentile'):
        """ Estimate a suitable starting display range
        """
        
        #discard NaNa
        d = d[np.isnan(d) == 0].ravel()
        
        if method == 'min-max':
            return d.min(), d.max()
        elif method == 'percentile':
            return d.min(), np.percentile(d, 99.)
        
    def get_hist_data(self):
        """
        Get data to display in an image histogram. Only returns a subset of the data for large images
        
        Returns
        -------

        """
        
        chan_d = []
        
        local_scaling = False

        sx, sy = self.ds.shape[:2]
        xr = (0, sx)
        yr = (0, sy)

        try:
            chunks = self.ds.chunks
    
            if not ((chunks[0] == sx) and (chunks[1] == sy)):
                # we have a chunked dataset where the chunks sub-sample the x-y space, do not attempt to load a full xy
                # slice into memoru
                local_scaling = True
                
                xr = (max(self.xp - 50, 0), min(self.xp + 50, sx))
                yr = (max(self.yp - 50, 0), min(self.yp + 50, sy))
        except AttributeError:
            pass
        
        for i in range(len(self.Chans)):
            if self.ds.ndim >= 5:
                c = self.ds[xr[0]:xr[1], yr[0]:yr[1], self.zp, self.tp, self.Chans[i]].ravel()
            elif self.ds.ndim == 4:
                c = self.ds[xr[0]:xr[1], yr[0]:yr[1],self.zp, self.Chans[i]].ravel()
            elif self.ds.ndim == 3:
                c = self.ds[xr[0]:xr[1], yr[0]:yr[1], self.zp].ravel()
            else:
                c = self.ds[xr[0]:xr[1], yr[0]:yr[1]]
            
            if np.iscomplexobj(c):
                if self.complexMode == 'real':
                    c = c.real
                elif self.complexMode == 'imag':
                    c = c.imag
                elif self.complexMode == 'angle':
                    c = np.angle(c)
                else:
                    c = np.abs(c)
                    
            if c.size > 1e4:
                c = c[::int(np.floor(c.size/1e4))]
            
            chan_d.append(c)
            
        return chan_d

    def Optimise(self):
        bds = [self._optimal_display_range(c) for c in self.get_hist_data()]
        
        for i, bd in enumerate(bds):
            low, high = bd

            self.Offs[i] = float(low)
            self.Gains[i] = 1.0/(float(high) - float(low)+ 1e-3)
                
        #print self.Offs, self.Gains

        self.OnChange()
