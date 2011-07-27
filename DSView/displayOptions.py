#!/usr/bin/python

##################
# displayOptions.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from matplotlib import cm
import numpy as np
#import tables

from PYME.DSView import dataWrap

def fast_grey(data):
    return data[:,:,None]*np.ones((1,1,4))

fast_grey.name = 'fastGrey'
     

class DisplayOpts(object):
    UPRIGHT, ROT90 = range(2)
    SLICE_XY, SLICE_XZ, SLICE_YZ = range(3)

    ACTION_POSITION, ACTION_SELECTION = range(2)
    SELECTION_RECTANGLE, SELECTION_LINE = range(2)

    def __init__(self, datasource, xp=0, yp=0, zp=0, aspect=1):
        self.WantChangeNotification = []
        
        self.Chans = []
        self.Gains = []
        self.Offs = []
        self.cmaps = []
        self.names = []
        
        self.xp=0
        self.yp=0

        self._zp=0

        self.SetDataStack(datasource)
        self.SetAspect(aspect)
        self.ResetSelection()

        self.orientation = self.UPRIGHT
        self.slice = self.SLICE_XY
        self.scale = 1.0

        self.leftButtonAction = self.ACTION_POSITION
        self.selectionMode = self.SELECTION_RECTANGLE

        self.selectionWidth = 1

        self.showSelection=False


    @property
    def zp(self):
        return self._zp

    @zp.setter
    def zp(self, value):
        self._zp = value
        #print 'z changed'
        self.OnChange()

    def ResetSelection(self):
        self.selection_begin_x = 0
        self.selection_begin_y = 0
        self.selection_begin_z = 0

        self.selection_end_x = self.ds.shape[0] - 1
        self.selection_end_y = self.ds.shape[1] - 1
        self.selection_end_z = self.ds.shape[2] - 1

    def SetSelection(self, (b_x,b_y,b_z),(e_x,e_y,e_z)):
        self.selection_begin_x = b_x
        self.selection_begin_y = b_y
        self.selection_begin_z = b_z

        self.selection_end_x = e_x
        self.selection_end_y = e_y
        self.selection_end_z = e_z

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

        nchans = self.ds.shape[3]

        if not nchans == len(self.Chans):
            if nchans == 1:
                self.Chans = [0]
                self.Gains = [1]
                self.Offs = [0]
                self.cmaps = [fast_grey]
            else:
                self.Chans = []
                self.Gains = []
                self.Offs = []
                self.cmaps = []

                cms = [cm.r, cm.g, cm.b]

                for i in range(nchans):
                    self.Chans.append(i)
                    self.Gains.append(1.)
                    self.Offs.append(0.)
                    self.cmaps.append(cms[i%len(cms)])

        self.names = ['Chan %d' %i for i in range(nchans)]

        self.OnChange()

    def SetGain(self, chan, gain):
        self.Gains[chan] = gain
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
        for fcn in self.WantChangeNotification:
            fcn()

    def Optimise(self):
        if len(self.ds.shape) == 2:
            self.Offs[0] = 1.*self.ds.min()
            self.Gains[0] =1./(self.ds.max()- self.ds.min())
        elif len(self.ds.shape) ==3:
            self.Offs[0] = 1.*self.ds[:,:,self.zp].min()
            self.Gains[0] =1./(self.ds[:,:,self.zp].max()- self.ds[:,:,self.zp].min())
        else:
            for i in range(len(self.Chans)):
                self.Offs[i] = self.ds[:,:,self.zp,self.Chans[i]].min()
                self.Gains[i] = 1.0/(self.ds[:,:,self.zp,self.Chans[i]].max() - self.Offs[i])

        self.OnChange()