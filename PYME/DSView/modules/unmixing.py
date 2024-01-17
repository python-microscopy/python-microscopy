#!/usr/bin/python

###############
# unmixing.py
#
# Copyright David Baddeley, 2012
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
################


import wx
import numpy as np

#from PYME.DSView.arrayViewPanel import *


class UnMixSettingsDlg(wx.Dialog):
    def __init__(self, parent, size=(-1, -1), mixMatrix = None, offset = 0.):
        wx.Dialog.__init__(self,parent, -1, 'Linear Unmixing', size=size)

        if not mixMatrix:
            mixMatrix = np.eye(2)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        psizer = wx.BoxSizer(wx.HORIZONTAL)

        bsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Mix Matrix'), wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.tMM00 = wx.TextCtrl(self, -1, '%1.2f'%(mixMatrix[0,0]), size=(40,-1))
        hsizer.Add(self.tMM00, 1, wx.ALL,2 )

        self.tMM01 = wx.TextCtrl(self, -1, '%1.2f'%(mixMatrix[0,1]), size=(40,-1))
        hsizer.Add(self.tMM01, 1, wx.ALL,2 )

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.tMM10 = wx.TextCtrl(self, -1, '%1.2f'%(mixMatrix[1,0]), size=(40,-1))
        hsizer.Add(self.tMM10, 1, wx.ALL,2 )

        self.tMM11 = wx.TextCtrl(self, -1, '%1.2f'%(mixMatrix[1,1]), size=(40,-1))
        hsizer.Add(self.tMM11, 1, wx.ALL,2 )

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        psizer.Add(bsizer, 0, wx.ALL, 0)


        bsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Offset'), wx.HORIZONTAL)
        self.tOffset = wx.TextCtrl(self, -1, '%1.2f'%(offset), size=(40,-1))
        #self.bOK = wx.Button(self, -1, 'OK', style = wx.BU_EXACTFIT)

        bsizer.Add(self.tOffset, 1, wx.ALL, 0)
        #bsizer.Add(self.bGrabOffset, 0, wx.LEFT, 5)
        psizer.Add(bsizer, 1, wx.LEFT|wx.RIGHT, 5)
        
        


        vsizer.Add(psizer, 1, wx.ALL|wx.EXPAND, 0)
        
        self.bOK = wx.Button(self, wx.ID_OK, 'OK')
        vsizer.Add(self.bOK, 0, wx.ALL|wx.EXPAND, 2)
        self.SetSizerAndFit(vsizer)


    def GetMixMatrix(self):
        mixMatrix = np.zeros([2,2])
        mixMatrix[0,0]= float(self.tMM00.GetValue())
        mixMatrix[0,1]= float(self.tMM01.GetValue())
        mixMatrix[1,0]= float(self.tMM10.GetValue())
        mixMatrix[1,1]= float(self.tMM11.GetValue())
        
        return mixMatrix
        
    def GetOffset(self):
        return float(self.tOffset.GetValue())

                                       
from ._base import Plugin
class Unmixer(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        
        dsviewer.AddMenuItem('Processing', "&Unmix", self.OnUnmix)

    def OnUnmix(self, event):
        #from PYME.Analysis import deTile
        from PYME.DSView import ViewIm3D, ImageStack
        from scipy import linalg

        mdh = self.image.mdh
        
        if 'Camera.ADOffset' in mdh.getEntryNames():
            offset = mdh['Camera.ADOffset']
        else:
            offset = 0
        
        dlg = UnMixSettingsDlg(self.dsviewer, offset=offset)
        if dlg.ShowModal() == wx.ID_OK:
            mm = dlg.GetMixMatrix()
            off = dlg.GetOffset()
            
            mmi = linalg.inv(mm)
            
            ch0 = self.image.data[:,:,:,0] - off
            ch1 = self.image.data[:,:,:,1] - off
            
            um0 = mmi[0,0]*ch0 + mmi[0,1]*ch1
            um1 = mmi[1,0]*ch0 + mmi[1,1]*ch1
            
            im = ImageStack([um0, um1], titleStub = 'Unmixed Image')
            im.mdh.copyEntriesFrom(self.image.mdh)
            im.mdh['Parent'] = self.image.filename
            im.mdh['Unmixing.MixMatrix'] = mm
            im.mdh['Unmixing.Offset'] = off
            #im.mdh['Processing.GaussianFilter'] = sigmas
    
            if self.dsviewer.mode == 'visGUI':
                mode = 'visGUI'
            else:
                mode = 'lite'
    
            dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))


def Plug(dsviewer):
    return Unmixer(dsviewer)
                                       
    
