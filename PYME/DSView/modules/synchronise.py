#!/usr/bin/python
##################
# vis3D.py
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
import os
#import pylab
#from PYME.IO import MetaDataHandler
#from PYME.DSView import image, View3D
#from PYME.IO import dataWrap
from PYME.DSView import dsviewer

from ._base import Plugin
class Syncer(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        
        dsviewer.AddMenuItem('Processing', "Sync Windows", self.OnSynchronise)



    def OnSynchronise(self, event):
        dlg = wx.SingleChoiceDialog(
                self.dsviewer, 'choose the image to composite with', 'Make Composite',
                list(dsviewer.openViewers.keys()),
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            other = dsviewer.openViewers[dlg.GetStringSelection()]

            other.do.syncedWith.append(self.do)

        dlg.Destroy()

    

       
    


def Plug(dsviewer):
    return Syncer(dsviewer)



