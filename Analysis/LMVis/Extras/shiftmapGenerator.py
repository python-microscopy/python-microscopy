#!/usr/bin/python
##################
# shiftmapGenerator.py
#
# Copyright David Baddeley, 2010
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
import pylab
import wx
import os

from PYME.FileUtils import nameUtils

class ShiftmapGenerator:
    def __init__(self, visFr):
        self.visFr = visFr

        ID_GEN_SHIFTMAP = wx.NewId()
        visFr.extras_menu.Append(ID_GEN_SHIFTMAP, "Calculate &Shiftmap")
        visFr.Bind(wx.EVT_MENU, self.OnGenShiftmap, id=ID_GEN_SHIFTMAP)

    def OnGenShiftmap(self, event):
        from PYME.Analysis import twoColour, twoColourPlot

        pipeline = self.visFr.pipeline        
        
        lx = len(pipeline.filter['x'])
        dx, dy, spx, spy = twoColour.genShiftVectorFieldSpline(pipeline.filter['x']+.1*pylab.randn(lx), pipeline.filter['y']+.1*pylab.randn(lx), pipeline.filter['fitResults_dx'], pipeline.filter['fitResults_dy'], pipeline.filter['fitError_dx'], pipeline.filter['fitError_dy'])
        twoColourPlot.PlotShiftField(dx, dy, spx, spy)
        twoColourPlot.PlotShiftResiduals(pipeline['x'], pipeline['y'], pipeline['fitResults_dx'], pipeline['fitResults_dy'], spx, spy)

        import cPickle

        defFile = os.path.splitext(os.path.split(self.visFr.GetTitle())[-1])[0] + '.sf'

        fdialog = wx.FileDialog(None, 'Save shift field as ...',
            wildcard='Shift Field file (*.sf)|*.sf', style=wx.SAVE|wx.HIDE_READONLY, defaultDir = nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()
            #save as a pickle containing the data and voxelsize

            fid = open(fpath, 'wb')
            cPickle.dump((spx, spy), fid, 2)
            fid.close()

def Plug(visFr):
    '''Plugs this module into the gui'''
    ShiftmapGenerator(visFr)


