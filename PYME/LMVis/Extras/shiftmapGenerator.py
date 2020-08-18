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
# import pylab
import numpy as np
import wx
import os

from PYME.IO.FileUtils import nameUtils

class ShiftmapGenerator:
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Corrections>Shiftmaps', "Calculate &Shiftmap", self.OnGenShiftmap)
        visFr.AddMenuItem('Corrections>Shiftmaps', "Calculate Shiftmap (model based)", self.OnGenShiftmapQuad)
        visFr.AddMenuItem('Corrections>Shiftmaps', "Calculate 3D Shiftmap (model based)", self.OnGenShiftmapQuadz)

    def OnGenShiftmap(self, event):
        from PYME.Analysis.points import twoColour, twoColourPlot

        pipeline = self.visFr.pipeline

        vs = pipeline.mdh.voxelsize_nm
        x0 = (pipeline.mdh['Camera.ROIPosX']-1)*vs[0]
        y0 = (pipeline.mdh['Camera.ROIPosY']-1)*vs[1]
        
        lx = len(pipeline.filter['x'])
        bbox = None#[0,(pipeline.mdh['Camera.ROIWidth'] + 1)*vs[0], 0,(pipeline.mdh['Camera.ROIHeight'] + 1)*vs[1]]
        dx, dy, spx, spy, good = twoColour.genShiftVectorFieldSpline(x0+pipeline.filter['x']+.1*np.random.randn(lx), y0+pipeline.filter['y']+.1*np.random.randn(lx), pipeline.filter['fitResults_dx'], pipeline.filter['fitResults_dy'], pipeline.filter['fitError_dx'], pipeline.filter['fitError_dy'], bbox=bbox)
        #twoColourPlot.PlotShiftField(dx, dy, spx, spy)
        twoColourPlot.PlotShiftField2(spx, spy, pipeline.mdh['Splitter.Channel0ROI'][2:], voxelsize=vs)
        twoColourPlot.PlotShiftResiduals(x0 + pipeline['x'][good], y0 + pipeline['y'][good], pipeline['fitResults_dx'][good], pipeline['fitResults_dy'][good], spx, spy)

        from six.moves import cPickle

        defFile = os.path.splitext(os.path.split(self.visFr.GetTitle())[-1])[0] + '.sf'

        fdialog = wx.FileDialog(None, 'Save shift field as ...',
            wildcard='Shift Field file (*.sf)|*.sf', style=wx.FD_SAVE, defaultDir = nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()
            #save as a pickle containing the data and voxelsize

            fid = open(fpath, 'wb')
            cPickle.dump((spx, spy), fid, 2)
            fid.close()
            
    def OnGenShiftmapQuad(self, event):
        from PYME.Analysis.points import twoColour, twoColourPlot
        from PYME.IO.MetaDataHandler import get_camera_roi_origin

        pipeline = self.visFr.pipeline

        vs = pipeline.mdh.voxelsize_nm
        
        roi_x0, roi_y0 = get_camera_roi_origin(pipeline.mdh)
        
        x0 = (roi_x0)*vs[0]
        y0 = (roi_y0)*vs[1]
        
        lx = len(pipeline.filter['x'])
        bbox = None#[0,(pipeline.mdh['Camera.ROIWidth'] + 1)*vs[0], 0,(pipeline.mdh['Camera.ROIHeight'] + 1)*vs[1]]
        dx, dy, spx, spy, good = twoColour.genShiftVectorFieldQ(pipeline.filter['x']+.1*np.random.randn(lx) + x0, pipeline.filter['y']+.1*np.random.randn(lx) + y0, pipeline.filter['fitResults_dx'], pipeline.filter['fitResults_dy'], pipeline.filter['fitError_dx'], pipeline.filter['fitError_dy'], bbox=bbox)
        #twoColourPlot.PlotShiftField(dx, dy, spx, spy)
        twoColourPlot.PlotShiftField2(spx, spy, pipeline.mdh['Splitter.Channel0ROI'][2:], voxelsize=vs)
        twoColourPlot.PlotShiftResiduals(pipeline['x'][good] + x0, pipeline['y'][good] + y0, pipeline['fitResults_dx'][good], pipeline['fitResults_dy'][good], spx, spy)

        from six.moves import cPickle

        defFile = os.path.splitext(os.path.split(self.visFr.GetTitle())[-1])[0] + '.sf'

        fdialog = wx.FileDialog(None, 'Save shift field as ...',
            wildcard='Shift Field file (*.sf)|*.sf', style=wx.FD_SAVE, defaultDir = nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()
            #save as a pickle containing the data and voxelsize

            fid = open(fpath, 'wb')
            cPickle.dump((spx, spy), fid, 2)
            fid.close()
            
    def OnGenShiftmapQuadz(self, event):
        from PYME.Analysis.points import twoColour, twoColourPlot

        pipeline = self.visFr.pipeline

        vs = pipeline.mdh.voxelsize_nm
        
        z0 = (pipeline['z']*pipeline['A']).sum()/pipeline['A'].sum()
        
        lx = len(pipeline.filter['x'])
        bbox = None#[0,(pipeline.mdh['Camera.ROIWidth'] + 1)*vs[0], 0,(pipeline.mdh['Camera.ROIHeight'] + 1)*vs[1]]
        dx, dy, spx, spy, good = twoColour.genShiftVectorFieldQz(pipeline.filter['x']+.1*np.random.randn(lx), pipeline.filter['y']+.1*np.random.randn(lx),pipeline.filter['z'] - z0, pipeline.filter['fitResults_dx'], pipeline.filter['fitResults_dy'], pipeline.filter['fitError_dx'], pipeline.filter['fitError_dy'], bbox=bbox)
        #twoColourPlot.PlotShiftField(dx, dy, spx, spy)
        twoColourPlot.PlotShiftField2(spx, spy, pipeline.mdh['Splitter.Channel0ROI'][2:], voxelsize=vs)
        twoColourPlot.PlotShiftResiduals(pipeline['x'][good], pipeline['y'][good], pipeline['fitResults_dx'][good], pipeline['fitResults_dy'][good], spx, spy, pipeline.filter['z'][good] - z0)

        from six.moves import cPickle

        defFile = os.path.splitext(os.path.split(self.visFr.GetTitle())[-1])[0] + '.sf'

        fdialog = wx.FileDialog(None, 'Save shift field as ...',
            wildcard='Shift Field file (*.sf)|*.sf', style=wx.FD_SAVE, defaultDir = nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()
            #save as a pickle containing the data and voxelsize

            fid = open(fpath, 'wb')
            cPickle.dump((spx, spy), fid, 2)
            fid.close()

def Plug(visFr):
    """Plugs this module into the gui"""
    ShiftmapGenerator(visFr)


