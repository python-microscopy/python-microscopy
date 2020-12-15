#!/usr/bin/python
##################
# photophysics.py
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

import wx
import numpy as np

class DecayAnalyser:
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Analysis>Photophysics', "Estimate decay lifetimes", self.OnCalcDecays)
        visFr.AddMenuItem('Analysis>Photophysics', "Retrieve Intensity steps", self.OnRetrieveIntensitySteps)

    def OnCalcDecays(self, event):
        from PYME.Analysis.BleachProfile import kinModels
        
        pipeline = self.visFr.pipeline

        if 'clumpSize' not in pipeline.keys():
            # Clump 
            self.visFr.particleTracker.OnFindClumps()

        kinModels.fitDecay(pipeline)
        kinModels.fitFluorBrightness(pipeline)
        #kinModels.fitFluorBrightnessT(pipeline)
        kinModels.fitOnTimes(pipeline)

    def OnRetrieveIntensitySteps(self, event):
        from PYME.Analysis import piecewiseMapping

        pipeline = self.visFr.pipeline        
        
        fw = piecewiseMapping.GeneratePMFromProtocolEvents(pipeline.events, pipeline.mdh, pipeline.mdh.getEntry('StartTime'), 10)
        pipeline.selectedDataSource.addColumn('filter', fw(pipeline.selectedDataSource['t']))
        #pipeline.selectedDataSource.setMapping('filter', 'fw')
        pipeline.selectedDataSource.setMapping('ColourNorm', '0.0*t')

        vals = list(set(pipeline.selectedDataSource['filter']))

        for key in vals:
            pipeline.mapping.setMapping('p_%d' % key, 'filter == %d' % key)

        #self.visFr.UpdatePointColourChoices()
        self.visFr.colourFilterPane.UpdateColourFilterChoices()
        


def Plug(visFr):
    """Plugs this module into the gui"""
    DecayAnalyser(visFr)



