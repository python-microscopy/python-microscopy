#!/usr/bin/python
##################
# particleTracking.py
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

class ParticleTracker:
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Extras>Tracking/Chaining', "&Track single molecule trajectories", self.OnTrackMolecules)
        visFr.AddMenuItem('Extras>Tracking/Chaining', "Plot Mean Squared Displacement", self.OnCalcMSDs)
        visFr.AddMenuItem('Extras>Tracking/Chaining', "Coalesce clumps", self.OnCoalesce)

    def _OnTrackMolecules(self, event):
        import PYME.Analysis.points.DeClump.deClumpGUI as deClumpGUI
        #import PYME.Analysis.points.DeClump.deClump as deClump
        import PYME.Analysis.Tracking.trackUtils as trackUtils

        visFr = self.visFr
        pipeline = visFr.pipeline

        #bCurr = wx.BusyCursor()
        dlg = deClumpGUI.deClumpDialog(None)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            trackUtils.findTracks(pipeline, dlg.GetClumpRadiusVariable(),dlg.GetClumpRadiusMultiplier(), dlg.GetClumpTimeWindow())

            pipeline.Rebuild()

        dlg.Destroy()
        
    def OnTrackMolecules(self, event):
        import PYME.Analysis.points.DeClump.deClumpGUI as deClumpGUI
        #import PYME.Analysis.points.DeClump.deClump as deClump
        import PYME.Analysis.Tracking.trackUtils as trackUtils

        visFr = self.visFr
        pipeline = visFr.pipeline

        #bCurr = wx.BusyCursor()
        dlg = deClumpGUI.deClumpDialog(None)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            from PYME.recipes import tracking
            recipe = self.visFr.pipeline.recipe
    
            recipe.add_module(tracking.FindClumps(recipe, inputName=pipeline.selectedDataSourceKey, outputName='with_clumps',
                                                  timeWindow=dlg.GetClumpTimeWindow(),
                                                  clumpRaduisVariable=dlg.GetClumpRadiusVariable(),
                                                  clumpRadiusScale=dlg.GetClumpRadiusMultiplier()))
    
            recipe.execute()
            self.visFr.pipeline.selectDataSource('with_clumps')
            self.visFr.CreateFoldPanel() #TODO: can we capture this some other way?

        dlg.Destroy()

    def OnCalcMSDs(self,event):
        import pylab
        from PYME.Analysis import _fithelpers as fh
        from PYME.Analysis.points.DistHist import msdHistogram

        def powerMod(p,t):
            D, alpha = p
            return 4*D*t**alpha #factor 4 for 2D (6 for 3D)
            
        pipeline = self.visFr.pipeline

        clumps = set(pipeline.mapping['clumpIndex'])

        dt = pipeline.mdh.getEntry('Camera.CycleTime')


        Ds = np.zeros(len(clumps))
        Ds_ =  np.zeros(pipeline.mapping['x'].shape)
        alphas = np.zeros(len(clumps))
        alphas_ =  np.zeros(pipeline.mapping['x'].shape)
        error_Ds = np.zeros(len(clumps))

        pylab.figure()

        for i, ci in enumerate(clumps):
            I = pipeline.mapping['clumpIndex'] == ci

            x = pipeline.mapping['x'][I]
            y = pipeline.mapping['y'][I]
            t = pipeline.mapping['t'][I]

            nT = (t.max() - t.min())/2


            h = msdHistogram(x, y, t, nT)

            t_ = dt*np.arange(len(h))

            pylab.plot(t_, h)

            res = fh.FitModel(powerMod, [h[-1]/t_[-1], 1.], h, t_)

            Ds[i] = res[0][0]
            Ds_[I] = res[0][0]
            alphas[i] = res[0][1]
            alphas_[I] = res[0][1]

            print((res[0]))#, res[1]
            if not res[1] is None:
                error_Ds[i] = np.sqrt(res[1][0,0])
            else:
                error_Ds[i] = -1e3

        pylab.figure()
        pylab.scatter(Ds, alphas)

        pipeline.addColumn('diffusionConst', Ds_, -1)
        pipeline.addColumn('diffusionExp', alphas_)

        pipeline.Rebuild()
        
    def _OnCoalesce(self, event):
        from PYME.IO import tabular
        from PYME.Analysis.points.DeClump import pyDeClump
        
        pipeline = self.visFr.pipeline
        
        dclumped = pyDeClump.coalesceClumps(pipeline.selectedDataSource.resultsSource.fitResults, pipeline.selectedDataSource['clumpIndex'])
        ds = tabular.fitResultsSource(dclumped)

        pipeline.addDataSource('Coalesced',  ds)
        pipeline.selectDataSource('Coalesced')

        self.visFr.CreateFoldPanel() #TODO: can we capture this some other way?
        
    def OnCoalesce(self, event):
        from PYME.recipes import localisations
        recipe = self.visFr.pipeline.recipe
    
        recipe.add_module(localisations.MergeClumps(recipe, inputName='with_clumps', outputName='coalesced'))
    
        recipe.execute()
        self.visFr.pipeline.selectDataSource('coalesced')
        self.visFr.CreateFoldPanel() #TODO: can we capture this some other way?



def Plug(visFr):
    """Plugs this module into the gui"""
    ParticleTracker(visFr)
