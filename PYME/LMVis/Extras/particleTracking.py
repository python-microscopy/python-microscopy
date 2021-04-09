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
from PYME.ui import progress

class ParticleTracker:
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Analysis>Tracking', "&Track single molecule trajectories", self.OnTrackMolecules)
        visFr.AddMenuItem('Corrections>Chaining', "Find consecutive appearances", self.OnFindClumps)
        visFr.AddMenuItem('Analysis>Tracking', "Plot Mean Squared Displacement", self.OnCalcMSDs)
        visFr.AddMenuItem('Corrections>Chaining', "Clump consecutive appearances", self.OnCoalesce)
        visFr.AddMenuItem('Extras', "Calculate clump widths", self.OnCalcWidths)

    # def _OnTrackMolecules(self, event):
    #     import PYME.Analysis.points.DeClump.deClumpGUI as deClumpGUI
    #     #import PYME.Analysis.points.DeClump.deClump as deClump
    #     import PYME.Analysis.Tracking.trackUtils as trackUtils
    #
    #     visFr = self.visFr
    #     pipeline = visFr.pipeline
    #
    #     #bCurr = wx.BusyCursor()
    #     dlg = deClumpGUI.deClumpDialog(None)
    #
    #     ret = dlg.ShowModal()
    #
    #     if ret == wx.ID_OK:
    #         trackUtils.findTracks(pipeline, dlg.GetClumpRadiusVariable(),dlg.GetClumpRadiusMultiplier(), dlg.GetClumpTimeWindow())
    #
    #         pipeline.Rebuild()
    #
    #     dlg.Destroy()
        
    def OnFindClumps(self, event=None):
        import PYME.Analysis.points.DeClump.deClumpGUI as deClumpGUI
        #import PYME.Analysis.points.DeClump.deClump as deClump
        import PYME.Analysis.Tracking.trackUtils as trackUtils
        

        visFr = self.visFr
        pipeline = visFr.pipeline

        #bCurr = wx.BusyCursor()
        dlg = deClumpGUI.deClumpDialog(None)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            #with progress.ComputationInProgress(visFr, 'finding consecutive appearances'):
            from PYME.recipes import tracking
            recipe = self.visFr.pipeline.recipe
    
            recipe.add_modules_and_execute([tracking.FindClumps(recipe, inputName=pipeline.selectedDataSourceKey, outputName='with_clumps',
                                                  timeWindow=dlg.GetClumpTimeWindow(),
                                                  clumpRadiusVariable=dlg.GetClumpRadiusVariable(),
                                                  clumpRadiusScale=dlg.GetClumpRadiusMultiplier()),])
    
            self.visFr.pipeline.selectDataSource('with_clumps')
            #self.visFr.CreateFoldPanel() #TODO: can we capture this some other way?

        dlg.Destroy()

    def OnTrackMolecules(self, event=None):
        import PYME.Analysis.points.DeClump.deClumpGUI as deClumpGUI
        #import PYME.Analysis.points.DeClump.deClump as deClump
        import PYME.Analysis.Tracking.trackUtils as trackUtils
        from PYME.LMVis.layers.tracks import TrackRenderLayer

        from PYME.recipes import tracking
        from PYME.recipes.tablefilters import FilterTable

        recipe = self.visFr.pipeline.recipe
    
        visFr = self.visFr
        pipeline = visFr.pipeline # type: PYME.LMVis.pipeline.Pipeline

        if hasattr(self, '_mol_tracking_module') and (self._mol_tracking_module in recipe.modules):
            # We have already tracked, edit existing tracking module instead
            wx.MessageBox('This dataset has already been tracked, edit parameters of existing tracking rather than starting again', 'Error', wx.OK|wx.ICON_ERROR, visFr)
            self._mol_tracking_module.configure_traits(kind='modal')
            return
            

        output_name = 'with_tracks'
        if output_name in recipe.namespace:
            # this should take care of, e.g. having tracked with feature based tracking or something in the recipe
            output_name = pipeline.new_ds_name('with_tracks')
            wx.MessageBox("Another module has already created a 'with_tracks' output, using the nonstandard name '%s' instead" % output_name, 'Warning', wx.OK|wx.ICON_WARNING, visFr)
        
        tracking_module = tracking.FindClumps(recipe, inputName=pipeline.selectedDataSourceKey,
                                              outputName=output_name,
                                    timeWindow=5,
                                    clumpRadiusVariable='1.0',
                                    clumpRadiusScale=250.,
                                    minClumpSize=50)
    
        if tracking_module.configure_traits(kind='modal'):
            self._mol_tracking_module = tracking_module
            recipe.add_modules_and_execute([tracking_module,
                                            # Add dynamic filtering on track length, etc.
                                            FilterTable(recipe,
                                                inputName=tracking_module.outputName,
                                                outputName='filtered_{}'.format(tracking_module.outputName),
                                                filters={'clumpSize':[tracking_module.minClumpSize, 1e6]})])
            
            self.visFr.pipeline.selectDataSource('filtered_{}'.format(tracking_module.outputName))
            #self.visFr.CreateFoldPanel() #TODO: can we capture this some other way?
            layer = TrackRenderLayer(pipeline, dsname='filtered_{}'.format(tracking_module.outputName), method='tracks')
            visFr.add_layer(layer)
        
        #dlg.Destroy()

    def OnCalcMSDs(self, event=None):
        #TODO - move this logic to reports - like dh5view module
        # import pylab
        import matplotlib.pyplot as plt
        from PYME.Analysis import _fithelpers as fh
        from PYME.Analysis.points.DistHist import msdHistogram

        def powerMod(p,t):
            D, alpha = p
            return 4*D*t**alpha #factor 4 for 2D (6 for 3D)
            
        pipeline = self.visFr.pipeline

        clumps = set(pipeline['clumpIndex'])

        dt = pipeline.mdh.getEntry('Camera.CycleTime')


        Ds = np.zeros(len(clumps))
        Ds_ =  np.zeros(pipeline['x'].shape)
        alphas = np.zeros(len(clumps))
        alphas_ =  np.zeros(pipeline['x'].shape)
        error_Ds = np.zeros(len(clumps))

        plt.figure()

        for i, ci in enumerate(clumps):
            I = pipeline['clumpIndex'] == ci

            x = pipeline['x'][I]
            y = pipeline['y'][I]
            t = pipeline['t'][I]

            nT = int((t.max() - t.min())/2)


            h = msdHistogram(x, y, t, nT)

            t_ = dt*np.arange(len(h))

            plt.plot(t_, h)

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

        plt.figure()
        plt.scatter(Ds, alphas)

        pipeline.addColumn('diffusionConst', Ds_, -1)
        pipeline.addColumn('diffusionExp', alphas_)

        pipeline.Rebuild()
        
    def _OnCoalesce(self, event=None):
        from PYME.IO import tabular
        from PYME.Analysis.points.DeClump import pyDeClump
        
        pipeline = self.visFr.pipeline
        try:
            nphotons = pipeline.selectedDataSource['nPhotons']
        except KeyError:
            nphotons = None
            
        #TODO - check what nPhotons is doing here!!!
        dclumped = pyDeClump.coalesceClumps(pipeline.selectedDataSource.resultsSource.fitResults,
                                            pipeline.selectedDataSource['clumpIndex'], nphotons)
        ds = tabular.FitResultsSource(dclumped)

        pipeline.addDataSource('Coalesced',  ds)
        pipeline.selectDataSource('Coalesced')

        #self.visFr.CreateFoldPanel() #TODO: can we capture this some other way?
        
    def OnCoalesce(self, event=None):
        #with progress.ComputationInProgress(self.visFr, 'coalescing consecutive appearances'):
        from PYME.recipes import localisations
        recipe = self.visFr.pipeline.recipe
        
        recipe.add_modules_and_execute([localisations.MergeClumps(recipe, inputName='with_clumps', outputName='coalesced'),])
    
        self.visFr.pipeline.selectDataSource('coalesced')
        #self.visFr.CreateFoldPanel() #TODO: can we capture this some other way?

    def OnCalcWidths(self, event=None):
        #FIXME - this is probably broken on modern VisGUI
        from scipy.stats import binned_statistic

        pipeline = self.visFr.pipeline
 
        # clumps = set(pipeline.mapping['clumpIndex'])
        bins = np.arange(pipeline.mapping['clumpIndex'].max()+1,dtype='float32')+0.5
        xv, edges, binnumber = binned_statistic(pipeline.mapping['clumpIndex'], pipeline.mapping['x'], statistic=np.var, bins=bins)
        xvars = xv[binnumber-1]
        yv, edges, binnumber = binned_statistic(pipeline.mapping['clumpIndex'], pipeline.mapping['y'], statistic=np.var, bins=bins)
        yvars = yv[binnumber-1]
        
        widths = np.sqrt(xvars+yvars)

        # for i, ci in enumerate(clumps):
        #     I = pipeline.mapping['clumpIndex'] == ci
            
        #     x = pipeline.mapping['x'][I]
        #     y = pipeline.mapping['y'][I]
            
        #     if len(x) > 1:
        #         #vx = x.var()
        #         #vy = y.var()
        #         vars[I] = 1.0 # math.sqrt(vx + vy)
                
        pipeline.selectedDataSource.clumpWidths = -1*np.ones(pipeline.selectedDataSource.clumpIndices.shape)
        pipeline.selectedDataSource.clumpWidths[pipeline.filter.Index] = widths
        pipeline.selectedDataSource.clumpWidthsX = -1*np.ones(pipeline.selectedDataSource.clumpIndices.shape)
        pipeline.selectedDataSource.clumpWidthsX[pipeline.filter.Index] = np.sqrt(xvars)
        pipeline.selectedDataSource.clumpWidthsY = -1*np.ones(pipeline.selectedDataSource.clumpIndices.shape)
        pipeline.selectedDataSource.clumpWidthsY[pipeline.filter.Index] = np.sqrt(yvars)
 
        pipeline.selectedDataSource.setMapping('clumpWidth', 'clumpWidths')
        pipeline.selectedDataSource.setMapping('clumpWidthX', 'clumpWidthsX')
        pipeline.selectedDataSource.setMapping('clumpWidthY', 'clumpWidthsY')
         
        self.visFr.RegenFilter()
        #self.visFr.CreateFoldPanel()
 

def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.particleTracker = ParticleTracker(visFr)
