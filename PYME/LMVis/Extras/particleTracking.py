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

        ID_TRACK_MOLECULES = wx.NewId()
        visFr.extras_menu.Append(ID_TRACK_MOLECULES, "&Track single molecule trajectories")
        visFr.Bind(wx.EVT_MENU, self.OnTrackMolecules, id=ID_TRACK_MOLECULES)

        ID_PLOT_MSD = wx.NewId()
        visFr.extras_menu.Append(ID_PLOT_MSD, "Plot Mean Squared Displacement")
        visFr.Bind(wx.EVT_MENU, self.OnCalcMSDs, id=ID_PLOT_MSD)
        
        ID_COALESCE = wx.NewId()
        visFr.extras_menu.Append(ID_COALESCE, "Coalesce clumps")
        visFr.Bind(wx.EVT_MENU, self.OnCoalesce, id=ID_COALESCE)

        ID_CALCWIDTH = wx.NewId()
        visFr.extras_menu.Append(ID_CALCWIDTH, "Calculate clump widths")
        visFr.Bind(wx.EVT_MENU, self.OnCalcWidths, id=ID_CALCWIDTH)

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
            #nFrames = dlg.GetClumpTimeWindow()
            #rad_var = dlg.GetClumpRadiusVariable()
            trackUtils.findTracks(pipeline, dlg.GetClumpRadiusVariable(),dlg.GetClumpRadiusMultiplier(), dlg.GetClumpTimeWindow())
#            if rad_var == '1.0':
#                delta_x = 0*pipeline.mapping['x'] + dlg.GetClumpRadiusMultiplier()
#            else:
#                delta_x = dlg.GetClumpRadiusMultiplier()*pipeline.mapping[rad_var]
#
#        clumpIndices = deClump.findClumps(pipeline.mapping['t'].astype('i'), pipeline.mapping['x'].astype('f4'), pipeline.mapping['y'].astype('f4'), delta_x.astype('f4'), nFrames)
#        numPerClump, b = np.histogram(clumpIndices, np.arange(clumpIndices.max() + 1.5) + .5)
#
#        trackVelocities = trackUtils.calcTrackVelocity(pipeline.mapping['x'], pipeline.mapping['y'], clumpIndices)
#        #print b
#
#        pipeline.selectedDataSource.clumpIndices = -1*np.ones(len(pipeline.selectedDataSource['x']))
#        pipeline.selectedDataSource.clumpIndices[pipeline.filter.Index] = clumpIndices
#
#        pipeline.selectedDataSource.clumpSizes = np.zeros(pipeline.selectedDataSource.clumpIndices.shape)
#        pipeline.selectedDataSource.clumpSizes[pipeline.filter.Index] = numPerClump[clumpIndices - 1]
#
#        pipeline.selectedDataSource.trackVelocities = np.zeros(pipeline.selectedDataSource.clumpIndices.shape)
#        pipeline.selectedDataSource.trackVelocities[pipeline.filter.Index] = trackVelocities
#
#        pipeline.selectedDataSource.setMapping('clumpIndex', 'clumpIndices')
#        pipeline.selectedDataSource.setMapping('clumpSize', 'clumpSizes')
#        pipeline.selectedDataSource.setMapping('trackVelocity', 'trackVelocities')

            visFr.RegenFilter()
            visFr.CreateFoldPanel()

        dlg.Destroy()

    def OnCalcMSDs(self,event):
        import pylab
        import PYME.Analysis._fithelpers as fh
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

            nT = int((t.max() - t.min())/2)


            h = msdHistogram(x, y, t, nT)

            t_ = dt*np.arange(len(h))

            pylab.plot(t_, h)

            res = fh.FitModel(powerMod, [h[-1]/t_[-1], 1.], h, t_)

            Ds[i] = res[0][0]
            Ds_[I] = res[0][0]
            alphas[i] = res[0][1]
            alphas_[I] = res[0][1]

            print((res[0]))#, res[1]
            if not res[1] == None:
                error_Ds[i] = np.sqrt(res[1][0,0])
            else:
                error_Ds[i] = -1e3

        pylab.figure()
        pylab.scatter(Ds, alphas)

        ds_diffusionConstants = -1*np.ones(pipeline.selectedDataSource['clumpIndex'].shape)
        ds_diffusionConstants[pipeline.filter.Index] = Ds_

        ds_diffusionExponents = np.zeros_like(ds_diffusionConstants)
        ds_diffusionExponents[pipeline.filter.Index] = alphas_

        pipeline.selectedDataSource.addColumn('diffusionConst', ds_diffusionConstants)
        pipeline.selectedDataSource.addColumn('diffusionExp', ds_diffusionExponents)

        self.visFr.RegenFilter()
        self.visFr.CreateFoldPanel()
        
    def OnCoalesce(self, event):
        from PYME.LMVis import inpFilt
        from PYME.Analysis.points.DeClump import pyDeClump
        
        pipeline = self.visFr.pipeline
        
        dclumped = pyDeClump.coalesceClumps(pipeline.selectedDataSource.resultsSource.fitResults, pipeline.selectedDataSource['clumpIndex'])
        ds = inpFilt.fitResultsSource(dclumped)

        pipeline.addDataSource('Coalesced',  ds)
        pipeline.selectDataSource('Coalesced')
        self.visFr.RegenFilter()
        self.visFr.CreateFoldPanel()

    def OnCalcWidths(self,event):
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
        self.visFr.CreateFoldPanel()
 

def Plug(visFr):
    """Plugs this module into the gui"""
    ParticleTracker(visFr)
