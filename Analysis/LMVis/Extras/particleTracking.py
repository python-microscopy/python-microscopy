#!/usr/bin/python
##################
# particleTracking.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
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

    def OnTrackMolecules(self, event):
        import PYME.Analysis.DeClump.deClumpGUI as deClumpGUI
        import PYME.Analysis.DeClump.deClump as deClump
        import PYME.Analysis.trackUtils as trackUtils

        visFr = self.visFr

        bCurr = wx.BusyCursor()
        dlg = deClumpGUI.deClumpDialog(visFr)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            nFrames = dlg.GetClumpTimeWindow()
            rad_var = dlg.GetClumpRadiusVariable()
            if rad_var == '1.0':
                delta_x = 0*visFr.mapping['x'] + dlg.GetClumpRadiusMultiplier()
            else:
                delta_x = dlg.GetClumpRadiusMultiplier()*visFr.mapping[rad_var]

        clumpIndices = deClump.findClumps(visFr.mapping['t'].astype('i'), visFr.mapping['x'].astype('f4'), visFr.mapping['y'].astype('f4'), delta_x.astype('f4'), nFrames)
        numPerClump, b = np.histogram(clumpIndices, np.arange(clumpIndices.max() + 1.5) + .5)

        trackVelocities = trackUtils.calcTrackVelocity(visFr.mapping['x'], visFr.mapping['y'], clumpIndices)
        #print b

        visFr.selectedDataSource.clumpIndices = -1*np.ones(len(visFr.selectedDataSource['x']))
        visFr.selectedDataSource.clumpIndices[visFr.filter.Index] = clumpIndices

        visFr.selectedDataSource.clumpSizes = np.zeros(visFr.selectedDataSource.clumpIndices.shape)
        visFr.selectedDataSource.clumpSizes[visFr.filter.Index] = numPerClump[clumpIndices - 1]

        visFr.selectedDataSource.trackVelocities = np.zeros(visFr.selectedDataSource.clumpIndices.shape)
        visFr.selectedDataSource.trackVelocities[visFr.filter.Index] = trackVelocities

        visFr.selectedDataSource.setMapping('clumpIndex', 'clumpIndices')
        visFr.selectedDataSource.setMapping('clumpSize', 'clumpSizes')
        visFr.selectedDataSource.setMapping('trackVelocity', 'trackVelocities')

        visFr.RegenFilter()
        visFr.CreateFoldPanel()

        dlg.Destroy()

    def OnCalcMSDs(self,event):
        import pylab
        from PYME.Analysis._fithelpers import *
        from PYME.Analysis.DistHist import msdHistogram

        def powerMod(p,t):
            D, alpha = p
            return 4*D*t**alpha #factor 4 for 2D (6 for 3D)

        clumps = set(self.visFr.mapping['clumpIndex'])

        dt = self.visFr.mdh.getEntry('Camera.CycleTime')


        Ds = np.zeros(len(clumps))
        Ds_ =  np.zeros(self.visFr.mapping['x'].shape)
        alphas = np.zeros(len(clumps))
        alphas_ =  np.zeros(self.visFr.mapping['x'].shape)
        error_Ds = np.zeros(len(clumps))

        pylab.figure()

        for i, ci in enumerate(clumps):
            I = self.visFr.mapping['clumpIndex'] == ci

            x = self.visFr.mapping['x'][I]
            y = self.visFr.mapping['y'][I]
            t = self.visFr.mapping['t'][I]

            nT = (t.max() - t.min())/2


            h = msdHistogram(x, y, t, nT)

            t_ = dt*np.arange(len(h))

            pylab.plot(t_, h)

            res = FitModel(powerMod, [h[-1]/t_[-1], 1.], h, t_)

            Ds[i] = res[0][0]
            Ds_[I] = res[0][0]
            alphas[i] = res[0][1]
            alphas_[I] = res[0][1]

            print res[0]#, res[1]
            error_Ds[i] = np.sqrt(res[1][0,0])

        pylab.figure()
        pylab.scatter(Ds, alphas)

        self.visFr.selectedDataSource.diffusionConstants = -1*np.ones(self.visFr.selectedDataSource.clumpIndices.shape)
        self.visFr.selectedDataSource.diffusionConstants[self.visFr.filter.Index] = Ds_

        self.visFr.selectedDataSource.diffusionExponents = np.zeros(self.visFr.selectedDataSource.clumpIndices.shape)
        self.visFr.selectedDataSource.diffusionExponents[self.visFr.filter.Index] = alphas_

        self.visFr.selectedDataSource.setMapping('diffusionConst', 'diffusionConstants')
        self.visFr.selectedDataSource.setMapping('diffusionExp', 'diffusionExponents')

        self.visFr.RegenFilter()
        self.visFr.CreateFoldPanel()



def Plug(visFr):
    '''Plugs this module into the gui'''
    ParticleTracker(visFr)