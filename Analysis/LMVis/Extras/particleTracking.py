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

def Plug(visFr):
    '''Plugs this module into the gui'''
    ParticleTracker(visFr)