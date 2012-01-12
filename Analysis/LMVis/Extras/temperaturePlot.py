#!/usr/bin/python
##################
# temperaturePlot.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx

class TempPlotter:
    def __init__(self, visFr):
        self.visFr = visFr

        ID_PLOT_TEMPERATURE = wx.NewId()
        visFr.extras_menu.Append(ID_PLOT_TEMPERATURE, "Plot temperature record")
        visFr.Bind(wx.EVT_MENU, self.OnPlotTemperature, id=ID_PLOT_TEMPERATURE)

    def OnPlotTemperature(self, event):
        from PYME.misc import tempDB
        import pylab
        
        pipeline = self.visFr.pipeline

        t, tm = tempDB.getEntries(pipeline.mdh.getEntry('StartTime'), pipeline.mdh.getEntry('EndTime'))
        t_, tm_ = tempDB.getEntries(pipeline.mdh.getEntry('StartTime') - 3600, pipeline.mdh.getEntry('EndTime'))

        pylab.figure()
        pylab.plot((t_ - pipeline.mdh.getEntry('StartTime'))/60, tm_)
        pylab.plot((t - pipeline.mdh.getEntry('StartTime'))/60, tm, lw=2)
        pylab.xlabel('Time [mins]')
        pylab.ylabel('Temperature [C]')



def Plug(visFr):
    '''Plugs this module into the gui'''
    TempPlotter(visFr)


