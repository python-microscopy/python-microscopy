#!/usr/bin/python
##################
# eventView.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
from PYME.DSView import eventLogViewer

def Plug(dsviewer):
    if 'events' in dir(dsviewer):
        st = dsviewer.mdh.getEntry('StartTime')
        if 'EndTime' in dsviewer.mdh.getEntryNames():
            et = dsviewer.mdh.getEntry('EndTime')
        else:
            et = piecewiseMapping.framesToTime(dsviewer.ds.getNumSlices(), dsviewer.events, dsviewer.mdh)
        dsviewer.elv = eventLogViewer.eventLogTPanel(dsviewer, dsviewer.events, dsviewer.mdh, [0, et-st]);
        dsviewer.AddPage(dsviewer.elv, False, 'Events')

        charts = []

        if 'ProtocolFocus' in dsviewer.elv.evKeyNames:
            dsviewer.zm = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, dsviewer.mdh, dsviewer.mdh.getEntry('StartTime'), dsviewer.mdh.getEntry('Protocol.PiezoStartPos'))
            charts.append(('Focus [um]', dsviewer.zm, 'ProtocolFocus'))

        if 'ScannerXPos' in dsviewer.elv.evKeyNames:
            x0 = 0
            if 'Positioning.Stage_X' in dsviewer.mdh.getEntryNames():
                x0 = dsviewer.mdh.getEntry('Positioning.Stage_X')
            dsviewer.xm = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, dsviewer.mdh, dsviewer.mdh.getEntry('StartTime'), x0, 'ScannerXPos', 0)
            charts.append(('XPos [um]', dsviewer.xm, 'ScannerXPos'))

        if 'ScannerYPos' in dsviewer.elv.evKeyNames:
            y0 = 0
            if 'Positioning.Stage_Y' in dsviewer.mdh.getEntryNames():
                y0 = dsviewer.mdh.getEntry('Positioning.Stage_Y')
            dsviewer.ym = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, dsviewer.mdh, dsviewer.mdh.getEntry('StartTime'), y0, 'ScannerYPos', 0)
            charts.append(('YPos [um]', dsviewer.ym, 'ScannerYPos'))

        if 'ScannerZPos' in dsviewer.elv.evKeyNames:
            z0 = 0
            if 'Positioning.PIFoc' in dsviewer.mdh.getEntryNames():
                z0 = dsviewer.mdh.getEntry('Positioning.PIFoc')
            dsviewer.zm = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, dsviewer.mdh, dsviewer.mdh.getEntry('StartTime'), z0, 'ScannerZPos', 0)
            charts.append(('ZPos [um]', dsviewer.zm, 'ScannerZPos'))

        dsviewer.elv.SetCharts(charts)


