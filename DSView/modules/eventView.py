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
from PYME.Analysis import piecewiseMapping

def Plug(dsviewer):
    image = dsviewer.image
    if 'events' in dir(image) and len(image.events) > 0:
        st = image.mdh.getEntry('StartTime')
        if 'EndTime' in image.mdh.getEntryNames():
            et = image.mdh.getEntry('EndTime')
        else:
            et = piecewiseMapping.framesToTime(image.data.getNumSlices(), image.events, image.mdh)
        dsviewer.elv = eventLogViewer.eventLogTPanel(dsviewer, image.events, image.mdh, [0, et-st]);
        dsviewer.AddPage(dsviewer.elv, False, 'Events')

        charts = []

        if 'ProtocolFocus' in dsviewer.elv.evKeyNames:
            dsviewer.zm = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, image.mdh, image.mdh.getEntry('StartTime'), image.mdh.getEntry('Protocol.PiezoStartPos'))
            charts.append(('Focus [um]', dsviewer.zm, 'ProtocolFocus'))

        if 'ScannerXPos' in dsviewer.elv.evKeyNames:
            x0 = 0
            if 'Positioning.Stage_X' in image.mdh.getEntryNames():
                x0 = image.mdh.getEntry('Positioning.Stage_X')
            dsviewer.xm = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, image.mdh, image.mdh.getEntry('StartTime'), x0, 'ScannerXPos', 0)
            charts.append(('XPos [um]', dsviewer.xm, 'ScannerXPos'))

        if 'ScannerYPos' in dsviewer.elv.evKeyNames:
            y0 = 0
            if 'Positioning.Stage_Y' in image.mdh.getEntryNames():
                y0 = image.mdh.getEntry('Positioning.Stage_Y')
            dsviewer.ym = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, image.mdh, image.mdh.getEntry('StartTime'), y0, 'ScannerYPos', 0)
            charts.append(('YPos [um]', dsviewer.ym, 'ScannerYPos'))

        if 'ScannerZPos' in dsviewer.elv.evKeyNames:
            z0 = 0
            if 'Positioning.PIFoc' in image.mdh.getEntryNames():
                z0 = image.mdh.getEntry('Positioning.PIFoc')
            dsviewer.zm = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, image.mdh, image.mdh.getEntry('StartTime'), z0, 'ScannerZPos', 0)
            charts.append(('ZPos [um]', dsviewer.zm, 'ScannerZPos'))

        dsviewer.elv.SetCharts(charts)


