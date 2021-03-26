#!/usr/bin/python
##################
# eventView.py
#
# Copyright David Baddeley, 2011
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
from PYME.DSView import eventLogViewer
from PYME.Analysis import piecewiseMapping

def Update(dsviewer):
    image = dsviewer.image
    if 'events' in dir(image) and len(image.events) > 0:
        st = image.mdh.getEntry('StartTime')
        if 'EndTime' in image.mdh.getEntryNames():
            et = image.mdh.getEntry('EndTime')
        else:
            et = piecewiseMapping.frames_to_times(image.data.getNumSlices(), image.events, image.mdh)
            
        dsviewer.elv.SetRange([0, et - st])
        #dsviewer.elv.SetRange([st, et])

def Plug(dsviewer):
    image = dsviewer.image
    if 'events' in dir(image) and len(image.events) > 0:
        st = min(image.events['Time'].min() - image.mdh['StartTime'], 0)
        stt = image.mdh.getEntry('StartTime')
        if 'EndTime' in image.mdh.getEntryNames():
            et = image.mdh.getEntry('EndTime')
        else:
            et = piecewiseMapping.frames_to_times(image.data.getNumSlices(), image.events, image.mdh)
        dsviewer.elv = eventLogViewer.eventLogTPanel(dsviewer, image.events, image.mdh, [st, et-stt], activate=True)
        dsviewer.AddPage(dsviewer.elv, False, 'Events')

        charts = []

        if b'ProtocolFocus' in dsviewer.elv.evKeyNames:
            dsviewer.zm = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, image.mdh, image.mdh.getEntry('StartTime'), image.mdh.getEntry('Protocol.PiezoStartPos'))
            charts.append(('Focus [um]', dsviewer.zm, b'ProtocolFocus'))

        if b'ScannerXPos' in dsviewer.elv.evKeyNames:
            x0 = 0
            if 'Positioning.Stage_X' in image.mdh.getEntryNames():
                x0 = image.mdh.getEntry('Positioning.Stage_X')
            dsviewer.xm = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, image.mdh, image.mdh.getEntry('StartTime'), x0, b'ScannerXPos', 0)
            charts.append(('XPos [um]', dsviewer.xm, b'ScannerXPos'))

        if b'ScannerYPos' in dsviewer.elv.evKeyNames:
            y0 = 0
            if 'Positioning.Stage_Y' in image.mdh.getEntryNames():
                y0 = image.mdh.getEntry('Positioning.Stage_Y')
            dsviewer.ym = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, image.mdh, image.mdh.getEntry('StartTime'), y0, b'ScannerYPos', 0)
            charts.append(('YPos [um]', dsviewer.ym, b'ScannerYPos'))

        if b'ScannerZPos' in dsviewer.elv.evKeyNames:
            z0 = 0
            if 'Positioning.PIFoc' in image.mdh.getEntryNames():
                z0 = image.mdh.getEntry('Positioning.PIFoc')
            dsviewer.zm = piecewiseMapping.GeneratePMFromEventList(dsviewer.elv.eventSource, image.mdh, image.mdh.getEntry('StartTime'), z0, b'ScannerZPos', 0)
            charts.append(('ZPos [um]', dsviewer.zm, b'ScannerZPos'))

        dsviewer.elv.SetCharts(charts)
        
        dsviewer.updateHooks.append(Update)


