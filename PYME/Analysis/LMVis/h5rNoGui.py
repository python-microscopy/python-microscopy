#!/usr/bin/python

###############
# h5rNoGui.py
#
# Copyright David Baddeley, 2012
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
################
from PYME.Analysis.LMVis import inpFilt

from PYME.Analysis import piecewiseMapping

#import time
import numpy as np
import scipy.special

from PYME.Acquire import MetaDataHandler

from PYME.Analysis.LMVis.visHelpers import ImageBounds
from PYME.Analysis.LMVis import dyeRatios
#from PYME.Analysis.BleachProfile.kinModels import getPhotonNums


class Pipeline:
    def __init__(self, filename=None):
        self.dataSources = []
        self.selectedDataSource = None
        self.filterKeys = {'error_x': (0,30), 'A':(5,2000), 'sig' : (95, 200)}

        self.filter = None
        self.mapping = None
        self.colourFilter = None

        self.fluorSpecies = {}
        self.fluorSpeciesDyes = {}
        self.chromaticShifts = {}
        self.t_p_dye = 0.1
        self.t_p_other = 0.1
        self.t_p_background = .01

        self.objThreshold = 30
        self.objMinSize = 10
        self.blobJitter = 0
        self.objects = None

        self.imageBounds = ImageBounds(0,0,0,0)
        
        self.edb = None
        self.GeneratedMeasures = {}

        if not filename==None:
            #self.glCanvas.OnPaint(None)
            self.OpenFile(filename)


    def RegenFilter(self):
        if not self.selectedDataSource == None:
            self.filter = inpFilt.resultsFilter(self.selectedDataSource, **self.filterKeys)
            if self.mapping:
                self.mapping.resultsSource = self.filter
            else:
                self.mapping = inpFilt.mappingFilter(self.filter)

            if not self.colourFilter:
                self.colourFilter = inpFilt.colourFilter(self.mapping, self)

        self.edb = None
        self.objects = None

        self.GeneratedMeasures = {}


    def OpenFile(self, filename):
        self.dataSources = []
        if 'zm' in dir(self):
            del self.zm
        self.filter = None
        self.mapping = None
        self.colourFilter = None
        self.filename = filename

        self.selectedDataSource = inpFilt.h5rSource(filename)
        self.dataSources.append(self.selectedDataSource)



        self.mdh = MetaDataHandler.HDFMDHandler(self.selectedDataSource.h5f)

        if 'Camera.ROIWidth' in self.mdh.getEntryNames():
            x0 = 0
            y0 = 0

            x1 = self.mdh.getEntry('Camera.ROIWidth')*1e3*self.mdh.getEntry('voxelsize.x')
            y1 = self.mdh.getEntry('Camera.ROIHeight')*1e3*self.mdh.getEntry('voxelsize.y')

            if 'Splitter' in self.mdh.getEntry('Analysis.FitModule'):
                y1 = y1/2

            self.imageBounds = ImageBounds(x0, y0, x1, y1)
        else:
            self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)

        if 'fitResults_Ag' in self.selectedDataSource.keys():
            #if we used the splitter set up a mapping so we can filter on total amplitude and ratio
            #if not 'fitError_Ag' in self.selectedDataSource.keys():

            if 'fitError_Ag' in self.selectedDataSource.keys():
                self.selectedDataSource = inpFilt.mappingFilter(self.selectedDataSource, A='fitResults_Ag + fitResults_Ar', gFrac='fitResults_Ag/(fitResults_Ag + fitResults_Ar)', error_gFrac = 'sqrt((fitError_Ag/fitResults_Ag)**2 + (fitError_Ag**2 + fitError_Ar**2)/(fitResults_Ag + fitResults_Ar)**2)*fitResults_Ag/(fitResults_Ag + fitResults_Ar)')
                sg = self.selectedDataSource['fitError_Ag']
                sr = self.selectedDataSource['fitError_Ar']
                g = self.selectedDataSource['fitResults_Ag']
                r = self.selectedDataSource['fitResults_Ar']
                I = self.selectedDataSource['A']
                self.selectedDataSource.colNorm = np.sqrt(2*np.pi)*sg*sr/(2*np.sqrt(sg**2 + sr**2)*I)*(
                    scipy.special.erf((sg**2*r + sr**2*(I-g))/(np.sqrt(2)*sg*sr*np.sqrt(sg**2+sr**2)))
                    - scipy.special.erf((sg**2*(r-I) - sr**2*g)/(np.sqrt(2)*sg*sr*np.sqrt(sg**2+sr**2))))
                self.selectedDataSource.setMapping('ColourNorm', '1.0*colNorm')
            else:
                self.selectedDataSource = inpFilt.mappingFilter(self.selectedDataSource, A='fitResults_Ag + fitResults_Ar', gFrac='fitResults_Ag/(fitResults_Ag + fitResults_Ar)', error_gFrac = '0*x + 0.01')
                self.selectedDataSource.setMapping('fitError_Ag', '1*sqrt(fitResults_Ag/1)')
                self.selectedDataSource.setMapping('fitError_Ar', '1*sqrt(fitResults_Ar/1)')
                sg = self.selectedDataSource['fitError_Ag']
                sr = self.selectedDataSource['fitError_Ar']
                g = self.selectedDataSource['fitResults_Ag']
                r = self.selectedDataSource['fitResults_Ar']
                I = self.selectedDataSource['A']
                self.selectedDataSource.colNorm = np.sqrt(2*np.pi)*sg*sr/(2*np.sqrt(sg**2 + sr**2)*I)*(
                    scipy.special.erf((sg**2*r + sr**2*(I-g))/(np.sqrt(2)*sg*sr*np.sqrt(sg**2+sr**2)))
                    - scipy.special.erf((sg**2*(r-I) - sr**2*g)/(np.sqrt(2)*sg*sr*np.sqrt(sg**2+sr**2))))
                self.selectedDataSource.setMapping('ColourNorm', '1.0*colNorm')

            self.dataSources.append(self.selectedDataSource)


        elif 'fitResults_sigxl' in self.selectedDataSource.keys():
            self.selectedDataSource = inpFilt.mappingFilter(self.selectedDataSource)
            self.dataSources.append(self.selectedDataSource)

            self.selectedDataSource.setMapping('sig', 'fitResults_sigxl + fitResults_sigyu')
            self.selectedDataSource.setMapping('sig_d', 'fitResults_sigxl - fitResults_sigyu')

            self.selectedDataSource.dsigd_dz = -30.
            self.selectedDataSource.setMapping('fitResults_z0', 'dsigd_dz*sig_d')
        else:
            self.selectedDataSource = inpFilt.mappingFilter(self.selectedDataSource)
            self.dataSources.append(self.selectedDataSource)



        if 'Events' in self.selectedDataSource.resultsSource.h5f.root:
            self.events = self.selectedDataSource.resultsSource.h5f.root.Events[:]

            evKeyNames = set()
            for e in self.events:
                evKeyNames.add(e['EventName'])


            if 'ProtocolFocus' in evKeyNames:
                self.zm = piecewiseMapping.GeneratePMFromEventList(self.events, self.mdh, self.mdh.getEntry('StartTime'), self.mdh.getEntry('Protocol.PiezoStartPos'))
                self.z_focus = 1.e3*self.zm(self.selectedDataSource['t'])
                #self.elv.SetCharts([('Focus [um]', self.zm, 'ProtocolFocus'),])


                self.selectedDataSource.z_focus = self.z_focus
                self.selectedDataSource.setMapping('focus', 'z_focus')

            if 'ScannerXPos' in evKeyNames:
                x0 = 0
                if 'Positioning.Stage_X' in self.mdh.getEntryNames():
                    x0 = self.mdh.getEntry('Positioning.Stage_X')
                self.xm = piecewiseMapping.GeneratePMFromEventList(self.elv.eventSource, self.mdh, self.mdh.getEntry('StartTime'), x0, 'ScannerXPos', 0)


                self.selectedDataSource.scan_x = 1.e3*self.xm(self.selectedDataSource['t']-.01)
                self.selectedDataSource.setMapping('ScannerX', 'scan_x')
                self.selectedDataSource.setMapping('x', 'x + scan_x')

            if 'ScannerYPos' in evKeyNames:
                y0 = 0
                if 'Positioning.Stage_Y' in self.mdh.getEntryNames():
                    y0 = self.mdh.getEntry('Positioning.Stage_Y')
                self.ym = piecewiseMapping.GeneratePMFromEventList(self.elv.eventSource, self.mdh, self.mdh.getEntry('StartTime'), y0, 'ScannerYPos', 0)


                self.selectedDataSource.scan_y = 1.e3*self.ym(self.selectedDataSource['t']-.01)
                self.selectedDataSource.setMapping('ScannerY', 'scan_y')
                self.selectedDataSource.setMapping('y', 'y + scan_y')

            if 'ScannerXPos' in evKeyNames or 'ScannerYPos' in evKeyNames:
                self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)



        if not 'foreShort' in dir(self.selectedDataSource):
            self.selectedDataSource.foreShort = 1.

        if not 'focus' in self.selectedDataSource.mappings.keys():
            self.selectedDataSource.focus= np.zeros(self.selectedDataSource['x'].shape)

        if 'fitResults_z0' in self.selectedDataSource.keys():
            self.selectedDataSource.setMapping('z', 'fitResults_z0 + foreShort*focus')
        else:
            self.selectedDataSource.setMapping('z', 'foreShort*focus')

        #if we've done a 3d fit
        #print self.selectedDataSource.keys()
        for k in self.filterKeys.keys():
            if not k in self.selectedDataSource.keys():
                self.filterKeys.pop(k)

        #print self.filterKeys
        self.RegenFilter()


        if 'Sample.Labelling' in self.mdh.getEntryNames():
            self.SpecFromMetadata()


    def SpecFromMetadata(self):
        labels = self.mdh.getEntry('Sample.Labelling')

        for structure, dye in labels:
            ratio = dyeRatios.getRatio(dye, self.mdh)

            if not ratio == None:
                self.fluorSpecies[structure] = ratio
                self.fluorSpeciesDyes[structure] = dye
                self.mapping.setMapping('p_%s' % structure, '(1.0/(ColourNorm*2*numpy.pi*fitError_Ag*fitError_Ar))*exp(-(fitResults_Ag - %f*A)**2/(2*fitError_Ag**2) - (fitResults_Ar - %f*A)**2/(2*fitError_Ar**2))' % (ratio, 1-ratio))







