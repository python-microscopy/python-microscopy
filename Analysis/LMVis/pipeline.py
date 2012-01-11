from PYME.Analysis.LMVis import inpFilt

from PYME.Analysis import piecewiseMapping

#import time
import numpy as np
import scipy.special

from PYME.Acquire import MetaDataHandler

from PYME.Analysis.LMVis.visHelpers import ImageBounds
from PYME.Analysis.LMVis import dyeRatios
import os
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


    def OpenFile(self, filename, **kwargs):
        self.dataSources = []
        if 'zm' in dir(self):
            del self.zm
        self.filter = None
        self.mapping = None
        self.colourFilter = None
        self.filename = filename
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        
        if os.path.splitext(filename)[1] == '.h5r':
            try:
                self.selectedDataSource = inpFilt.h5rSource(filename)
                self.dataSources.append(self.selectedDataSource)

                self.filesToClose.append(self.selectedDataSource.h5f)

                if 'DriftResults' in self.selectedDataSource.h5f.root:
                    self.dataSources.append(inpFilt.h5rDSource(self.selectedDataSource.h5f))

                    if len(self.selectedDataSource['x']) == 0:
                        self.selectedDataSource = self.dataSources[-1]

            except:
                self.selectedDataSource = inpFilt.h5rDSource(filename)
                self.dataSources.append(self.selectedDataSource)
                
                self.filesToClose.append(self.selectedDataSource.h5f)

            #once we get around to storing the some metadata with the results
            if 'MetaData' in self.selectedDataSource.h5f.root:
                self.mdh = MetaDataHandler.HDFMDHandler(self.selectedDataSource.h5f)

                

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
            elif not 'y' in self.selectedDataSource.keys():
                #    print 'foo'
                self.selectedDataSource = inpFilt.mappingFilter(self.selectedDataSource, y='10*t')
                self.dataSources.append(self.selectedDataSource)
            else:
                self.selectedDataSource = inpFilt.mappingFilter(self.selectedDataSource)
                self.dataSources.append(self.selectedDataSource)

                if 'mdh' in dir(self):                    
                    if 'Camera.ROIWidth' in self.mdh.getEntryNames():
                        x0 = 0
                        y0 = 0

                        x1 = self.mdh.getEntry('Camera.ROIWidth')*1e3*self.mdh.getEntry('voxelsize.x')
                        y1 = self.mdh.getEntry('Camera.ROIHeight')*1e3*self.mdh.getEntry('voxelsize.y')

                        if 'Splitter' in self.mdh.getEntry('Analysis.FitModule'):
                            y1 = y1/2

                        self.imageBounds = ImageBounds(x0, y0, x1, y1)
                    else:
                        #print 'foo', self.selectedDataSource['x'].max()
                        self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)

                else:
                    self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)    

            if ('Events' in self.selectedDataSource.resultsSource.h5f.root) and ('StartTime' in self.mdh.keys()):
                self.events = self.selectedDataSource.resultsSource.h5f.root.Events[:]


                evKeyNames = set()
                for e in self.events:
                    evKeyNames.add(e['EventName'])

                if 'ProtocolFocus' in evKeyNames:
                    self.zm = piecewiseMapping.GeneratePMFromEventList(self.events, self.mdh, self.mdh.getEntry('StartTime'), self.mdh.getEntry('Protocol.PiezoStartPos'))
                    self.z_focus = 1.e3*self.zm(self.selectedDataSource['t'])

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

                if 'ScannerXPos' in evKeyNames or 'ScannerYPos' in self.elv.evKeyNames:
                    self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)


            if not 'foreShort' in dir(self.selectedDataSource):
                self.selectedDataSource.foreShort = 1.

            if not 'focus' in self.selectedDataSource.mappings.keys():
                self.selectedDataSource.focus= np.zeros(self.selectedDataSource['x'].shape)
                
            if 'fitResults_z0' in self.selectedDataSource.keys():
                self.selectedDataSource.setMapping('z', 'fitResults_z0 + foreShort*focus')
            else:
                self.selectedDataSource.setMapping('z', 'foreShort*focus')

                        
        elif os.path.splitext(filename)[1] == '.mat': #matlab file
            ds = inpFilt.matfileSource(filename, kwargs['FieldNames'], kwargs['VarName'])
            self.selectedDataSource = ds
            self.dataSources.append(ds)

        else: #assume it's a text file
            ds = inpFilt.textfileSource(filename, kwargs['FieldNames'])
            self.selectedDataSource = ds
            self.dataSources.append(ds)

        #Retrieve or estimate image bounds
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
        
        
        #remove any keys from the filter which are not present in the data
        for k in self.filterKeys.keys():
            if not k in self.selectedDataSource.keys():
                self.filterKeys.pop(k)

        
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







