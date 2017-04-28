#!/usr/bin/python

###############
# pipeline.py
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
from PYME.IO import tabular
from PYME.IO.image import ImageBounds
from PYME.LMVis import dyeRatios
from PYME.LMVis import statusLog
from PYME.LMVis import renderers
from PYME.LMVis.triBlobs import BlobSettings

from PYME.Analysis import piecewiseMapping
from PYME.IO import MetaDataHandler

#from traits.api import HasTraits
#from traitsui.api import View
from PYME.recipes.base import ModuleCollection

import numpy as np
import scipy.special
import os

import dispatch

from PYME.Analysis.BleachProfile.kinModels import getPhotonNums

import logging
logger = logging.getLogger(__name__)

def _processPriSplit(ds):
    """set mappings ascociated with the use of a splitter"""

    ds.setMapping('gFrac', 'fitResults_ratio')
    ds.setMapping('error_gFrac', 'fitError_ratio')

    ds.setMapping('fitResults_Ag', 'gFrac*A')
    ds.setMapping('fitResults_Ar', '(1.0 - gFrac)*A + error_gFrac*A')
    ds.setMapping('fitError_Ag', 'gFrac*fitError_A + error_gFrac*A')
    ds.setMapping('fitError_Ar', '(1.0 - gFrac)*fitError_A')
    #ds.setMapping('fitError_Ag', '1*sqrt(fitResults_Ag/1e-3)')
    #ds.setMapping('fitError_Ar', '1*sqrt(fitResults_Ar/1e-3)')

    sg = ds['fitError_Ag']
    sr = ds['fitError_Ar']
    g = ds['fitResults_Ag']
    r = ds['fitResults_Ar']
    I = ds['A']

    colNorm = np.sqrt(2 * np.pi) * sg * sr / (2 * np.sqrt(sg ** 2 + sr ** 2) * I) * (
        scipy.special.erf((sg ** 2 * r + sr ** 2 * (I - g)) / (np.sqrt(2) * sg * sr * np.sqrt(sg ** 2 + sr ** 2)))
        - scipy.special.erf((sg ** 2 * (r - I) - sr ** 2 * g) / (np.sqrt(2) * sg * sr * np.sqrt(sg ** 2 + sr ** 2))))

    colNorm /= (sg * sr)

    ds.addColumn('ColourNorm', colNorm)


def _processSplitter(ds):
    """set mappings ascociated with the use of a splitter"""

    #ds.gF_zcorr = 0
    ds.setMapping('A', 'fitResults_Ag + fitResults_Ar')
    if 'fitResults_z0' in ds.keys():
        ds.addVariable('gF_zcorr', 0)
        ds.setMapping('gFrac', 'fitResults_Ag/(fitResults_Ag + fitResults_Ar) + gF_zcorr*fitResults_z0')
    else:
        ds.setMapping('gFrac', 'fitResults_Ag/(fitResults_Ag + fitResults_Ar)')

    if 'fitError_Ag' in ds.keys():
        ds.setMapping('error_gFrac',
                      'sqrt((fitError_Ag/fitResults_Ag)**2 + (fitError_Ag**2 + fitError_Ar**2)/(fitResults_Ag + fitResults_Ar)**2)*fitResults_Ag/(fitResults_Ag + fitResults_Ar)')
    else:
        ds.setMapping('error_gFrac', '0*x + 0.01')
        ds.setMapping('fitError_Ag', '1*sqrt(fitResults_Ag/1)')
        ds.setMapping('fitError_Ar', '1*sqrt(fitResults_Ar/1)')

    sg = ds['fitError_Ag']
    sr = ds['fitError_Ar']
    g = ds['fitResults_Ag']
    r = ds['fitResults_Ar']
    I = ds['A']

    colNorm = np.sqrt(2 * np.pi) * sg * sr / (2 * np.sqrt(sg ** 2 + sr ** 2) * I) * (
        scipy.special.erf((sg ** 2 * r + sr ** 2 * (I - g)) / (np.sqrt(2) * sg * sr * np.sqrt(sg ** 2 + sr ** 2)))
        - scipy.special.erf((sg ** 2 * (r - I) - sr ** 2 * g) / (np.sqrt(2) * sg * sr * np.sqrt(sg ** 2 + sr ** 2))))

    ds.addColumn('ColourNorm', colNorm)


def _add_eventvars_to_ds(ds, ev_mappings):
    zm = ev_mappings.get('zm', None)
    if zm:
        z_focus = 1.e3 * zm(ds['t'])
        ds.addColumn('focus', z_focus)

    xm = ev_mappings.get('xm', None)
    if xm:
        scan_x = 1.e3 * xm(ds['t'] - .01)
        ds.addColumn('scanx', scan_x)
        ds.setMapping('x', 'x + scanx')

    ym = ev_mappings.get('ym', None)
    if ym:
        scan_y = 1.e3 * ym(ds['t'] - .01)
        ds.addColumn('scany', scan_y)
        ds.setMapping('y', 'y + scany')

    driftx = ev_mappings.get('driftx', None)
    drifty = ev_mappings.get('drifty', None)
    driftz = ev_mappings.get('driftz', None)
    if driftx:
        ds.addColumn('driftx', driftx(ds['t'] - .01))
        ds.addColumn('drifty', drifty(ds['t'] - .01))
        ds.addColumn('driftz', driftz(ds['t'] - .01))


def _add_missing_ds_keys(mapped_ds, ev_mappings={}):
    """
    VisGUI, and various rendering and postprocessing commands rely on having certain parameters defined or re-mapped
    within a data source. Take care of these here.

    Parameters
    ----------
    mapped_ds

    """
    _add_eventvars_to_ds(mapped_ds, ev_mappings)

    #handle special cases which get detected by looking for the presence or
    #absence of certain variables in the data.
    if 'fitResults_Ag' in mapped_ds.keys():
        #if we used the splitter set up a number of mappings e.g. total amplitude and ratio
        _processSplitter(mapped_ds)

    if 'fitResults_ratio' in mapped_ds.keys():
        #if we used the splitter set up a number of mappings e.g. total amplitude and ratio
        _processPriSplit(mapped_ds)

    if 'fitResults_sigxl' in mapped_ds.keys():
        #fast, quickpalm like astigmatic fitting
        mapped_ds.setMapping('sig', 'fitResults_sigxl + fitResults_sigyu')
        mapped_ds.setMapping('sig_d', 'fitResults_sigxl - fitResults_sigyu')

        mapped_ds.addVariable('dsigd_dz', -30.)
        mapped_ds.setMapping('fitResults_z0', 'dsigd_dz*sig_d')
    if not 'y' in mapped_ds.keys():
        mapped_ds.setMapping('y', '10*t')

    #set up correction for foreshortening and z focus stepping
    if not 'foreShort' in dir(mapped_ds):
        mapped_ds.addVariable('foreShort', 1.)

    if not 'focus' in mapped_ds.keys():
        #set up a dummy focus variable if not already present
        mapped_ds.setMapping('focus', '0*x')

    if not 'z' in mapped_ds.keys():
        if 'fitResults_z0' in mapped_ds.keys():
            mapped_ds.setMapping('z', 'fitResults_z0 + foreShort*focus')
        elif 'astigZ' in mapped_ds.keys():
            mapped_ds.setMapping('z', 'astigZ + foreShort*focus')
        else:
            mapped_ds.setMapping('z', 'foreShort*focus')

    if not 'A' in mapped_ds.keys() and 'fitResults_photons' in mapped_ds.keys():
        mapped_ds.setMapping('A', 'fitResults_photons')



def _processEvents(ds, events, mdh):
    """Read data from events table and translate it into mappings for,
    e.g. z position"""

    eventCharts = []
    ev_mappings = {}

    if not events is None:
        evKeyNames = set()
        for e in events:
            evKeyNames.add(e['EventName'])

        if 'ProtocolFocus' in evKeyNames:
            zm = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh['StartTime'], mdh['Protocol.PiezoStartPos'])
            ev_mappings['zm'] = zm
            eventCharts.append(('Focus [um]', zm, 'ProtocolFocus'))

        if 'ScannerXPos' in evKeyNames:
            x0 = 0
            if 'Positioning.Stage_X' in mdh.getEntryNames():
                x0 = mdh.getEntry('Positioning.Stage_X')
            xm = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh['StartTime'], x0, 'ScannerXPos', 0)
            ev_mappings['xm'] = xm
            eventCharts.append(('XPos [um]', xm, 'ScannerXPos'))

        if 'ScannerYPos' in evKeyNames:
            y0 = 0
            if 'Positioning.Stage_Y' in mdh.getEntryNames():
                y0 = mdh.getEntry('Positioning.Stage_Y')
            ym = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh.getEntry('StartTime'), y0, 'ScannerYPos', 0)
            ev_mappings['ym'] = ym
            eventCharts.append(('YPos [um]', ym, 'ScannerYPos'))

        if 'ShiftMeasure' in evKeyNames:
            driftx = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh.getEntry('StartTime'), 0, 'ShiftMeasure',
                                                              0)
            drifty = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh.getEntry('StartTime'), 0, 'ShiftMeasure',
                                                              1)
            driftz = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh.getEntry('StartTime'), 0, 'ShiftMeasure',
                                                              2)

            ev_mappings['driftx'] = driftx
            ev_mappings['drifty'] = drifty
            ev_mappings['driftz'] = driftz

            eventCharts.append(('X Drift [px]', driftx, 'ShiftMeasure'))
            eventCharts.append(('Y Drift [px]', drifty, 'ShiftMeasure'))
            eventCharts.append(('Z Drift [px]', driftz, 'ShiftMeasure'))

            #self.eventCharts = eventCharts
            #self.ev_mappings = ev_mappings
    elif all(k in mdh.keys() for k in ['StackSettings.FramesPerStep', 'StackSettings.StepSize',
                                       'StackSettings.NumSteps', 'StackSettings.NumCycles']):
        # if we dont have events file, see if we can use metadata to spoof focus
        print('No events found, spoofing focus position using StackSettings metadata')
        frames = np.arange(0, mdh['StackSettings.FramesPerStep'] * mdh['StackSettings.NumSteps'] * mdh[
            'StackSettings.NumCycles'], mdh['StackSettings.FramesPerStep'])
        position = np.arange(mdh.getOrDefault('Protocol.PiezoStartPos', 0),
                             mdh.getOrDefault('Protocol.PiezoStartPos', 0) + mdh['StackSettings.NumSteps'] * mdh[
                                 'StackSettings.StepSize'], mdh['StackSettings.StepSize'])
        position = np.tile(position, mdh['StackSettings.NumCycles'])

        zm = piecewiseMapping.piecewiseMap(0, frames, position, mdh['Camera.CycleTime'], xIsSecs=False)
        ev_mappings['zm'] = zm
        eventCharts.append(('Focus [um]', zm, 'ProtocolFocus'))

    return ev_mappings, eventCharts

class Pipeline:
    def __init__(self, filename=None, visFr=None):
        self.recipe = ModuleCollection(execute_on_invalidation=True)

        self.selectedDataSourceKey = None
        self.filterKeys = {'error_x': (0,30), 'error_y':(0,30),'A':(5,20000), 'sig' : (95, 200)}

        self.filter = None
        self.mapping = None
        self.colourFilter = None
        self.events = None

        self.fluorSpecies = {}
        self.fluorSpeciesDyes = {}

        self.blobSettings = BlobSettings()
        self.objects = None

        self.imageBounds = ImageBounds(0,0,0,0)
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        
        self.Triangles = None
        self.edb = None
        self.Quads = None
        self.GeneratedMeasures = {}
        
        self.QTGoalPixelSize = 5
        
        self.filesToClose = []

        self.ev_mappings = {}

        #define a signal which a GUI can hook if the pipeline is rebuilt (i.e. the output changes)
        self.onRebuild = dispatch.Signal()

        #a cached list of our keys to be used to decide whether to fire a keys changed signal
        self._keys = None
        #define a signal which can be hooked if the pipeline keys have changed
        self.onKeysChanged = dispatch.Signal()
        
        self.ready = False
        self.visFr = visFr

        if not filename is None:
            self.OpenFile(filename)
            
        #renderers.renderMetadataProviders.append(self.SaveMetadata)
            
    def __getitem__(self, keys):
        """gets values from the 'tail' of the pipeline (ie the colourFilter)"""
        
        return self.colourFilter[keys]

    def keys(self):
        return self.colourFilter.keys()

    @property
    def chromaticShifts(self):
        return self.colourFilter.chromaticShifts

    @property
    def dataSources(self):
        return self.recipe.namespace

    @property
    def selectedDataSource(self):
        """

        The currently selected data source (an instance of tabular.inputFilter derived class)

        """
        if self.selectedDataSourceKey is None:
            return None
        else:
            return self.dataSources[self.selectedDataSourceKey]

    def selectDataSource(self, dskey):
        """
        Set the currently selected data source

        Parameters
        ----------
        dskey : string
            The data source name

        """
        if not dskey in self.dataSources.keys():
            raise KeyError('Data Source "%s" not found' % dskey)

        self.selectedDataSourceKey = dskey

        #remove any keys from the filter which are not present in the data
        for k in self.filterKeys.keys():
            if not k in self.selectedDataSource.keys():
                self.filterKeys.pop(k)

        self.Rebuild()

    def addColumn(self, name, values, default = 0):
        """
        Adds a column to the currently selected data source. Attempts to guess whether the size matches the input or
        the output, and adds padding values appropriately if it matches the output.

        Parameters
        ----------
        name : str
            The column name
        values : array like
            The values
        default : float
            The default value to pad with if we've given an output-sized array

        """

        ds_len = len(self.selectedDataSource[self.selectedDataSource.keys()[0]])
        val_len = len(values)

        if val_len == ds_len:
            #length matches the length of our input data source - do a simple add
            self.selectedDataSource.addColumn(name, values)
        elif val_len == len(self[self.keys()[0]]):

            col_index = self.colourFilter.index

            idx = np.copy(self.filter.Index)
            idx[self.filter.Index] = col_index

            ds_vals = np.zeros(ds_len) + default
            ds_vals[idx] = np.array(values)

            self.selectedDataSource.addColumn(name, ds_vals)
        else:
            raise RuntimeError("Length of new column doesn't match either the input or output lengths")

    def addDataSource(self, dskey, ds, add_missing_vars=True):
        """
        Add a new data source

        Parameters
        ----------
        dskey : str
            The name of the new data source
        ds : an tabular.inputFilter derived class
            The new data source

        """
        #check that we have a suitable object - note that this could potentially be relaxed
        assert isinstance(ds, tabular.TabularBase)

        if not isinstance(ds, tabular.mappingFilter):
            #wrap with a mapping filter
            ds = tabular.mappingFilter(ds)

        #add keys which might not already be defined
        if add_missing_vars:
            _add_missing_ds_keys(ds,self.ev_mappings)

        if getattr(ds, 'mdh', None) is None:
            try:
                ds.mdh = self.mdh
            except AttributeError:
                logger.error('No metadata defined in pipeline')
                pass

        self.dataSources[dskey] = ds



    def Rebuild(self, **kwargs):
        """
        Rebuild the pipeline. Called when the selected data source is changed/modified and/or the filter is changed.

        """
        for s in self.dataSources.values():
            if 'setMapping' in dir(s):
                #keep raw measurements available
                s.setMapping('x_raw', 'x')
                s.setMapping('y_raw', 'y')
                
                if 'z' in  s.keys():
                    s.setMapping('z_raw', 'z')
                
        if not self.selectedDataSource is None:
            #we can recycle the mapping object
            if self.mapping:
                self.mapping.resultsSource = self.selectedDataSource
            else:
                self.mapping = tabular.mappingFilter(self.selectedDataSource)

            #the filter, however needs to be re-generated with new keys and or data source
            self.filter = tabular.resultsFilter(self.mapping, **self.filterKeys)

            #we can also recycle the colour filter
            if not self.colourFilter:
                self.colourFilter = tabular.colourFilter(self.filter)
            else:
                self.colourFilter.resultsSource = self.filter

            self._process_colour()
            
            self.ready = True

        self.ClearGenerated()
        
    def ClearGenerated(self):
        self.Triangles = None
        self.edb = None
        self.GeneratedMeasures = {}
        self.Quads = None

        self.onRebuild.send(sender=self)

        #check to see if any of the keys have changed - if so, fire a keys changed event so the GUI can update
        newKeys = self.keys()
        if not newKeys == self._keys:
            self.onKeysChanged.send(sender=self)
        
    def CloseFiles(self):
        while len(self.filesToClose) > 0:
            self.filesToClose.pop().close()

    def _ds_from_file(self, filename, **kwargs):
        """
        loads a data set from a file

        Parameters
        ----------
        filename : str
        kwargs : any additional arguments (see OpenFile)

        Returns
        -------

        ds : the dataset

        """

        if os.path.splitext(filename)[1] == '.h5r':
            try:
                ds = tabular.h5rSource(filename)
                self.filesToClose.append(ds.h5f)

                if 'DriftResults' in ds.h5f.root:
                    driftDS = tabular.h5rDSource(ds.h5f)
                    self.driftInputMapping = tabular.mappingFilter(driftDS)
                    self.dataSources['Fiducials'] = self.driftInputMapping

                    if len(ds['x']) == 0:
                        self.selectDataSource('Fiducials')

            except: #fallback to catch series that only have drift data
                ds = tabular.h5rDSource(filename)
                self.filesToClose.append(ds.h5f)

                self.driftInputMapping = tabular.mappingFilter(ds)
                self.dataSources['Fiducials'] = self.driftInputMapping
                #self.selectDataSource('Fiducials')

            #catch really old files which don't have any metadata
            if 'MetaData' in ds.h5f.root:
                self.mdh.copyEntriesFrom(MetaDataHandler.HDFMDHandler(ds.h5f))

            if ('Events' in ds.h5f.root) and ('StartTime' in self.mdh.keys()):
                self.events = ds.h5f.root.Events[:]

        elif filename.endswith('.hdf'):
            #recipe output - handles generically formatted .h5
            import tables
            h5f = tables.open_file(filename)

            for t in h5f.list_nodes('/'):
                if isinstance(t, tables.table.Table):
                    tab = tabular.hdfSource(h5f, t.name)
                    self.addDataSource(t.name, tab)
                        
                    if 'EventName' in t.description._v_names: #FIXME - we shouldn't have a special case here
                        self.events = t[:]  # this does not handle multiple events tables per hdf file

            if 'MetaData' in h5f.root:
                self.mdh.copyEntriesFrom(MetaDataHandler.HDFMDHandler(h5f))

            for dsname, ds_ in self.dataSources.items():
                #loop through tables until we get one which defines x. If no table defines x, take the last table to be added
                #TODO make this logic better.
                ds = ds_.resultsSource
                if 'x' in ds.keys():
                    break

        elif os.path.splitext(filename)[1] == '.mat': #matlab file
            ds = tabular.matfileSource(filename, kwargs['FieldNames'], kwargs['VarName'])

        elif os.path.splitext(filename)[1] == '.csv':
            #special case for csv files - tell np.loadtxt to use a comma rather than whitespace as a delimeter
            if 'SkipRows' in kwargs.keys():
                ds = tabular.textfileSource(filename, kwargs['FieldNames'], delimiter=',', skiprows=kwargs['SkipRows'])
            else:
                ds = tabular.textfileSource(filename, kwargs['FieldNames'], delimiter=',')

        else: #assume it's a tab (or other whitespace) delimited text file
            if 'SkipRows' in kwargs.keys():
                ds = tabular.textfileSource(filename, kwargs['FieldNames'], skiprows=kwargs['SkipRows'])
            else:
                ds = tabular.textfileSource(filename, kwargs['FieldNames'])

        return ds

    def OpenFile(self, filename= '', ds = None, **kwargs):
        """Open a file - accepts optional keyword arguments for use with files
        saved as .txt and .mat. These are:
            
            FieldNames: a list of names for the fields in the text file or
                        matlab variable.
            VarName:    the name of the variable in the .mat file which 
                        contains the data.
            SkipRows:   Number of header rows to skip for txt file data
            
            PixelSize:  Pixel size if not in nm
            
        """
        
        #close any files we had open previously
        while len(self.filesToClose) > 0:
            self.filesToClose.pop().close()
        
        #clear our state
        self.dataSources.clear()
        if 'zm' in dir(self):
            del self.zm
        self.filter = None
        self.mapping = None
        self.colourFilter = None
        self.events = None
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        
        self.filename = filename
        
        if ds is None:
            #load from file
            ds = self._ds_from_file(filename, **kwargs)

            
        #wrap the data source with a mapping so we can fiddle with things
        #e.g. combining z position and focus 
        mapped_ds = tabular.mappingFilter(ds)

        
        if 'PixelSize' in kwargs.keys():
            mapped_ds.addVariable('pixelSize', kwargs['PixelSize'])
            mapped_ds.setMapping('x', 'x*pixelSize')
            mapped_ds.setMapping('y', 'y*pixelSize')

        #extract information from any events
        self.ev_mappings, self.eventCharts = _processEvents(mapped_ds, self.events, self.mdh)

        #Retrieve or estimate image bounds
        if False:#'imgBounds' in kwargs.keys():
            self.imageBounds = kwargs['imgBounds']
        elif (not ('scanx' in mapped_ds.keys() or 'scany' in mapped_ds.keys())) and 'Camera.ROIWidth' in self.mdh.getEntryNames():
            self.imageBounds = ImageBounds.extractFromMetadata(self.mdh)
        else:
            self.imageBounds = ImageBounds.estimateFromSource(mapped_ds)

        #Fit module specific filter settings        
        if 'Analysis.FitModule' in self.mdh.getEntryNames():
            fitModule = self.mdh['Analysis.FitModule']
            
            #print 'fitModule = %s' % fitModule
            
            if 'Interp' in fitModule:
                self.filterKeys['A'] = (5, 100000)
            
            if 'LatGaussFitFR' in fitModule:
                mapped_ds.addColumn('nPhotons', getPhotonNums(mapped_ds, self.mdh))
                
            if fitModule == 'SplitterShiftEstFR':
                self.filterKeys['fitError_dx'] = (0,10)
                self.filterKeys['fitError_dy'] = (0,10)

        self._get_dye_ratios_from_metadata()

        self.addDataSource('Localizations', mapped_ds)
        self.selectDataSource('Localizations') #NB - this rebuilds the pipeline
        
        #self._process_colour()



    def _process_colour(self):
        """
        Locate any colour / channel information and munge it into a format that the colourFilter understands.

        We currently accept 3 ways of specifying channels:

         - ratiometric colour, where 'gFrac' is defined to be the ratio between our observation channels
         - defining a 'probe' column in the input data which gives a channel index for each point
         - specifying colour ranges in the metadata

         All of these get munged into the p_dye type entries that the colour filter needs.

        """
        #clear out old colour keys
        for k in self.mapping.mappings.keys():
            if k.startswith('p_'):
                self.mapping.mappings.pop(k)
        
        if 'gFrac' in self.selectedDataSource.keys():
            #ratiometric
            for structure, ratio in self.fluorSpecies.items():
                if not ratio is None:
                    self.mapping.setMapping('p_%s' % structure, 'exp(-(%f - gFrac)**2/(2*error_gFrac**2))/(error_gFrac*sqrt(2*numpy.pi))' % ratio)
        else:
            if 'probe' in self.mapping.keys():
                #non-ratiometric (i.e. sequential) colour
                #color channel is given in 'probe' column
                self.mapping.setMapping('ColourNorm', '1.0 + 0*probe')
    
                for i in range(int(self['probe'].min()), int(self['probe'].max() + 1)):
                    self.mapping.setMapping('p_chan%d' % i, '1.0*(probe == %d)' % i)
    
            nSeqCols = self.mdh.getOrDefault('Protocol.NumberSequentialColors', 1)
            if nSeqCols > 1:
                for i in range(nSeqCols):
                    self.mapping.setMapping('ColourNorm', '1.0 + 0*t')
                    cr = self.mdh['Protocol.ColorRange%d' % i]
                    self.mapping.setMapping('p_chan%d' % i, '(t>= %d)*(t<%d)' % cr)
                
        #self.ClearGenerated()


    def _get_dye_ratios_from_metadata(self):
        labels = self.mdh.getOrDefault('Sample.Labelling', [])

        for structure, dye in labels:
            ratio = dyeRatios.getRatio(dye, self.mdh)

            if not ratio is None:
                self.fluorSpecies[structure] = ratio
                self.fluorSpeciesDyes[structure] = dye
                #self.mapping.setMapping('p_%s' % structure, '(1.0/(ColourNorm*2*numpy.pi*fitError_Ag*fitError_Ar))*exp(-(fitResults_Ag - %f*A)**2/(2*fitError_Ag**2) - (fitResults_Ar - %f*A)**2/(2*fitError_Ar**2))' % (ratio, 1-ratio))
                #self.mapping.setMapping('p_%s' % structure, 'exp(-(%f - gFrac)**2/(2*error_gFrac**2))/(error_gFrac*sqrt(2*numpy.pi))' % ratio)
                

    def getNeighbourDists(self, forceRetriang = False):
        from PYME.LMVis import visHelpers
        
        if forceRetriang or not 'neighbourDistances' in self.GeneratedMeasures.keys():
            statNeigh = statusLog.StatusLogger("Calculating mean neighbour distances ...")
            self.GeneratedMeasures['neighbourDistances'] = np.array(visHelpers.calcNeighbourDists(self.getTriangles(forceRetriang)))
            
        return self.GeneratedMeasures['neighbourDistances']
        
    def getTriangles(self, recalc = False):
        from matplotlib import delaunay
        
        if self.Triangles is None or recalc:
            statTri = statusLog.StatusLogger("Generating Triangulation ...")
            self.Triangles = delaunay.Triangulation(self.colourFilter['x'] + .1*np.random.normal(size=len(self.colourFilter['x'])), self.colourFilter['y']+ .1*np.random.normal(size=len(self.colourFilter['x'])))
            
            #reset things which will have changed
            self.edb = None
            try:
                self.GeneratedMeasures.pop('neighbourDistances')
            except KeyError:
                pass
            
        return self.Triangles
        
    def getEdb(self):
        from PYME.Analysis.points.EdgeDB import edges
        if self.edb is None:
            self.edb = edges.EdgeDB(self.getTriangles())
            
        return self.edb
            
    def getBlobs(self):
        from PYME.Analysis.points.EdgeDB import edges
        
        tri = self.getTriangles()        
        edb = self.getEdb()
        
        
        if self.blobSettings.jittering == 0:
            self.objIndices = edges.objectIndices(edb.segment(self.blobSettings.distThreshold), self.blobSettings.minSize)
            self.objects = [np.vstack((tri.x[oi], tri.y[oi])).T for oi in self.objIndices]
        else:
            from matplotlib import delaunay
            
            ndists = self.getNeighbourDists()
            
            x_ = np.hstack([self['x'] + 0.5*ndists*np.random.normal(size=ndists.size) for i in range(self.blobSettings.jittering)])
            y_ = np.hstack([self['y'] + 0.5*ndists*np.random.normal(size=ndists.size) for i in range(self.blobSettings.jittering)])

            T = delaunay.Triangulation(x_, y_)
            edb = edges.EdgeDB(T)
            
            objIndices = edges.objectIndices(edb.segment(self.blobSettings.distThreshold), self.blobSettings.minSize)
            self.objects = [np.vstack((T.x[oi], T.y[oi])).T for oi in objIndices]
            
        return self.objects, self.blobSettings.distThreshold
        
    def GenQuads(self):
        from PYME.Analysis.points.QuadTree import pointQT
        
        di = max(self.imageBounds.x1 - self.imageBounds.x0, 
                 self.imageBounds.y1 - self.imageBounds.y0)

        numPixels = di/self.QTGoalPixelSize

        di = self.QTGoalPixelSize*2**np.ceil(np.log2(numPixels))

        
        self.Quads = pointQT.qtRoot(self.imageBounds.x0, self.imageBounds.x0+di, 
                                    self.imageBounds.y0, self.imageBounds.y0 + di)

        for xi, yi in zip(self['x'],self['y']):
            self.Quads.insert(pointQT.qtRec(xi,yi, None))
            
    def measureObjects(self):
        from PYME.Analysis.points import objectMeasure
        
        self.objectMeasures = objectMeasure.measureObjects(self.objects, self.objThreshold)
        
        return self.objectMeasures
        
    def save_txt(self, outFile, keys=None):
        if keys is None:
            keys = self.keys()

        #nRecords = len(ds[keys[0]])
    
        of = open(outFile, 'w')
    
        of.write('#' + '\t'.join(['%s' % k for k in keys]) + '\n')
    
        for row in zip(*[self[k] for k in keys]):
            of.write('\t'.join(['%e' % c for c in row]) + '\n')
    
        of.close()
        
    def save_hdf(self, filename):
        self.colourFilter.to_hdf(filename, tablename='Localizations', metadata=self.mdh)
        
    def toDataFrame(self, keys=None):
        import pandas as pd
        if keys is None:
            keys = self.keys()
        
        d = {k: self[k] for k in keys}
        
        return pd.DataFrame(d)
        
    @property
    def dtypes(self):
        return {k: str(self[k, :2].dtype) for k in self.keys()}
    








